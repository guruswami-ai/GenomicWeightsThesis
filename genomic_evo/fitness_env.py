"""
Brax-based fitness environment for the Genomic Weight Thesis experiment.

Uses JIT-compiled rollouts with weights as traced inputs for performance.
Achieves ~4000+ steps/sec on M3 Ultra vs ~50 steps/sec without JIT.
"""
import os
# Force CPU before JAX import - MJX doesn't support Metal
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from brax import envs
from brax.envs.wrappers.training import wrap
from phenotype_forward import (
    flat_forward, hierarchical_forward, topological_forward,
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, NUM_BLOCKS, BLOCK_DIM, GRAPH_NODES
)

# Environment config
ENV_NAME = "ant"
BRAX_BACKEND = "generalized"  # Works on CPU without MJX
EPISODE_LENGTH = 200

# Cache for environment (avoid recreation overhead)
_ENV_CACHE = None


def _get_env():
    """Get cached Brax environment."""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        env = envs.get_environment(ENV_NAME, backend=BRAX_BACKEND)
        _ENV_CACHE = wrap(env, episode_length=EPISODE_LENGTH)
    return _ENV_CACHE


# =============================================================================
# JIT-compiled rollout functions for each strategy
# These compile ONCE and reuse for all individuals with same-shaped weights
# =============================================================================

# Pre-JIT the environment step function
_ENV_STEP_FN = None

def _get_env_step():
    """Get JIT-compiled environment step function."""
    global _ENV_STEP_FN
    if _ENV_STEP_FN is None:
        env = _get_env()
        _ENV_STEP_FN = jax.jit(env.step)
    return _ENV_STEP_FN


def _make_rollout_flat(env_step_fn):
    """Create JIT-compiled flat rollout with captured env step."""
    @partial(jit, static_argnums=(3,))
    def rollout(env_state, rng, weights, episode_length):
        def step_fn(carry, _):
            state, key = carry
            obs = state.obs[0]
            action = flat_forward(obs, weights)
            action = jnp.clip(action, -1.0, 1.0)
            action = jnp.expand_dims(action, 0)
            next_state = env_step_fn(state, action)
            return (next_state, key), next_state.reward[0]

        (_, _), rewards = jax.lax.scan(
            step_fn, (env_state, rng), None, length=episode_length
        )
        return jnp.sum(rewards)
    return rollout


def _make_rollout_hierarchical(env_step_fn):
    """Create JIT-compiled hierarchical rollout with captured env step."""
    @partial(jit, static_argnums=(4,))
    def rollout(env_state, rng, blocks, projection, episode_length):
        def step_fn(carry, _):
            state, key = carry
            obs = state.obs[0]
            action = hierarchical_forward(obs, blocks, projection)
            action = jnp.clip(action, -1.0, 1.0)
            action = jnp.expand_dims(action, 0)
            next_state = env_step_fn(state, action)
            return (next_state, key), next_state.reward[0]

        (_, _), rewards = jax.lax.scan(
            step_fn, (env_state, rng), None, length=episode_length
        )
        return jnp.sum(rewards)
    return rollout


def _make_rollout_topological(env_step_fn):
    """Create JIT-compiled topological rollout with captured env step."""
    @partial(jit, static_argnums=(4,))
    def rollout(env_state, rng, adjacency, projection, episode_length):
        def step_fn(carry, _):
            state, key = carry
            obs = state.obs[0]
            action = topological_forward(obs, adjacency, projection)
            action = jnp.clip(action, -1.0, 1.0)
            action = jnp.expand_dims(action, 0)
            next_state = env_step_fn(state, action)
            return (next_state, key), next_state.reward[0]

        (_, _), rewards = jax.lax.scan(
            step_fn, (env_state, rng), None, length=episode_length
        )
        return jnp.sum(rewards)
    return rollout


# Cached rollout functions (lazily initialized)
_ROLLOUT_FLAT = None
_ROLLOUT_HIERARCHICAL = None
_ROLLOUT_TOPOLOGICAL = None


def _get_rollout_fn(strategy):
    """Get cached JIT-compiled rollout function for strategy."""
    global _ROLLOUT_FLAT, _ROLLOUT_HIERARCHICAL, _ROLLOUT_TOPOLOGICAL

    env_step = _get_env_step()

    if strategy == 'flat':
        if _ROLLOUT_FLAT is None:
            _ROLLOUT_FLAT = _make_rollout_flat(env_step)
        return _ROLLOUT_FLAT
    elif strategy == 'hierarchical':
        if _ROLLOUT_HIERARCHICAL is None:
            _ROLLOUT_HIERARCHICAL = _make_rollout_hierarchical(env_step)
        return _ROLLOUT_HIERARCHICAL
    elif strategy == 'topological':
        if _ROLLOUT_TOPOLOGICAL is None:
            _ROLLOUT_TOPOLOGICAL = _make_rollout_topological(env_step)
        return _ROLLOUT_TOPOLOGICAL
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# Public API
# =============================================================================

def brax_ant_fitness_jit(phenotype_data: dict, num_rollouts: int = 2) -> float:
    """
    Evaluate phenotype on Brax Ant using JIT-compiled rollouts.

    This is the fast version that compiles once per strategy and reuses
    the compiled function for all individuals.

    Args:
        phenotype_data: Dict with 'strategy' and strategy-specific weight arrays
        num_rollouts: Number of rollouts to average

    Returns:
        Mean fitness (reward) across rollouts
    """
    env = _get_env()
    rng = jax.random.PRNGKey(0)
    strategy = phenotype_data['strategy']
    rollout_fn = _get_rollout_fn(strategy)

    rewards = []
    for i in range(num_rollouts):
        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        state = env.reset(jax.random.split(reset_rng, 1))

        if strategy == 'flat':
            weights = jnp.asarray(phenotype_data['weights']).squeeze()
            reward = rollout_fn(state, rollout_rng, weights, EPISODE_LENGTH)

        elif strategy == 'hierarchical':
            blocks = jnp.asarray(phenotype_data['blocks'])
            if blocks.ndim == 1:
                block_size = blocks.shape[0] // NUM_BLOCKS
                blocks = blocks.reshape(NUM_BLOCKS, block_size)
            projection = jnp.asarray(phenotype_data['projection']).squeeze()
            reward = rollout_fn(state, rollout_rng, blocks, projection, EPISODE_LENGTH)

        elif strategy == 'topological':
            adjacency = jnp.asarray(phenotype_data['adjacency']).squeeze()
            projection = jnp.asarray(phenotype_data['projection']).squeeze()
            reward = rollout_fn(state, rollout_rng, adjacency, projection, EPISODE_LENGTH)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        rewards.append(float(reward))

    return sum(rewards) / len(rewards)


def brax_ant_fitness(phenotype_net, phenotype_data=None, num_rollouts=2):
    """
    Legacy API wrapper - converts phenotype_net call to JIT version.

    For backward compatibility with code that passes phenotype_net functions.
    New code should use brax_ant_fitness_jit directly with phenotype_data.
    """
    if phenotype_data is not None:
        return brax_ant_fitness_jit(phenotype_data, num_rollouts)

    # Fallback: run without JIT (slow)
    env = _get_env()
    rng = jax.random.PRNGKey(0)
    rewards = []

    for i in range(num_rollouts):
        rng, rollout_rng = jax.random.split(rng)
        state = env.reset(jax.random.split(rollout_rng, 1))
        total_reward = 0.0

        for _ in range(EPISODE_LENGTH):
            obs = state.obs[0]
            action = phenotype_net(obs)
            action = jnp.clip(action, -1.0, 1.0)
            action = jnp.expand_dims(action, 0)
            state = env.step(state, action)
            total_reward += float(state.reward[0])

        rewards.append(total_reward)

    return sum(rewards) / len(rewards)


# Alias for backward compatibility
predator_avoidance_fitness = brax_ant_fitness
