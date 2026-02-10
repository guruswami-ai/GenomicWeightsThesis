"""
Batched Brax fitness environment for CPU (Apple Silicon / x86).

Uses jax.vmap to evaluate entire populations in parallel.
On M3 Ultra with unified memory, this leverages all 24 CPU cores efficiently.
Expected performance: 10,000-50,000 steps/sec on M3 Ultra.

Supports multiple environments via the env_configs registry.
"""
import os
# Force CPU before JAX import
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from brax import envs
from brax.envs.wrappers.training import wrap

from phenotype_forward import flat_forward, hierarchical_forward, topological_forward
from env_configs import get_config, EnvConfig

# Per-environment caches
_ENV_CACHE = {}
_ENV_STEP_CACHE = {}
_BATCHED_EVAL_CACHE = {}


def _get_env(env_name: str) -> tuple:
    """Get cached Brax environment and config."""
    if env_name not in _ENV_CACHE:
        config = get_config(env_name)
        env = envs.get_environment(config.env_name, backend=config.backend)
        _ENV_CACHE[env_name] = (wrap(env, episode_length=config.episode_length), config)
    return _ENV_CACHE[env_name]


def _get_env_step(env_name: str):
    """Get JIT-compiled environment step function."""
    if env_name not in _ENV_STEP_CACHE:
        env, _ = _get_env(env_name)
        _ENV_STEP_CACHE[env_name] = jax.jit(env.step)
    return _ENV_STEP_CACHE[env_name]


# =============================================================================
# Single rollout functions (will be vmapped)
# =============================================================================

def _make_single_rollout_flat(config: EnvConfig):
    """Create single rollout function for flat strategy with given config."""
    episode_length = config.episode_length
    obs_dim = config.obs_dim
    action_dim = config.action_dim
    hidden_dim = config.hidden_dim

    def _single_rollout(env_step_fn, env_state, rng, weights):
        def step_fn(carry, _):
            state, key = carry
            obs = state.obs[0]
            action = flat_forward(obs, weights,
                                  obs_dim=obs_dim,
                                  action_dim=action_dim,
                                  hidden_dim=hidden_dim)
            action = jnp.clip(action, -1.0, 1.0)
            action = jnp.expand_dims(action, 0)
            next_state = env_step_fn(state, action)
            return (next_state, key), next_state.reward[0]

        (_, _), rewards = jax.lax.scan(step_fn, (env_state, rng), None, length=episode_length)
        return jnp.sum(rewards)

    return _single_rollout


def _make_single_rollout_hierarchical(config: EnvConfig):
    """Create single rollout function for hierarchical strategy with given config."""
    episode_length = config.episode_length
    obs_dim = config.obs_dim
    num_blocks = config.num_blocks
    block_dim = config.block_dim

    def _single_rollout(env_step_fn, env_state, rng, blocks, projection):
        def step_fn(carry, _):
            state, key = carry
            obs = state.obs[0]
            action = hierarchical_forward(obs, blocks, projection,
                                         num_blocks=num_blocks,
                                         block_dim=block_dim,
                                         obs_dim=obs_dim)
            action = jnp.clip(action, -1.0, 1.0)
            action = jnp.expand_dims(action, 0)
            next_state = env_step_fn(state, action)
            return (next_state, key), next_state.reward[0]

        (_, _), rewards = jax.lax.scan(step_fn, (env_state, rng), None, length=episode_length)
        return jnp.sum(rewards)

    return _single_rollout


def _make_single_rollout_topological(config: EnvConfig):
    """Create single rollout function for topological strategy with given config."""
    episode_length = config.episode_length
    obs_dim = config.obs_dim
    graph_nodes = config.graph_nodes

    def _single_rollout(env_step_fn, env_state, rng, adjacency, projection):
        def step_fn(carry, _):
            state, key = carry
            obs = state.obs[0]
            action = topological_forward(obs, adjacency, projection,
                                        obs_dim=obs_dim,
                                        graph_nodes=graph_nodes)
            action = jnp.clip(action, -1.0, 1.0)
            action = jnp.expand_dims(action, 0)
            next_state = env_step_fn(state, action)
            return (next_state, key), next_state.reward[0]

        (_, _), rewards = jax.lax.scan(step_fn, (env_state, rng), None, length=episode_length)
        return jnp.sum(rewards)

    return _single_rollout


# =============================================================================
# Batched evaluation functions
# =============================================================================

def _make_batched_eval_flat(env, env_step_fn, config: EnvConfig):
    """Create batched evaluation function for flat strategy."""
    single_rollout = _make_single_rollout_flat(config)
    num_rollouts = config.num_rollouts

    def single_eval(rng, weights):
        """Evaluate single individual with multiple rollouts."""
        rngs = jax.random.split(rng, num_rollouts * 2)

        total_reward = 0.0
        for i in range(num_rollouts):
            state = env.reset(jax.random.split(rngs[i * 2], 1))
            r = single_rollout(env_step_fn, state, rngs[i * 2 + 1], weights)
            total_reward += r

        return total_reward / num_rollouts

    return jax.jit(jax.vmap(single_eval, in_axes=(0, 0)))


def _make_batched_eval_hierarchical(env, env_step_fn, config: EnvConfig):
    """Create batched evaluation function for hierarchical strategy."""
    single_rollout = _make_single_rollout_hierarchical(config)
    num_rollouts = config.num_rollouts

    def single_eval(rng, blocks, projection):
        rngs = jax.random.split(rng, num_rollouts * 2)

        total_reward = 0.0
        for i in range(num_rollouts):
            state = env.reset(jax.random.split(rngs[i * 2], 1))
            r = single_rollout(env_step_fn, state, rngs[i * 2 + 1], blocks, projection)
            total_reward += r

        return total_reward / num_rollouts

    return jax.jit(jax.vmap(single_eval, in_axes=(0, 0, 0)))


def _make_batched_eval_topological(env, env_step_fn, config: EnvConfig):
    """Create batched evaluation function for topological strategy."""
    single_rollout = _make_single_rollout_topological(config)
    num_rollouts = config.num_rollouts

    def single_eval(rng, adjacency, projection):
        rngs = jax.random.split(rng, num_rollouts * 2)

        total_reward = 0.0
        for i in range(num_rollouts):
            state = env.reset(jax.random.split(rngs[i * 2], 1))
            r = single_rollout(env_step_fn, state, rngs[i * 2 + 1], adjacency, projection)
            total_reward += r

        return total_reward / num_rollouts

    return jax.jit(jax.vmap(single_eval, in_axes=(0, 0, 0)))


def get_batched_eval_fn(strategy: str, env_name: str = 'ant'):
    """Get cached batched evaluation function for strategy and environment."""
    cache_key = (strategy, env_name)

    if cache_key not in _BATCHED_EVAL_CACHE:
        env, config = _get_env(env_name)
        env_step_fn = _get_env_step(env_name)

        if strategy == 'flat' or strategy == 'cppn':
            # CPPN outputs weights in same format as flat
            _BATCHED_EVAL_CACHE[cache_key] = _make_batched_eval_flat(env, env_step_fn, config)
        elif strategy == 'hierarchical':
            _BATCHED_EVAL_CACHE[cache_key] = _make_batched_eval_hierarchical(env, env_step_fn, config)
        elif strategy == 'topological':
            _BATCHED_EVAL_CACHE[cache_key] = _make_batched_eval_topological(env, env_step_fn, config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return _BATCHED_EVAL_CACHE[cache_key]


# =============================================================================
# Public API
# =============================================================================

def evaluate_population_batched(phenotype_data_list: list, strategy: str,
                                env_name: str = 'ant') -> jnp.ndarray:
    """
    Evaluate entire population in parallel.

    Args:
        phenotype_data_list: List of phenotype data dicts from genotype network
        strategy: Strategy name ('flat', 'hierarchical', 'topological', 'cppn')
        env_name: Environment name from env_configs ('ant', 'swimmer', etc.)

    Returns:
        Array of fitness values for each individual
    """
    pop_size = len(phenotype_data_list)
    batched_eval = get_batched_eval_fn(strategy, env_name)

    # Generate random keys for each individual
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, pop_size)

    if strategy == 'flat' or strategy == 'cppn':
        # Both flat and CPPN use the same weight format
        weights_batch = jnp.stack([
            jnp.asarray(p['weights']).squeeze() for p in phenotype_data_list
        ])
        fitnesses = batched_eval(rngs, weights_batch)

    elif strategy == 'hierarchical':
        blocks_batch = jnp.stack([
            jnp.asarray(p['blocks']).squeeze() for p in phenotype_data_list
        ])
        proj_batch = jnp.stack([
            jnp.asarray(p['projection']).squeeze() for p in phenotype_data_list
        ])
        fitnesses = batched_eval(rngs, blocks_batch, proj_batch)

    elif strategy == 'topological':
        adj_batch = jnp.stack([
            jnp.asarray(p['adjacency']).squeeze() for p in phenotype_data_list
        ])
        proj_batch = jnp.stack([
            jnp.asarray(p['projection']).squeeze() for p in phenotype_data_list
        ])
        fitnesses = batched_eval(rngs, adj_batch, proj_batch)

    return fitnesses


def brax_fitness_batched(phenotype_data: dict, env_name: str = 'ant') -> float:
    """Single individual evaluation."""
    fitnesses = evaluate_population_batched([phenotype_data], phenotype_data['strategy'], env_name)
    return float(fitnesses[0])


# Legacy API for backwards compatibility
def brax_ant_fitness_jit(phenotype_data: dict, num_rollouts: int = 2) -> float:
    """Single individual evaluation (legacy API)."""
    return brax_fitness_batched(phenotype_data, env_name='ant')
