"""
Multi-task fitness evaluation for compression experiments.

Tests the hypothesis that structured encodings provide better compression
when a single genotype must control multiple tasks. This creates the
strongest compression pressure.

Multi-task Ant: Single controller must work for forward/backward/left/right movement.
"""
import os
# Force CPU before JAX import
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import jit, vmap
from brax import envs
from brax.envs.wrappers.training import wrap

from phenotype_forward import flat_forward, hierarchical_forward, topological_forward
from env_configs import get_config, EnvConfig


# Multi-task configurations
# All use Ant body but with different reward directions
MULTITASK_CONFIGS = {
    'ant_forward': {'direction': jnp.array([1.0, 0.0])},   # +x velocity
    'ant_backward': {'direction': jnp.array([-1.0, 0.0])}, # -x velocity
    'ant_left': {'direction': jnp.array([0.0, 1.0])},      # +y velocity
    'ant_right': {'direction': jnp.array([0.0, -1.0])},    # -y velocity
}

# Cache
_MULTITASK_ENV_CACHE = None
_MULTITASK_STEP_FN = None


def _get_multitask_env():
    """Get cached Ant environment for multi-task evaluation."""
    global _MULTITASK_ENV_CACHE
    if _MULTITASK_ENV_CACHE is None:
        config = get_config('ant')
        env = envs.get_environment('ant', backend=config.backend)
        _MULTITASK_ENV_CACHE = (wrap(env, episode_length=config.episode_length), config)
    return _MULTITASK_ENV_CACHE


def _get_multitask_step():
    """Get JIT-compiled environment step function."""
    global _MULTITASK_STEP_FN
    if _MULTITASK_STEP_FN is None:
        env, _ = _get_multitask_env()
        _MULTITASK_STEP_FN = jax.jit(env.step)
    return _MULTITASK_STEP_FN


def _compute_directional_reward(state, direction: jnp.ndarray) -> float:
    """Compute reward based on velocity in specified direction.

    Args:
        state: Brax environment state
        direction: 2D unit vector for desired movement direction

    Returns:
        Reward = dot(velocity, direction) + survival bonus
    """
    # Brax Ant qvel[0:2] is x,y velocity
    # state.qvel is (batch, qvel_dim), we take [0] for single env
    velocity_xy = state.pipeline_state.qd[0, :2]  # x, y velocity
    directional_reward = jnp.dot(velocity_xy, direction)
    survival_bonus = 1.0
    return directional_reward + survival_bonus


# =============================================================================
# Multi-task rollout functions
# =============================================================================

def _make_multitask_rollout_flat(config: EnvConfig):
    """Create multi-task rollout function for flat strategy."""
    episode_length = config.episode_length
    obs_dim = config.obs_dim
    action_dim = config.action_dim
    hidden_dim = config.hidden_dim

    def _single_rollout(env_step_fn, env_state, rng, weights, direction):
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

            # Custom directional reward instead of default forward reward
            velocity_xy = next_state.pipeline_state.qd[0, :2]
            reward = jnp.dot(velocity_xy, direction) + 1.0

            return (next_state, key), reward

        (_, _), rewards = jax.lax.scan(step_fn, (env_state, rng), None, length=episode_length)
        return jnp.sum(rewards)

    return _single_rollout


def _make_multitask_rollout_hierarchical(config: EnvConfig):
    """Create multi-task rollout function for hierarchical strategy."""
    episode_length = config.episode_length
    obs_dim = config.obs_dim
    num_blocks = config.num_blocks
    block_dim = config.block_dim

    def _single_rollout(env_step_fn, env_state, rng, blocks, projection, direction):
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

            velocity_xy = next_state.pipeline_state.qd[0, :2]
            reward = jnp.dot(velocity_xy, direction) + 1.0

            return (next_state, key), reward

        (_, _), rewards = jax.lax.scan(step_fn, (env_state, rng), None, length=episode_length)
        return jnp.sum(rewards)

    return _single_rollout


def _make_multitask_rollout_topological(config: EnvConfig):
    """Create multi-task rollout function for topological strategy."""
    episode_length = config.episode_length
    obs_dim = config.obs_dim
    graph_nodes = config.graph_nodes

    def _single_rollout(env_step_fn, env_state, rng, adjacency, projection, direction):
        def step_fn(carry, _):
            state, key = carry
            obs = state.obs[0]
            action = topological_forward(obs, adjacency, projection,
                                        obs_dim=obs_dim,
                                        graph_nodes=graph_nodes)
            action = jnp.clip(action, -1.0, 1.0)
            action = jnp.expand_dims(action, 0)
            next_state = env_step_fn(state, action)

            velocity_xy = next_state.pipeline_state.qd[0, :2]
            reward = jnp.dot(velocity_xy, direction) + 1.0

            return (next_state, key), reward

        (_, _), rewards = jax.lax.scan(step_fn, (env_state, rng), None, length=episode_length)
        return jnp.sum(rewards)

    return _single_rollout


# =============================================================================
# Batched multi-task evaluation
# =============================================================================

_MULTITASK_EVAL_CACHE = {}


def _make_batched_multitask_eval_flat(env, env_step_fn, config: EnvConfig, directions: list):
    """Create batched evaluation for flat strategy across multiple directions."""
    single_rollout = _make_multitask_rollout_flat(config)
    num_rollouts = config.num_rollouts

    def single_eval(rng, weights):
        """Evaluate single individual across all task directions."""
        total_reward = 0.0
        rngs = jax.random.split(rng, len(directions) * num_rollouts * 2)
        rng_idx = 0

        for direction in directions:
            task_reward = 0.0
            for _ in range(num_rollouts):
                state = env.reset(jax.random.split(rngs[rng_idx], 1))
                r = single_rollout(env_step_fn, state, rngs[rng_idx + 1], weights, direction)
                task_reward += r
                rng_idx += 2
            total_reward += task_reward / num_rollouts

        # Average across tasks
        return total_reward / len(directions)

    return jax.jit(jax.vmap(single_eval, in_axes=(0, 0)))


def _make_batched_multitask_eval_hierarchical(env, env_step_fn, config: EnvConfig, directions: list):
    """Create batched evaluation for hierarchical strategy across multiple directions."""
    single_rollout = _make_multitask_rollout_hierarchical(config)
    num_rollouts = config.num_rollouts

    def single_eval(rng, blocks, projection):
        total_reward = 0.0
        rngs = jax.random.split(rng, len(directions) * num_rollouts * 2)
        rng_idx = 0

        for direction in directions:
            task_reward = 0.0
            for _ in range(num_rollouts):
                state = env.reset(jax.random.split(rngs[rng_idx], 1))
                r = single_rollout(env_step_fn, state, rngs[rng_idx + 1], blocks, projection, direction)
                task_reward += r
                rng_idx += 2
            total_reward += task_reward / num_rollouts

        return total_reward / len(directions)

    return jax.jit(jax.vmap(single_eval, in_axes=(0, 0, 0)))


def _make_batched_multitask_eval_topological(env, env_step_fn, config: EnvConfig, directions: list):
    """Create batched evaluation for topological strategy across multiple directions."""
    single_rollout = _make_multitask_rollout_topological(config)
    num_rollouts = config.num_rollouts

    def single_eval(rng, adjacency, projection):
        total_reward = 0.0
        rngs = jax.random.split(rng, len(directions) * num_rollouts * 2)
        rng_idx = 0

        for direction in directions:
            task_reward = 0.0
            for _ in range(num_rollouts):
                state = env.reset(jax.random.split(rngs[rng_idx], 1))
                r = single_rollout(env_step_fn, state, rngs[rng_idx + 1], adjacency, projection, direction)
                task_reward += r
                rng_idx += 2
            total_reward += task_reward / num_rollouts

        return total_reward / len(directions)

    return jax.jit(jax.vmap(single_eval, in_axes=(0, 0, 0)))


def get_multitask_eval_fn(strategy: str, tasks: list = None):
    """Get cached multi-task evaluation function.

    Args:
        strategy: 'flat', 'hierarchical', or 'topological'
        tasks: List of task names (default: all 4 directions)

    Returns:
        JIT-compiled batched evaluation function
    """
    if tasks is None:
        tasks = ['ant_forward', 'ant_backward', 'ant_left', 'ant_right']

    cache_key = (strategy, tuple(tasks))

    if cache_key not in _MULTITASK_EVAL_CACHE:
        env, config = _get_multitask_env()
        env_step_fn = _get_multitask_step()

        # Get direction vectors for selected tasks
        directions = [MULTITASK_CONFIGS[t]['direction'] for t in tasks]

        if strategy == 'flat':
            _MULTITASK_EVAL_CACHE[cache_key] = _make_batched_multitask_eval_flat(
                env, env_step_fn, config, directions)
        elif strategy == 'hierarchical':
            _MULTITASK_EVAL_CACHE[cache_key] = _make_batched_multitask_eval_hierarchical(
                env, env_step_fn, config, directions)
        elif strategy == 'topological':
            _MULTITASK_EVAL_CACHE[cache_key] = _make_batched_multitask_eval_topological(
                env, env_step_fn, config, directions)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return _MULTITASK_EVAL_CACHE[cache_key]


# =============================================================================
# Public API
# =============================================================================

def evaluate_population_multitask(phenotype_data_list: list, strategy: str,
                                  tasks: list = None) -> jnp.ndarray:
    """
    Evaluate population across multiple tasks (average fitness).

    This is the key test for compression: a single genotype must produce
    a phenotype that works well across multiple different objectives.

    Args:
        phenotype_data_list: List of phenotype data dicts from genotype network
        strategy: Strategy name ('flat', 'hierarchical', 'topological')
        tasks: List of task names (default: all 4 directions)

    Returns:
        Array of fitness values (averaged across tasks) for each individual
    """
    pop_size = len(phenotype_data_list)
    batched_eval = get_multitask_eval_fn(strategy, tasks)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, pop_size)

    if strategy == 'flat':
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
