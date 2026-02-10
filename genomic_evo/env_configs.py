"""Environment configuration registry for genomic compression experiments.

Centralizes all environment-specific parameters to enable easy switching
between benchmarks while maintaining consistent architecture.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvConfig:
    """Configuration for a Brax environment."""
    env_name: str           # Brax environment name
    obs_dim: int            # Observation space dimension
    action_dim: int         # Action space dimension
    num_blocks: int         # Hierarchical encoding blocks
    block_dim: int          # Dimension per block
    graph_nodes: int        # Topological encoding nodes
    hidden_dim: int         # Flat encoding hidden dimension
    episode_length: int     # Steps per episode
    num_rollouts: int       # Rollouts per fitness evaluation
    backend: str = "generalized"  # Brax backend


# Environment configurations
ENV_CONFIGS = {
    # Original benchmark - baseline where flat encoding wins
    'ant': EnvConfig(
        env_name='ant',
        obs_dim=27,
        action_dim=8,
        num_blocks=8,
        block_dim=128,
        graph_nodes=64,
        hidden_dim=64,
        episode_length=200,
        num_rollouts=2,
    ),

    # Swimmer with 6 segments - strong repeated structure
    # Each segment is identical -> blocks should map 1:1 to segments
    # Wave locomotion requires coordinated phase between adjacent segments
    'swimmer': EnvConfig(
        env_name='swimmer',
        obs_dim=8,           # Standard 3-segment swimmer: 2 angles + 6 velocities
        action_dim=2,        # 2 joints
        num_blocks=3,        # Match segment count
        block_dim=64,        # Smaller blocks for simpler task
        graph_nodes=32,
        hidden_dim=32,
        episode_length=500,  # Longer episodes for swimming
        num_rollouts=2,
    ),

    # Humanoid - natural hierarchical body structure
    # torso -> limbs -> joints creates tree hierarchy
    # Bilateral symmetry should favor weight sharing
    'humanoid': EnvConfig(
        env_name='humanoid',
        obs_dim=376,         # Large observation space
        action_dim=17,       # Many actuated joints
        num_blocks=8,        # Body-part level: torso + 4 limbs + 3 sub-parts
        block_dim=256,       # Larger blocks for complex task
        graph_nodes=128,     # More nodes for complex body
        hidden_dim=128,
        episode_length=1000, # Longer for complex locomotion
        num_rollouts=2,
    ),

    # Walker2D - simpler hierarchical structure
    # Left/right leg symmetry
    'walker2d': EnvConfig(
        env_name='walker2d',
        obs_dim=17,
        action_dim=6,
        num_blocks=4,        # torso + 2 legs (can decompose further)
        block_dim=64,
        graph_nodes=32,
        hidden_dim=64,
        episode_length=500,
        num_rollouts=2,
    ),

    # HalfCheetah - sequential body structure
    'halfcheetah': EnvConfig(
        env_name='halfcheetah',
        obs_dim=17,
        action_dim=6,
        num_blocks=4,
        block_dim=64,
        graph_nodes=32,
        hidden_dim=64,
        episode_length=500,
        num_rollouts=2,
    ),
}


def get_config(env_name: str) -> EnvConfig:
    """Get environment configuration by name.

    Args:
        env_name: Name of environment (ant, swimmer, humanoid, etc.)

    Returns:
        EnvConfig with all parameters for the environment

    Raises:
        ValueError: If environment name is not recognized
    """
    if env_name not in ENV_CONFIGS:
        available = ', '.join(sorted(ENV_CONFIGS.keys()))
        raise ValueError(f"Unknown environment: {env_name}. Available: {available}")
    return ENV_CONFIGS[env_name]


def list_environments() -> list:
    """Return list of available environment names."""
    return sorted(ENV_CONFIGS.keys())
