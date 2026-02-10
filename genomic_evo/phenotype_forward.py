"""
Pure functional phenotype network implementations for JIT compilation.

These functions take weights as explicit JAX array arguments, allowing
the JIT to compile once and reuse across all individuals in the population.

All dimension parameters are now configurable to support multiple environments.
"""
import jax
import jax.numpy as jnp


# Default architecture constants (Ant environment - for backwards compatibility)
DEFAULT_OBS_DIM = 27
DEFAULT_ACTION_DIM = 8
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_BLOCKS = 8
DEFAULT_BLOCK_DIM = 128
DEFAULT_GRAPH_NODES = 64


def flat_forward(obs: jnp.ndarray,
                 weights: jnp.ndarray,
                 obs_dim: int = DEFAULT_OBS_DIM,
                 action_dim: int = DEFAULT_ACTION_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM) -> jnp.ndarray:
    """
    Flat strategy: 2-layer MLP using genotype weights.

    Args:
        obs: Observation array (obs_dim,)
        weights: Flat weight vector from FlatCompressor
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension

    Returns:
        Action array (action_dim,)
    """
    # Layer sizes
    w1_size = obs_dim * hidden_dim
    w2_size = hidden_dim * action_dim

    # Extract and reshape weights
    w1 = weights[:w1_size].reshape(obs_dim, hidden_dim)
    w2 = weights[w1_size:w1_size + w2_size].reshape(hidden_dim, action_dim)

    # Forward pass
    h = jnp.matmul(obs, w1)
    h = jax.nn.relu(h)
    action = jnp.matmul(h, w2)

    return jnp.tanh(action)


def hierarchical_forward(obs: jnp.ndarray,
                         blocks: jnp.ndarray,
                         projection: jnp.ndarray,
                         num_blocks: int = DEFAULT_NUM_BLOCKS,
                         block_dim: int = DEFAULT_BLOCK_DIM,
                         obs_dim: int = DEFAULT_OBS_DIM) -> jnp.ndarray:
    """
    Hierarchical strategy: TAD-like modular blocks.

    Each block processes the output of the previous block (or input obs).
    All block outputs are concatenated and projected to action space.

    Args:
        obs: Observation array (obs_dim,)
        blocks: Stacked block weights (num_blocks, block_dim)
        projection: Final projection weights (total_hidden, action_dim)
        num_blocks: Number of processing blocks
        block_dim: Dimension of each block
        obs_dim: Observation dimension

    Returns:
        Action array (action_dim,)
    """
    x = obs
    outputs = []

    # Process each block
    for i in range(num_blocks):
        block = blocks[i]
        in_dim = x.shape[-1]
        block_out_dim = block.shape[-1] // in_dim if in_dim > 0 else block_dim // obs_dim

        # Reshape block into kernel
        kernel = block[:in_dim * block_out_dim].reshape(in_dim, block_out_dim)
        h = jnp.matmul(x, kernel)
        h = jax.nn.relu(h)
        outputs.append(h)
        x = h

    # Concatenate all block outputs
    h = jnp.concatenate(outputs, axis=-1)

    # Project to action space
    h_dim = h.shape[-1]
    proj_h = projection.shape[0]

    if h_dim == proj_h:
        action = jnp.matmul(h, projection)
    elif h_dim > proj_h:
        action = jnp.matmul(h[:proj_h], projection)
    else:
        action = jnp.matmul(h, projection[:h_dim, :])

    return jnp.tanh(action)


def topological_forward(obs: jnp.ndarray,
                        adjacency: jnp.ndarray,
                        projection: jnp.ndarray,
                        obs_dim: int = DEFAULT_OBS_DIM,
                        graph_nodes: int = DEFAULT_GRAPH_NODES) -> jnp.ndarray:
    """
    Topological strategy: Graph message passing with chromatin-like structure.

    The distance penalty in the genotype network biases toward local
    connections, mimicking chromatin physics where long-range loops are costly.

    Args:
        obs: Observation array (obs_dim,)
        adjacency: Graph adjacency matrix (graph_nodes, graph_nodes)
        projection: Output projection weights (graph_nodes, action_dim)
        obs_dim: Observation dimension
        graph_nodes: Number of graph nodes

    Returns:
        Action array (action_dim,)
    """
    n_nodes = adjacency.shape[-1]

    # Pad or truncate obs to match graph nodes
    if obs_dim < n_nodes:
        x = jnp.pad(obs, (0, n_nodes - obs_dim))
    else:
        x = obs[:n_nodes]

    # Two rounds of message passing (graph diffusion)
    x = jnp.matmul(x, adjacency)
    x = jax.nn.relu(x)
    x = jnp.matmul(x, adjacency)
    x = jax.nn.relu(x)

    # Project to action space
    action = jnp.matmul(x, projection)

    return jnp.tanh(action)


def phenotype_forward(obs: jnp.ndarray,
                      strategy: str,
                      obs_dim: int = DEFAULT_OBS_DIM,
                      action_dim: int = DEFAULT_ACTION_DIM,
                      hidden_dim: int = DEFAULT_HIDDEN_DIM,
                      num_blocks: int = DEFAULT_NUM_BLOCKS,
                      block_dim: int = DEFAULT_BLOCK_DIM,
                      graph_nodes: int = DEFAULT_GRAPH_NODES,
                      **weight_arrays) -> jnp.ndarray:
    """
    Dispatch to the appropriate forward function based on strategy.

    Args:
        obs: Observation array (obs_dim,)
        strategy: One of 'flat', 'hierarchical', 'topological'
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension (flat strategy)
        num_blocks: Number of blocks (hierarchical strategy)
        block_dim: Block dimension (hierarchical strategy)
        graph_nodes: Number of graph nodes (topological strategy)
        **weight_arrays: Strategy-specific weight arrays

    Returns:
        Action array (action_dim,)
    """
    if strategy == 'flat':
        return flat_forward(obs, weight_arrays['weights'],
                           obs_dim=obs_dim, action_dim=action_dim,
                           hidden_dim=hidden_dim)
    elif strategy == 'hierarchical':
        return hierarchical_forward(obs, weight_arrays['blocks'],
                                   weight_arrays['projection'],
                                   num_blocks=num_blocks, block_dim=block_dim,
                                   obs_dim=obs_dim)
    elif strategy == 'topological':
        return topological_forward(obs, weight_arrays['adjacency'],
                                  weight_arrays['projection'],
                                  obs_dim=obs_dim, graph_nodes=graph_nodes)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
