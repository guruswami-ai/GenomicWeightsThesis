"""
Genotype network architectures for genomic compression experiments.

These networks map a latent code z to phenotype parameters through different
compression strategies inspired by genomic organization.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict


class FlatCompressor(nn.Module):
    """Zador baseline: direct mapping with no structural constraints.

    Maps latent code to flat weight vector for a 2-layer MLP phenotype.
    """
    hidden_dim: int = 512      # Genotype network hidden dimension
    output_dim: int = 10000    # Phenotype weight count

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Dict:
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.relu(x)
        weights = nn.Dense(self.output_dim)(x)
        return {'strategy': 'flat', 'weights': weights}


class HierarchicalCompressor(nn.Module):
    """TAD-inspired encoding with modular block structure.

    Emits num_blocks separate weight blocks, mimicking how TADs create
    functionally-related gene clusters. Each block can specialize for
    different aspects of the control task.
    """
    num_blocks: int = 8
    block_size: int = 128
    action_dim: int = 8

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Dict:
        # Generate block structure (TADs) - modular weight blocks
        block_logits = nn.Dense(self.num_blocks * self.block_size)(z)
        blocks = jnp.split(block_logits, self.num_blocks, axis=-1)

        # Emit projection weights: (total_hidden) -> action_dim
        total_hidden = self.num_blocks * self.block_size
        projection_weights = nn.Dense(total_hidden * self.action_dim)(z)

        return {
            'strategy': 'hierarchical',
            'blocks': blocks,
            'projection': projection_weights.reshape(-1, total_hidden, self.action_dim)
        }


class TopologicalCompressor(nn.Module):
    """Chromatin-inspired encoding with learned graph structure.

    Emits a graph adjacency matrix with distance penalty mimicking the
    physical cost of long-range chromatin loops. This biases toward
    locally-coherent representations while allowing learned long-range
    connections when beneficial.
    """
    n_nodes: int = 64
    action_dim: int = 8
    distance_penalty: float = 0.1

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Dict:
        # Generate adjacency matrix with distance bias
        adj_logits = nn.Dense(self.n_nodes * self.n_nodes)(z)
        adj = jax.nn.softmax(adj_logits.reshape(-1, self.n_nodes, self.n_nodes), axis=-1)

        # Distance penalty (chromatin loop cost)
        distances = jnp.abs(jnp.arange(self.n_nodes)[:, None] - jnp.arange(self.n_nodes))
        adj *= jnp.exp(-self.distance_penalty * distances)
        adj = adj / (adj.sum(axis=-1, keepdims=True) + 1e-8)

        # Emit projection weights: n_nodes -> action_dim
        projection_weights = nn.Dense(self.n_nodes * self.action_dim)(z)

        return {
            'strategy': 'topological',
            'adjacency': adj,
            'projection': projection_weights.reshape(-1, self.n_nodes, self.action_dim)
        }


# Factory functions for environment-specific compressors

def create_flat_compressor(obs_dim: int, action_dim: int, hidden_dim: int = 64,
                           genotype_hidden: int = 512) -> FlatCompressor:
    """Create FlatCompressor sized for a specific environment.

    The output dimension is computed to provide weights for a 2-layer MLP:
    w1: obs_dim -> hidden_dim
    w2: hidden_dim -> action_dim

    Args:
        obs_dim: Environment observation dimension
        action_dim: Environment action dimension
        hidden_dim: Phenotype network hidden dimension
        genotype_hidden: Genotype network hidden dimension

    Returns:
        Configured FlatCompressor instance
    """
    output_dim = obs_dim * hidden_dim + hidden_dim * action_dim
    return FlatCompressor(hidden_dim=genotype_hidden, output_dim=output_dim)


def create_hierarchical_compressor(action_dim: int, num_blocks: int = 8,
                                   block_size: int = 128) -> HierarchicalCompressor:
    """Create HierarchicalCompressor with specified block structure.

    For tasks with natural modularity:
    - num_blocks should match the number of functional units (e.g., segments, limbs)
    - block_size controls the capacity of each module

    Args:
        action_dim: Environment action dimension
        num_blocks: Number of TAD-like blocks (match to task structure)
        block_size: Dimension of each block

    Returns:
        Configured HierarchicalCompressor instance
    """
    return HierarchicalCompressor(
        num_blocks=num_blocks,
        block_size=block_size,
        action_dim=action_dim
    )


def create_topological_compressor(action_dim: int, n_nodes: int = 64,
                                  distance_penalty: float = 0.1) -> TopologicalCompressor:
    """Create TopologicalCompressor with specified graph size.

    The n_nodes parameter should be >= obs_dim for the environment.
    Larger values allow more complex graph structures but increase
    computation.

    Args:
        action_dim: Environment action dimension
        n_nodes: Number of graph nodes
        distance_penalty: Strength of local connection bias (0.1 is default)

    Returns:
        Configured TopologicalCompressor instance
    """
    return TopologicalCompressor(
        n_nodes=n_nodes,
        action_dim=action_dim,
        distance_penalty=distance_penalty
    )


class CPPNCompressor(nn.Module):
    """CPPN (Compositional Pattern-Producing Network) indirect encoding.

    Classic indirect encoding from neuroevolution literature (Stanley 2007).
    Uses a small neural network to generate weights based on coordinate
    positions, creating smooth, pattern-based weight matrices.

    The CPPN takes (x_in, x_out, distance, bias) as input and outputs
    the weight value for that position. This creates natural patterns
    like symmetry, gradients, and repetition.
    """
    cppn_hidden: int = 32      # CPPN internal hidden dimension
    obs_dim: int = 27          # Target phenotype observation dim
    action_dim: int = 8        # Target phenotype action dim
    hidden_dim: int = 64       # Target phenotype hidden dim

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Dict:
        batch_size = z.shape[0] if len(z.shape) > 1 else 1

        # The latent z modulates the CPPN weights via hypernetwork approach
        # Generate CPPN parameters from z
        cppn_w1 = nn.Dense(4 * self.cppn_hidden)(z)  # 4 inputs -> cppn_hidden
        cppn_w1 = cppn_w1.reshape(batch_size, 4, self.cppn_hidden)

        cppn_w2 = nn.Dense(self.cppn_hidden * self.cppn_hidden)(z)
        cppn_w2 = cppn_w2.reshape(batch_size, self.cppn_hidden, self.cppn_hidden)

        cppn_w3 = nn.Dense(self.cppn_hidden)(z)  # cppn_hidden -> 1
        cppn_w3 = cppn_w3.reshape(batch_size, self.cppn_hidden, 1)

        # Generate coordinate grids for weight matrices
        # W1: obs_dim -> hidden_dim
        w1_rows = self.obs_dim
        w1_cols = self.hidden_dim

        # Create normalized coordinates for W1
        x_in_1 = jnp.linspace(-1, 1, w1_rows)
        x_out_1 = jnp.linspace(-1, 1, w1_cols)
        grid_in_1, grid_out_1 = jnp.meshgrid(x_in_1, x_out_1, indexing='ij')
        dist_1 = jnp.abs(grid_in_1 - grid_out_1)
        bias_1 = jnp.ones_like(grid_in_1)

        # Stack into coordinate inputs: (obs_dim, hidden_dim, 4)
        coords_1 = jnp.stack([grid_in_1, grid_out_1, dist_1, bias_1], axis=-1)
        coords_1_flat = coords_1.reshape(-1, 4)  # (obs_dim * hidden_dim, 4)

        # W2: hidden_dim -> action_dim
        w2_rows = self.hidden_dim
        w2_cols = self.action_dim

        x_in_2 = jnp.linspace(-1, 1, w2_rows)
        x_out_2 = jnp.linspace(-1, 1, w2_cols)
        grid_in_2, grid_out_2 = jnp.meshgrid(x_in_2, x_out_2, indexing='ij')
        dist_2 = jnp.abs(grid_in_2 - grid_out_2)
        bias_2 = jnp.ones_like(grid_in_2)

        coords_2 = jnp.stack([grid_in_2, grid_out_2, dist_2, bias_2], axis=-1)
        coords_2_flat = coords_2.reshape(-1, 4)  # (hidden_dim * action_dim, 4)

        # Query CPPN for each coordinate (batched across individuals)
        def query_cppn(coords, w1, w2, w3):
            """Query CPPN at given coordinates."""
            h = jnp.matmul(coords, w1)  # (n_coords, cppn_hidden)
            h = jnp.sin(h)  # Sinusoidal activation (classic CPPN)
            h = jnp.matmul(h, w2)
            h = jnp.tanh(h)
            out = jnp.matmul(h, w3)  # (n_coords, 1)
            return out.squeeze(-1)

        # Generate weights for each individual in batch
        w1_weights = jax.vmap(query_cppn, in_axes=(None, 0, 0, 0))(
            coords_1_flat, cppn_w1, cppn_w2, cppn_w3
        )  # (batch, obs_dim * hidden_dim)

        w2_weights = jax.vmap(query_cppn, in_axes=(None, 0, 0, 0))(
            coords_2_flat, cppn_w1, cppn_w2, cppn_w3
        )  # (batch, hidden_dim * action_dim)

        # Concatenate into flat weight vector (same format as FlatCompressor)
        weights = jnp.concatenate([w1_weights, w2_weights], axis=-1)

        return {'strategy': 'cppn', 'weights': weights}


def create_cppn_compressor(obs_dim: int, action_dim: int, hidden_dim: int = 64,
                           cppn_hidden: int = 32) -> CPPNCompressor:
    """Create CPPNCompressor for a specific environment.

    CPPN is a classic indirect encoding that generates weights through
    a pattern-producing neural network. This provides a fundamentally
    different compression approach than TAD/chromatin-inspired methods.

    Args:
        obs_dim: Environment observation dimension
        action_dim: Environment action dimension
        hidden_dim: Phenotype network hidden dimension
        cppn_hidden: CPPN internal hidden dimension

    Returns:
        Configured CPPNCompressor instance
    """
    return CPPNCompressor(
        cppn_hidden=cppn_hidden,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )


def create_compressor(strategy: str, obs_dim: int, action_dim: int,
                      hidden_dim: int = 64, num_blocks: int = 8,
                      block_dim: int = 128, graph_nodes: int = 64):
    """Create a compressor for the given strategy and environment dimensions.

    This is the main factory function for creating compressors based on
    environment configuration.

    Args:
        strategy: One of 'flat', 'hierarchical', 'topological', 'cppn'
        obs_dim: Environment observation dimension
        action_dim: Environment action dimension
        hidden_dim: Phenotype hidden dimension (flat/cppn strategy)
        num_blocks: Number of blocks (hierarchical strategy)
        block_dim: Block dimension (hierarchical strategy)
        graph_nodes: Number of graph nodes (topological strategy)

    Returns:
        Configured compressor instance

    Raises:
        ValueError: If strategy is not recognized
    """
    if strategy == 'flat':
        return create_flat_compressor(obs_dim, action_dim, hidden_dim)
    elif strategy == 'hierarchical':
        return create_hierarchical_compressor(action_dim, num_blocks, block_dim)
    elif strategy == 'topological':
        return create_topological_compressor(action_dim, graph_nodes)
    elif strategy == 'cppn':
        return create_cppn_compressor(obs_dim, action_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Available: flat, hierarchical, topological, cppn")
