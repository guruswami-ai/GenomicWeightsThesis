import jax
import jax.numpy as jnp

def spectral_modularity(adj_matrix: jnp.ndarray) -> float:
    """Simplified modularity score using eigenvalues"""
    # This is a proxy for how 'block-diagonal' the matrix is
    eigvals = jnp.linalg.eigvalsh(adj_matrix)
    return jnp.std(eigvals)

def chromatin_loss(adj_matrix: jnp.ndarray) -> float:
    """Penalize long-range connections like chromatin loop costs"""
    n = adj_matrix.shape[-1]
    distances = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n))
    
    # P(s) ~ s^-1 decay (biological signature)
    long_range_penalty = jnp.mean(adj_matrix * (distances > 32))
    modularity = spectral_modularity(adj_matrix)
    
    return long_range_penalty - 0.5 * modularity
