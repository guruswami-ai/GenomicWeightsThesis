import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict

class PhenotypeNet(nn.Module):
    """
    Phenotype network that uses ONLY genotype-emitted weights.
    No nn.Dense layers that require Flax initialization.
    This is critical for the thesis: genotype encodes ALL phenotype structure.
    """
    phenotype_data: Dict
    action_dim: int = 8  # Brax Ant action space
    
    @nn.compact
    def __call__(self, x):
        strategy = self.phenotype_data['strategy']
        
        # Ensure x has consistent shape
        original_ndim = x.ndim
        if x.ndim == 1:
            x = x[None, :]  # Add batch dim
        
        if strategy == 'flat':
            # Use g-net emitted weights as kernels
            weights = jnp.squeeze(self.phenotype_data['weights'])
            # Simple 2-layer MLP: obs (27) -> hidden -> action (8)
            in_dim = x.shape[-1]
            hidden_dim = min(64, weights.size // (in_dim + self.action_dim))
            
            # First layer: obs -> hidden
            w1_size = in_dim * hidden_dim
            if w1_size <= weights.size:
                w1 = weights[:w1_size].reshape(in_dim, hidden_dim)
                h = jnp.matmul(x, w1)
                h = jax.nn.relu(h)
                
                # Second layer: hidden -> action
                w2_size = hidden_dim * self.action_dim
                if w1_size + w2_size <= weights.size:
                    w2 = weights[w1_size:w1_size + w2_size].reshape(hidden_dim, self.action_dim)
                    action = jnp.matmul(h, w2)
                    result = jnp.tanh(action)
                    return result.squeeze(0) if original_ndim == 1 else result
            
            # Fallback: simple projection
            result = jnp.tanh(x[..., :self.action_dim] * 0.1)
            return result.squeeze(0) if original_ndim == 1 else result
            
        elif strategy == 'hierarchical':
            # Each block acts as a modular layer (TAD-like structure)
            outputs = []
            for block in self.phenotype_data['blocks']:
                block = jnp.squeeze(block)
                in_dim = x.shape[-1]
                block_dim = block.size // in_dim if in_dim > 0 else 0
                if block_dim > 0:
                    kernel = block[:in_dim * block_dim].reshape(in_dim, block_dim)
                    h = jnp.matmul(x, kernel)
                    h = jax.nn.relu(h)
                    outputs.append(h)
                    x = h  # Feed-forward through blocks
            
            if outputs:
                h = jnp.concatenate(outputs, axis=-1)
                
                # Use genotype-emitted projection weights (NO nn.Dense!)
                proj = jnp.squeeze(self.phenotype_data['projection'])
                if proj.ndim == 3:
                    proj = proj[0]  # Remove batch dim from projection
                
                # Project: (batch, total_hidden) @ (total_hidden, action_dim)
                h_dim = h.shape[-1]
                proj_h = proj.shape[0] if proj.ndim == 2 else proj.size // self.action_dim
                
                if h_dim == proj_h:
                    action = jnp.matmul(h, proj)
                else:
                    # Truncate or pad to match
                    if h_dim > proj_h:
                        action = jnp.matmul(h[..., :proj_h], proj)
                    else:
                        action = jnp.matmul(h, proj[:h_dim, :])
                
                result = jnp.tanh(action)
                return result.squeeze(0) if original_ndim == 1 else result
            else:
                result = jnp.tanh(x[..., :self.action_dim] * 0.1)
                return result.squeeze(0) if original_ndim == 1 else result
            
        elif strategy == 'topological':
            adj = jnp.squeeze(self.phenotype_data['adjacency'])
            if adj.ndim == 3:
                adj = adj[0]  # Remove batch dim
            n_nodes = adj.shape[-1]
            
            # Project obs to node features
            feat_dim = x.shape[-1]
            if feat_dim < n_nodes:
                pad_width = [(0, 0)] * (x.ndim - 1) + [(0, n_nodes - feat_dim)]
                x = jnp.pad(x, tuple(pad_width))
            elif feat_dim > n_nodes:
                x = x[..., :n_nodes]
            
            # Graph message passing: x @ adj (simple diffusion on graph)
            # x: (batch, n_nodes), adj: (n_nodes, n_nodes)
            x = jnp.matmul(x, adj)
            x = jax.nn.relu(x)
            
            # Second pass of message passing for richer features
            x = jnp.matmul(x, adj)
            x = jax.nn.relu(x)
            
            # Use genotype-emitted projection weights (NO nn.Dense!)
            proj = jnp.squeeze(self.phenotype_data['projection'])
            if proj.ndim == 3:
                proj = proj[0]  # Remove batch dim
            
            # proj: (n_nodes, action_dim)
            action = jnp.matmul(x, proj)
            result = jnp.tanh(action)
            
            return result.squeeze(0) if original_ndim == 1 else result
            
        # Fallback
        result = jnp.zeros((*x.shape[:-1], self.action_dim))
        return result.squeeze(0) if original_ndim == 1 else result
