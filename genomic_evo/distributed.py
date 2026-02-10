import jax
import jax.numpy as jnp
import mlx.core as mx
from evotorch import Problem
from evotorch.algorithms import SNES
from genotype_nets import FlatCompressor, HierarchicalCompressor, TopologicalCompressor
from phenotype_net import PhenotypeNet
from fitness_env import predator_avoidance_fitness

from jax.flatten_util import ravel_pytree

def get_net_and_params(strategy):
    z = jnp.zeros((1, 128))
    if strategy == 'flat':
        net = FlatCompressor()
    elif strategy == 'hierarchical':
        net = HierarchicalCompressor()
    else:
        net = TopologicalCompressor()
    
    variables = net.init(jax.random.PRNGKey(0), z)
    flat_params, unflatten_fn = ravel_pytree(variables)
    return net, unflatten_fn, len(flat_params)

def evaluate_compression(params_flat, strategy, net, unflatten_fn):
    """Initialize p-net from g-net params and evaluate fitness"""
    # Unflatten evolved params back into Flax structure
    variables = unflatten_fn(params_flat)
    z = jnp.zeros((1, 128)) # Idealized experience vector
    
    p_data = net.apply(variables, z)
    p_net = PhenotypeNet(phenotype_data=p_data)
    
    # Evaluate fitness in Brax Ant environment
    # Pass phenotype_data to enable chromatin loss for topological strategies
    fitness = predator_avoidance_fitness(
        lambda x: p_net.apply({}, x),
        phenotype_data=p_data
    )
    return fitness

import torch
from evotorch import Problem
from evotorch.core import SolutionBatch

class ShardedProblem(Problem):
    def __init__(self, strategy, group, net, unflatten_fn, solution_length, **kwargs):
        super().__init__(objective_sense="max", solution_length=solution_length, **kwargs)
        self.strategy = strategy
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        self.net = net
        self.unflatten_fn = unflatten_fn

    def _evaluate_batch(self, solutions: SolutionBatch):
        pop_total = len(solutions)
        pop_per_node = pop_total // self.world_size
        
        start_idx = self.rank * pop_per_node
        end_idx = (self.rank + 1) * pop_per_node
        
        # Local evaluation of shard
        my_shard_values = solutions.values[start_idx:end_idx].cpu().numpy()
        
        # Pass net and unflatten_fn to evaluate_compression
        eval_fn = lambda s: evaluate_compression(s, self.strategy, self.net, self.unflatten_fn)
        local_fitnesses = jax.vmap(eval_fn)(jnp.array(my_shard_values))
        
        # Sync via JACCL/MLX
        local_fit_mlx = mx.array(local_fitnesses.tolist())
        all_fitnesses_gathered = mx.distributed.all_gather(local_fit_mlx, group=self.group)
        
        # Flatten and broadcast global fitnesses
        global_fitnesses = torch.tensor(all_fitnesses_gathered.tolist(), device=solutions.device).flatten()
        solutions.set_evals(global_fitnesses)

def distributed_evolution(strategy: str, world_size: int, resume_from=None):
    """Shard population across cluster nodes"""
    import gc
    
    group = mx.distributed.init()
    rank = group.rank()
    
    # Get strategy-specific network metadata
    net, unflatten_fn, sol_len = get_net_and_params(strategy)
    
    problem = ShardedProblem(
        strategy=strategy,
        group=group,
        net=net,
        unflatten_fn=unflatten_fn,
        solution_length=sol_len,
        initial_bounds=(-0.1, 0.1),
        device="mps"
    )
    problem._vectorized = True
    
    searcher = SNES(problem, stdev_init=0.01)
    
    # Resume from checkpoint if provided
    start_gen = 0
    if resume_from and rank == 0:
        checkpoint = torch.load(resume_from)
        start_gen = checkpoint['generation'] + 1
        print(f"Resuming from generation {start_gen}")
    
    for generation in range(start_gen, 5000):
        searcher.step()
        
        if rank == 0 and generation % 10 == 0:
            avg_fit = torch.mean(searcher.population.evals)
            print(f"[{strategy}] Gen {generation}: Avg Fitness {float(avg_fit):.4f}")
            
            # Memory management - force garbage collection every 10 gens
            gc.collect()
        
        # Checkpoint every 100 generations (rank 0 only)
        if rank == 0 and generation % 100 == 0 and generation > 0:
            checkpoint_path = f"checkpoint_{strategy}_g{generation}.pt"
            torch.save({
                'generation': generation,
                'strategy': strategy,
                'population': searcher.population.access_values(),
                'fitnesses': searcher.population.access_evals()
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoint to free memory
            if generation > 100:
                old_checkpoint = f"checkpoint_{strategy}_g{generation-100}.pt"
                try:
                    import os
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                except Exception:
                    pass
            
    return {'final_fitness': float(torch.max(searcher.population.evals))}
