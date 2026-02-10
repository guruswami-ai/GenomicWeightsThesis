#!/usr/bin/env python3
"""
Single-node evolution for Genomic Weight Thesis experiment.

Uses JIT-compiled fitness evaluation for performance (~4000 steps/sec).
"""
# Import fitness_env FIRST to set JAX_PLATFORMS=cpu before other JAX imports
from fitness_env import brax_ant_fitness_jit

import jax
import jax.numpy as jnp
import torch
from evotorch import Problem
from evotorch.algorithms import SNES
from genotype_nets import FlatCompressor, HierarchicalCompressor, TopologicalCompressor
from jax.flatten_util import ravel_pytree


def get_net_and_params(strategy):
    """Get genotype network and parameter utilities for a strategy."""
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


def evaluate_individual(params_flat, strategy, net, unflatten_fn):
    """
    Evaluate a single individual using JIT-compiled fitness.

    Args:
        params_flat: Flattened genotype parameters (numpy array)
        strategy: Strategy name ('flat', 'hierarchical', 'topological')
        net: Genotype network (Flax module)
        unflatten_fn: Function to unflatten params to pytree

    Returns:
        Fitness value (float)
    """
    # Reconstruct genotype network variables
    variables = unflatten_fn(jnp.asarray(params_flat))
    z = jnp.zeros((1, 128))

    # Generate phenotype data from genotype
    p_data = net.apply(variables, z)

    # Evaluate using JIT-compiled fitness (fast!)
    fitness = brax_ant_fitness_jit(p_data, num_rollouts=2)
    return fitness


class BraxProblem(Problem):
    """EvoTorch Problem wrapper for Brax fitness evaluation."""

    def __init__(self, strategy, net, unflatten_fn, solution_length, **kwargs):
        super().__init__(objective_sense="max", solution_length=solution_length, **kwargs)
        self.strategy = strategy
        self.net = net
        self.unflatten_fn = unflatten_fn

    def _evaluate_batch(self, solutions):
        """Evaluate batch of solutions sequentially (JIT caching makes this fast)."""
        pop_values = solutions.values.cpu().numpy()

        # Evaluate each individual - JIT compiles once, then reuses
        fitnesses = []
        for i in range(len(pop_values)):
            fitness = evaluate_individual(
                pop_values[i], self.strategy, self.net, self.unflatten_fn
            )
            fitnesses.append(fitness)

        # Set fitness values
        solutions.set_evals(torch.tensor(fitnesses, device=solutions.device))

def single_node_evolution(strategy: str, num_generations: int = 100, seed: int = None):
    """Run evolution on single node for testing"""
    import gc
    import time as time_module
    
    # Use provided seed or generate one
    if seed is None:
        seed = int(time_module.time() * 1000) % 2**31
    
    # Set all random seeds for reproducibility
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    # Note: JAX key is set at geno net init
    
    print(f"\n{'='*60}")
    print(f"Starting Single-Node Evolution: {strategy.upper()}")
    print(f"SEED: {seed}")
    print(f"{'='*60}\n")
    
    # Get strategy-specific network metadata
    net, unflatten_fn, sol_len = get_net_and_params(strategy)
    print(f"Solution length: {sol_len} parameters")
    
    # Initialize validator for this run
    from validation import ExperimentValidator, quick_sanity_check
    validator = ExperimentValidator(strategy, seed)
    
    problem = BraxProblem(
        strategy=strategy,
        net=net,
        unflatten_fn=unflatten_fn,
        solution_length=sol_len,
        initial_bounds=(-0.1, 0.1),
        device="mps"
    )
    problem._vectorized = True
    
    searcher = SNES(problem, stdev_init=0.01, popsize=50)
    
    # Track fitness history for analysis
    fitness_history = []
    max_fitness_history = []  # For validation
    
    print(f"Starting evolution for {num_generations} generations...")
    for generation in range(num_generations):
        searcher.step()
        
        avg_fit = float(torch.mean(searcher.population.evals))
        max_fit = float(torch.max(searcher.population.evals))
        min_fit = float(torch.min(searcher.population.evals))
        
        # Validate fitness values
        validator.validate_fitness(avg_fit, generation, max_fitness_history)
        max_fitness_history.append(max_fit)
        
        # Check for NaN in population
        if torch.any(torch.isnan(searcher.population.evals)):
            validator.log_issue("ERROR", f"Gen {generation}: NaN in population fitness")
            print(f"❌ FATAL: NaN detected in generation {generation}, stopping early")
            break
            
        if generation % 10 == 0:
            print(f"Gen {generation:3d}: Avg={avg_fit:7.2f}, Max={max_fit:7.2f}, Min={min_fit:7.2f}")
            
            # Periodic deep validation on best individual
            if generation > 0:
                best_idx = torch.argmax(searcher.population.evals)
                best_params = searcher.population.values[best_idx].cpu().numpy()
                
                # Validate genotype
                result = validator.validate_genotype_weights(best_params, generation)
                if isinstance(result, tuple):
                    valid, stats = result
                    if generation % 50 == 0:
                        print(f"     Weight stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            
            # Memory management
            gc.collect()
        
        fitness_history.append({
            'generation': generation,
            'avg_fitness': avg_fit,
            'max_fitness': max_fit,
            'min_fitness': min_fit
        })
    
    final_fitness = float(torch.max(searcher.population.evals))
    
    # Final validation: check best individual's phenotype output
    best_idx = torch.argmax(searcher.population.evals)
    best_params = searcher.population.values[best_idx].cpu().numpy()
    variables = unflatten_fn(jnp.array(best_params))
    z = jnp.zeros((1, 128))
    p_data = net.apply(variables, z)

    # Validate phenotype data
    validator.validate_phenotype_data(p_data, num_generations)

    # Test phenotype network output using pure forward functions
    from phenotype_forward import flat_forward, hierarchical_forward, topological_forward
    test_obs = jnp.ones(27) * 0.5

    if strategy == 'flat':
        weights = jnp.asarray(p_data['weights']).squeeze()
        action = flat_forward(test_obs, weights)
    elif strategy == 'hierarchical':
        blocks = jnp.asarray(p_data['blocks'])
        projection = jnp.asarray(p_data['projection']).squeeze()
        action = hierarchical_forward(test_obs, blocks, projection)
    else:  # topological
        adjacency = jnp.asarray(p_data['adjacency']).squeeze()
        projection = jnp.asarray(p_data['projection']).squeeze()
        action = topological_forward(test_obs, adjacency, projection)

    validator.validate_phenotype_output(action, num_generations)
    
    # Save validation report
    validation_summary = validator.get_summary()
    validator.save_report(f"validation_{strategy}_{seed}.json")
    
    print(f"\n✅ Evolution complete! Final fitness: {final_fitness:.2f}")
    print(f"   Validation: {validation_summary['status']} ({validation_summary['checks_passed']} checks passed)")
    
    return {
        'final_fitness': final_fitness,
        'fitness_history': fitness_history,
        'seed': seed,
        'strategy': strategy,
        'num_generations': num_generations,
        'validation_status': validation_summary['status'],
        'validation_errors': validation_summary['errors'],
        'validation_warnings': len(validation_summary['warnings_list'])
    }

if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else 'flat'
    num_gens = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    result = single_node_evolution(strategy, num_gens)
    print(f"\nResult: {result}")
