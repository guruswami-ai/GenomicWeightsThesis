#!/usr/bin/env python3
"""
Batched CPU evolution for Genomic Weight Thesis experiment.

Uses jax.vmap to evaluate entire populations in parallel on CPU.
On M3 Ultra with unified memory, leverages all 24 cores efficiently.
Expected performance: 10,000-50,000 steps/sec on M3 Ultra.

Supports multiple environments via --env flag.
"""
# Import batched fitness env FIRST to set JAX_PLATFORMS=cpu
from fitness_env_batched import evaluate_population_batched

import jax
import jax.numpy as jnp
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import SNES
from genotype_nets import create_compressor
from env_configs import get_config
from jax.flatten_util import ravel_pytree


def get_net_and_params(strategy: str, env_name: str = 'ant'):
    """Get genotype network and parameter utilities for a strategy and environment."""
    config = get_config(env_name)

    z = jnp.zeros((1, 128))

    # Create compressor with environment-specific dimensions
    net = create_compressor(
        strategy=strategy,
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        block_dim=config.block_dim,
        graph_nodes=config.graph_nodes
    )

    variables = net.init(jax.random.PRNGKey(0), z)
    flat_params, unflatten_fn = ravel_pytree(variables)
    return net, unflatten_fn, len(flat_params), config


class BatchedBraxProblem(Problem):
    """EvoTorch Problem with batched CPU evaluation."""

    def __init__(self, strategy, env_name, net, unflatten_fn, solution_length, config, **kwargs):
        super().__init__(objective_sense="max", solution_length=solution_length, **kwargs)
        self.strategy = strategy
        self.env_name = env_name
        self.net = net
        self.unflatten_fn = unflatten_fn
        self.config = config

    def _evaluate_batch(self, solutions):
        """Evaluate entire batch in parallel."""
        pop_values = solutions.values.cpu().numpy()
        pop_size = len(pop_values)

        # Convert all genotypes to phenotypes
        z = jnp.zeros((1, 128))
        phenotype_data_list = []

        for i in range(pop_size):
            variables = self.unflatten_fn(jnp.asarray(pop_values[i]))
            p_data = self.net.apply(variables, z)
            phenotype_data_list.append(p_data)

        # Evaluate entire population in parallel
        fitnesses = evaluate_population_batched(phenotype_data_list, self.strategy, self.env_name)

        # Convert back to torch
        solutions.set_evals(torch.tensor(np.array(fitnesses), device=solutions.device))


def single_node_evolution_batched(strategy: str, num_generations: int = 100,
                                  seed: int = None, env_name: str = 'ant'):
    """Run batched CPU evolution."""
    import gc
    import time as time_module

    if seed is None:
        seed = int(time_module.time() * 1000) % 2**31

    torch.manual_seed(seed)
    np.random.seed(seed)

    net, unflatten_fn, sol_len, config = get_net_and_params(strategy, env_name)

    print(f"\n{'='*60}")
    print(f"Starting BATCHED CPU Evolution: {strategy.upper()}")
    print(f"Environment: {env_name} (obs={config.obs_dim}, act={config.action_dim})")
    print(f"SEED: {seed}")
    print(f"JAX devices: {jax.devices()}")
    print(f"{'='*60}\n")

    print(f"Solution length: {sol_len} parameters")
    print(f"Config: blocks={config.num_blocks}, block_dim={config.block_dim}, "
          f"graph_nodes={config.graph_nodes}")

    # Use MPS if available (Apple Silicon), otherwise CPU
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"EvoTorch device: {device}")

    problem = BatchedBraxProblem(
        strategy=strategy,
        env_name=env_name,
        net=net,
        unflatten_fn=unflatten_fn,
        solution_length=sol_len,
        config=config,
        initial_bounds=(-0.1, 0.1),
        device=device,
        dtype=torch.float32
    )

    popsize = 50
    searcher = SNES(problem, stdev_init=0.01, popsize=popsize)
    print(f"Population size: {popsize}")

    fitness_history = []
    start_time = time_module.time()

    print(f"\nStarting evolution for {num_generations} generations...")
    print("(First generation includes JIT compilation - will be slower)\n")

    for generation in range(num_generations):
        gen_start = time_module.time()
        searcher.step()
        gen_time = time_module.time() - gen_start

        avg_fit = float(torch.mean(searcher.population.evals))
        max_fit = float(torch.max(searcher.population.evals))
        min_fit = float(torch.min(searcher.population.evals))

        if torch.any(torch.isnan(searcher.population.evals)):
            print(f"FATAL: NaN detected in generation {generation}")
            break

        # Calculate steps/sec for this generation
        steps_this_gen = popsize * config.num_rollouts * config.episode_length
        steps_per_sec = steps_this_gen / gen_time if gen_time > 0 else 0

        if generation % 10 == 0 or generation < 5:
            print(f"Gen {generation:3d}: Avg={avg_fit:7.2f}, Max={max_fit:7.2f} | {gen_time:.2f}s | {steps_per_sec:,.0f} steps/sec")

            if generation > 0:
                gc.collect()

        fitness_history.append({
            'generation': generation,
            'avg_fitness': avg_fit,
            'max_fitness': max_fit,
            'min_fitness': min_fit,
            'time': gen_time,
            'steps_per_sec': steps_per_sec
        })

    total_time = time_module.time() - start_time
    final_fitness = float(torch.max(searcher.population.evals))

    # Calculate overall stats (excluding first gen due to JIT)
    if len(fitness_history) > 1:
        avg_time_per_gen = sum(h['time'] for h in fitness_history[1:]) / (len(fitness_history) - 1)
        avg_steps_per_sec = sum(h['steps_per_sec'] for h in fitness_history[1:]) / (len(fitness_history) - 1)
    else:
        avg_time_per_gen = fitness_history[0]['time']
        avg_steps_per_sec = fitness_history[0]['steps_per_sec']

    print(f"\n{'='*60}")
    print(f"Evolution complete!")
    print(f"{'='*60}")
    print(f"   Final fitness: {final_fitness:.2f}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Avg time/gen (excl. JIT): {avg_time_per_gen:.2f}s")
    print(f"   Avg steps/sec (excl. JIT): {avg_steps_per_sec:,.0f}")
    print(f"   Total steps: {num_generations * popsize * config.num_rollouts * config.episode_length:,}")

    return {
        'final_fitness': final_fitness,
        'fitness_history': fitness_history,
        'seed': seed,
        'strategy': strategy,
        'env_name': env_name,
        'num_generations': num_generations,
        'total_time': total_time,
        'avg_steps_per_sec': avg_steps_per_sec
    }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Run batched evolution experiment')
    parser.add_argument('strategy', nargs='?', default='flat',
                       choices=['flat', 'hierarchical', 'topological', 'cppn'],
                       help='Encoding strategy (default: flat)')
    parser.add_argument('--env', default='ant',
                       help='Environment name (default: ant)')
    parser.add_argument('--generations', '-g', type=int, default=100,
                       help='Number of generations (default: 100)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed (default: random)')

    args = parser.parse_args()

    print("=== System Info ===")
    print(f"JAX devices: {jax.devices()}")
    print(f"PyTorch MPS available: {torch.backends.mps.is_available()}")
    print("")

    result = single_node_evolution_batched(
        strategy=args.strategy,
        num_generations=args.generations,
        seed=args.seed,
        env_name=args.env
    )
    print(f"\nResult: {result}")
