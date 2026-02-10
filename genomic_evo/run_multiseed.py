#!/usr/bin/env python3
"""
Multi-seed experiment runner for Genomic Weight Thesis.

Runs multiple seeds for a given strategy and saves structured results.
Supports multiple environments and multi-task experiments.

Usage:
    python3 run_multiseed.py <strategy> [options]

Examples:
    python3 run_multiseed.py flat --seeds 5
    python3 run_multiseed.py hierarchical --env swimmer --seeds 5
    python3 run_multiseed.py flat --multitask --seeds 5
"""
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Import the batched evolution function
from test_single_node_batched import single_node_evolution_batched


def run_multiseed_experiment(strategy: str, num_seeds: int = 5, start_seed: int = None,
                             env_name: str = 'ant', num_generations: int = 100,
                             multitask: bool = False):
    """Run multiple seeds for a strategy and collect results.

    Args:
        strategy: Encoding strategy ('flat', 'hierarchical', 'topological')
        num_seeds: Number of random seeds to run
        start_seed: Starting seed value (default: random based on time)
        env_name: Environment name from env_configs
        num_generations: Generations per run
        multitask: If True, use multi-task evaluation

    Returns:
        Dictionary with experiment results and summary statistics
    """
    if start_seed is None:
        start_seed = int(time.time()) % 100000

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_suffix = "_multitask" if multitask else ""
    experiment_id = f"{strategy}_{env_name}{task_suffix}_{timestamp}"

    print(f"\n{'='*70}")
    print(f"MULTI-SEED EXPERIMENT: {strategy.upper()}")
    print(f"Environment: {env_name}")
    print(f"Multi-task: {multitask}")
    print(f"Seeds: {num_seeds} (starting from {start_seed})")
    print(f"Generations: {num_generations}")
    print(f"Experiment ID: {experiment_id}")
    print(f"{'='*70}\n")

    all_results = {
        'experiment_id': experiment_id,
        'strategy': strategy,
        'env_name': env_name,
        'multitask': multitask,
        'num_seeds': num_seeds,
        'num_generations': num_generations,
        'start_seed': start_seed,
        'timestamp': timestamp,
        'runs': []
    }

    for i in range(num_seeds):
        seed = start_seed + i * 1000  # Space seeds apart
        print(f"\n{'='*70}")
        print(f"RUN {i+1}/{num_seeds} - Seed: {seed}")
        print(f"{'='*70}")

        try:
            if multitask:
                # Multi-task evolution
                result = run_multitask_evolution(
                    strategy=strategy,
                    num_generations=num_generations,
                    seed=seed
                )
            else:
                # Standard single-task evolution
                result = single_node_evolution_batched(
                    strategy=strategy,
                    num_generations=num_generations,
                    seed=seed,
                    env_name=env_name
                )

            # Extract key metrics
            run_summary = {
                'seed': seed,
                'run_index': i,
                'final_fitness': result['final_fitness'],
                'total_time': result['total_time'],
                'avg_steps_per_sec': result['avg_steps_per_sec'],
                'fitness_curve': [h['max_fitness'] for h in result['fitness_history']],
                'avg_fitness_curve': [h['avg_fitness'] for h in result['fitness_history']],
                'success': True
            }

            all_results['runs'].append(run_summary)

            print(f"\nRun {i+1} complete: Final fitness = {result['final_fitness']:.2f}")

        except Exception as e:
            print(f"\nRun {i+1} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results['runs'].append({
                'seed': seed,
                'run_index': i,
                'success': False,
                'error': str(e)
            })

    # Calculate summary statistics
    successful_runs = [r for r in all_results['runs'] if r.get('success', False)]

    if successful_runs:
        fitnesses = [r['final_fitness'] for r in successful_runs]
        all_results['summary'] = {
            'num_successful': len(successful_runs),
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'std_fitness': (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses)) ** 0.5,
            'mean_time': sum(r['total_time'] for r in successful_runs) / len(successful_runs),
            'mean_steps_per_sec': sum(r['avg_steps_per_sec'] for r in successful_runs) / len(successful_runs)
        }

    # Save results
    results_file = results_dir / f"{experiment_id}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {strategy.upper()}")
    print(f"{'='*70}")
    print(f"Successful runs: {len(successful_runs)}/{num_seeds}")
    if successful_runs:
        print(f"Mean fitness: {all_results['summary']['mean_fitness']:.2f} +/- {all_results['summary']['std_fitness']:.2f}")
        print(f"Max fitness: {all_results['summary']['max_fitness']:.2f}")
        print(f"Min fitness: {all_results['summary']['min_fitness']:.2f}")
    print(f"Results saved to: {results_file}")

    return all_results


def run_multitask_evolution(strategy: str, num_generations: int = 100, seed: int = None):
    """Run multi-task evolution experiment.

    Uses the multi-task fitness evaluator where a single genotype must
    perform well across all 4 directional tasks.
    """
    import gc
    import time as time_module
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    from evotorch import Problem
    from evotorch.algorithms import SNES
    from jax.flatten_util import ravel_pytree

    from genotype_nets import create_compressor
    from env_configs import get_config
    from fitness_env_multitask import evaluate_population_multitask

    if seed is None:
        seed = int(time_module.time() * 1000) % 2**31

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use Ant config for multi-task (all tasks use Ant body)
    config = get_config('ant')

    z = jnp.zeros((1, 128))
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
    sol_len = len(flat_params)

    print(f"\n{'='*60}")
    print(f"Starting MULTI-TASK Evolution: {strategy.upper()}")
    print(f"Tasks: forward, backward, left, right")
    print(f"SEED: {seed}")
    print(f"{'='*60}\n")

    class MultitaskProblem(Problem):
        def __init__(self, **kwargs):
            super().__init__(objective_sense="max", solution_length=sol_len, **kwargs)

        def _evaluate_batch(self, solutions):
            pop_values = solutions.values.cpu().numpy()
            pop_size = len(pop_values)

            z = jnp.zeros((1, 128))
            phenotype_data_list = []

            for i in range(pop_size):
                variables = unflatten_fn(jnp.asarray(pop_values[i]))
                p_data = net.apply(variables, z)
                phenotype_data_list.append(p_data)

            fitnesses = evaluate_population_multitask(phenotype_data_list, strategy)
            solutions.set_evals(torch.tensor(np.array(fitnesses), device=solutions.device))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    problem = MultitaskProblem(
        initial_bounds=(-0.1, 0.1),
        device=device,
        dtype=torch.float32
    )

    popsize = 50
    searcher = SNES(problem, stdev_init=0.01, popsize=popsize)

    fitness_history = []
    start_time = time_module.time()

    for generation in range(num_generations):
        gen_start = time_module.time()
        searcher.step()
        gen_time = time_module.time() - gen_start

        avg_fit = float(torch.mean(searcher.population.evals))
        max_fit = float(torch.max(searcher.population.evals))
        min_fit = float(torch.min(searcher.population.evals))

        # Multi-task: 4 tasks × 2 rollouts × 200 steps
        steps_this_gen = popsize * 4 * config.num_rollouts * config.episode_length
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

    if len(fitness_history) > 1:
        avg_steps_per_sec = sum(h['steps_per_sec'] for h in fitness_history[1:]) / (len(fitness_history) - 1)
    else:
        avg_steps_per_sec = fitness_history[0]['steps_per_sec']

    print(f"\n{'='*60}")
    print(f"Multi-task evolution complete!")
    print(f"Final fitness (avg across 4 tasks): {final_fitness:.2f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*60}")

    return {
        'final_fitness': final_fitness,
        'fitness_history': fitness_history,
        'seed': seed,
        'strategy': strategy,
        'env_name': 'ant_multitask',
        'num_generations': num_generations,
        'total_time': total_time,
        'avg_steps_per_sec': avg_steps_per_sec
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run multi-seed evolution experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 run_multiseed.py flat --seeds 5
    python3 run_multiseed.py hierarchical --env swimmer --seeds 5
    python3 run_multiseed.py flat --multitask --seeds 5
        """
    )
    parser.add_argument('strategy',
                       choices=['flat', 'hierarchical', 'topological', 'cppn'],
                       help='Encoding strategy')
    parser.add_argument('--seeds', '-n', type=int, default=5,
                       help='Number of seeds to run (default: 5)')
    parser.add_argument('--start-seed', type=int, default=None,
                       help='Starting seed value (default: random)')
    parser.add_argument('--env', default='ant',
                       help='Environment name (default: ant)')
    parser.add_argument('--generations', '-g', type=int, default=100,
                       help='Generations per run (default: 100)')
    parser.add_argument('--multitask', action='store_true',
                       help='Use multi-task evaluation (Ant 4-directional)')

    args = parser.parse_args()

    results = run_multiseed_experiment(
        strategy=args.strategy,
        num_seeds=args.seeds,
        start_seed=args.start_seed,
        env_name=args.env,
        num_generations=args.generations,
        multitask=args.multitask
    )
