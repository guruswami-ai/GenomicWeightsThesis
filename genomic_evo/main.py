import os
import sys
import time
import json
import socket
import subprocess
import jax
import jax.numpy as jnp
import torch
import mlx.core as mx
from distributed import distributed_evolution

PYTHON_BIN = "/Users/admin/miniforge3/envs/genomic_evo/bin"

def log_configuration(args, seed, output_dir="."):
    """Log all run configuration for reproducibility"""
    config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "hostname": socket.gethostname(),
        "seed": seed,
        "strategy": args.strategy,
        "generations": args.generations,
        "single_node": args.single_node,
        "python_version": sys.version,
        "jax_version": jax.__version__,
        "torch_version": torch.__version__,
        # Hyperparameters (hardcoded in test_single_node.py)
        "population_size": 50,
        "stdev_init": 0.01,
        "initial_bounds": [-0.1, 0.1],
        # Environment
        "environment": "brax_ant_v4",
        "episode_length": 200,
        "num_rollouts": 2,
        "action_dim": 8,
        "observation_dim": 27,
    }
    
    # Save to file
    config_filename = f"config_{args.strategy}_{seed}.json"
    config_path = os.path.join(output_dir, config_filename)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RUN CONFIGURATION (for reproducibility)")
    print("="*60)
    print(f"  Timestamp:    {config['timestamp']}")
    print(f"  Hostname:     {config['hostname']}")
    print(f"  SEED:         {seed}  ← CRITICAL FOR REPRODUCTION")
    print(f"  Strategy:     {config['strategy']}")
    print(f"  Generations:  {config['generations']}")
    print(f"  Population:   {config['population_size']}")
    print(f"  Config file:  {config_path}")
    print("="*60 + "\n")
    
    return config

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Genomic Compression Experiment")
    parser.add_argument('--prototype', action='store_true', help='Run 1-gen prototype')
    parser.add_argument('--suite', action='store_true', help='Run full 9-run suite')
    parser.add_argument('--single-node', action='store_true', help='Skip distributed init')
    parser.add_argument('--strategy', type=str, choices=['flat', 'hierarchical', 'topological'],
                       help='Run specific strategy only')
    parser.add_argument('--generations', type=int, default=1000,
                       help='Number of generations to run (default: 1000)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: auto-generate)')
    args = parser.parse_args()
    
    # Generate or use provided seed
    if args.seed is None:
        seed = int(time.time() * 1000) % 2**31  # Millisecond-based seed
    else:
        seed = args.seed
    
    # Set all random seeds for reproducibility
    torch.manual_seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    
    print("Starting Genomic Compression Experiment...")
    print(f"Random Seed: {seed}")
    
    if not args.single_node:
        group = mx.distributed.init()
        rank = group.rank()
        world_size = group.size()
    else:
        rank = 0
        world_size = 1
    
    if args.strategy:
        # Log configuration (rank 0 only)
        if rank == 0:
            config = log_configuration(args, seed)
        
        # Run single strategy
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Strategy: {args.strategy.upper()}")
            print(f"{'='*60}\n")
        
        if args.single_node:
            # Import single-node version (uses Brax generalized backend on CPU)
            from test_single_node import single_node_evolution
            result = single_node_evolution(args.strategy, num_generations=args.generations, seed=seed)
        else:
            # Distributed still uses legacy for now (JACCL fix needed)
            result = distributed_evolution(args.strategy, world_size=world_size)
        
        if rank == 0:
            print(f"\nResult: {result}")
            
            # Save results
            results_filename = f"results_{args.strategy}_{seed}.json"
            result['seed'] = seed
            result['strategy'] = args.strategy
            result['generations'] = args.generations
            with open(results_filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {results_filename}")
        
    elif args.prototype:
        # Quick 1-generation test
        if rank == 0:
            print(f"🚀 Running prototype (World Size: {world_size})...")
        result = distributed_evolution('flat', world_size=world_size)
        if rank == 0:
            print(f"Prototype result: {result}")
        
    elif args.suite:  # Full suite
        if rank == 0:
            print(f"🏗️ Launching Full Suite (World Size: {world_size})...")
        strategies = ['flat', 'hierarchical', 'topological']
        results = {}
        
        for strategy in strategies:
            for replicate in range(3):
                run_name = f"{strategy}_rep{replicate}"
                if rank == 0:
                    print(f"\n{'='*60}")
                    print(f"Run: {run_name}")
                    print(f"{'='*60}\n")
                
                result = distributed_evolution(strategy, world_size=world_size)
                if rank == 0:
                    results[run_name] = result
                    print(f"{run_name}: {result}")
        
        if rank == 0:
            print("\n✅ All 9 runs complete!")
            print("\n🎯 FINAL RANKING:")
            # Sort by final fitness
            sorted_res = sorted(results.items(), key=lambda x: x[1]['final_fitness'] if isinstance(x[1], dict) else 0.0, reverse=True)
            for i, (name, res) in enumerate(sorted_res):
                fit = res['final_fitness'] if isinstance(res, dict) else 0.0
                print(f"{i+1}. {name}: {fit:.3f}")
            
    else:
        # Usage instructions if no flags provided
        if rank == 0:
            print("Usage: python main.py [--prototype | --suite]")
    
    if rank == 0:
        print("✅ Experiment step complete.")

if __name__ == "__main__":
    main()
