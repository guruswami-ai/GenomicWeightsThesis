#!/usr/bin/env python3
import subprocess
import time
import sys
import random

# Seed Categories from Hypothesis
SEEDS = {
    'fibonacci': [5, 8, 13, 21, 55, 89, 144, 233],
    'golden': [1618, 16180],
    'euler': [2718, 27182],
    'pi': [314, 3141, 31415],
    'primes': [2, 3, 5, 7, 11, 13, 17, 19, 23],
    'arbitrary': [123, 7890, 4242, 9999]
}

STRATEGIES = ['flat', 'hierarchical', 'topological']
NODES = ['muladhara', 'anahata', 'vishuddha', 'kathaka'] # Free nodes

def run_remote(node, cmd):
    ssh_cmd = f"ssh {node} 'source ~/.zshrc; {cmd}'"
    print(f"[{node}] Launching: {cmd}")
    # Run in background on remote
    proc = subprocess.Popen(f"ssh {node} \"nohup {cmd} > ~/genomic_evo/multiseed.log 2>&1 &\"", shell=True)
    return proc

def generate_commands():
    commands = []
    for strategy in STRATEGIES:
        for category, seeds in SEEDS.items():
            for seed in seeds:
                # Short 20-gen runs for survey
                cmd = f"cd ~/genomic_evo && /opt/mlx-distributed/.venv/bin/python main.py --strategy {strategy} --seed {seed} --generations 20 --single-node"
                log_file = f"{strategy}_{category}_{seed}.log"
                full_cmd = f"{cmd} > {log_file} 2>&1"
                commands.append((strategy, category, seed, full_cmd))
    return commands

def main():
    cmds = generate_commands()
    random.shuffle(cmds) # Shuffle to distribute load
    
    print(f"Generated {len(cmds)} tasks")
    
    # Simple dispatcher (manual for now, just printing for verification)
    # In reality, we'd need a queue system.
    # For now, let's just pick 3 distinct experiments to launch on the 3 free nodes as a test
    
    available_nodes = NODES
    for i, node in enumerate(available_nodes):
        if i >= len(cmds): break
        strat, cat, seed, cmd = cmds[i]
        print(f"Assigning {strat} (seed {seed}, {cat}) to {node}")
        # run_remote(node, cmd) 

if __name__ == "__main__":
    main()
