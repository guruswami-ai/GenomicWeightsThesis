#!/bin/bash
# Single-Node Production Run (while debugging distributed)
# This validates the algorithm and gets results NOW

set -e

cd /opt/mlx-distributed
source .venv/bin/activate
cd ~/genomic_evo

# Performance tuning
export MLX_METAL_FAST_SYNCH=1
export MLX_BFS_MAX_WIDTH=64
export MLX_MAX_OPS_PER_BUFFER=100

echo "🚀 Launching Single-Node Genomic Evolution (All 3 Strategies)"
echo "  Node: muladhara (M3 Ultra)"
echo "  Population: 200 per strategy"
echo "  Generations: 1000"
echo ""

# Run all 3 strategies sequentially on single node
for strategy in flat hierarchical topological; do
    echo "Starting $strategy strategy..."
    python3 -u main.py --strategy $strategy --single-node 2>&1 | tee -a results_${strategy}.log
    echo "✅ $strategy complete"
    echo ""
done

echo "✅ All strategies complete! Results in ~/genomic_evo/results_*.log"
