#!/bin/bash
# Genomic Weight Thesis - Distributed Evolution
# Mirrors /opt/mlx-distributed/launch_inference.sh pattern

set -e

# Activate MLX distributed environment
cd /opt/mlx-distributed
source .venv/bin/activate

# Navigate to experiment directory
cd ~/genomic_evo

# Performance tuning
export MLX_METAL_FAST_SYNCH=1
export MLX_BFS_MAX_WIDTH=64
export MLX_MAX_OPS_PER_BUFFER=100

HOSTFILE=/opt/chakra/inference/hostfiles/chakra-tp5.json

echo "🚀 Launching Distributed Genomic Evolution Experiment"
echo "  Hostfile: $HOSTFILE"
echo "  Backend: JACCL (TB5 RDMA)"
echo ""

# Launch exactly like LLM inference
exec mlx.launch \
    --hostfile "$HOSTFILE" \
    --backend jaccl \
    python3 main.py --suite

