# Hardware Utilization Analysis

## Optimizing for Reproducibility on Alternative Platforms

---

# Hardware & Network Reference

## 1. The "Chakra" Cluster Specification

The experiment runs on a specialized 7-node Apple Silicon cluster designed for high-throughput evolutionary computation.

### Compute Nodes (The "Body" - M3 Ultra)

These 5 nodes form the primary compute mesh for the Genomic Weight experiment.

| Node Name | Chakra | Role | Hardware Spec |
|-----------|--------|------|---------------|
| **muladhara** | Root | Compute | **M3 Ultra**, 512GB RAM, 76-core GPU, 32-core CPU |
| **svadhisthana**| Sacral | Compute | **M3 Ultra**, 512GB RAM, 76-core GPU, 32-core CPU |
| **manipura** | Solar | Compute | **M3 Ultra**, 512GB RAM, 76-core GPU, 32-core CPU |
| **anahata** | Heart | Compute | **M3 Ultra**, 512GB RAM, 76-core GPU, 32-core CPU |
| **vishuddha** | Throat | Compute | **M3 Ultra**, 512GB RAM, 76-core GPU, 32-core CPU |

- **Total VRAM**: 2.5 TB (Unified)
- **Total Compute**: 380 GPU Cores
- **Interconnect**: Thunderbolt 5 Mesh (20Gbps+ RDMA-capable)

### Management Nodes (The "Mind" - M4 Pro)

These nodes handle orchestration, monitoring (MQTT), and control. They do NOT run the core evolutionary simulation to ensure valid benchmarking.

| Node Name | Chakra | Role | Hardware Spec |
|-----------|--------|------|---------------|
| **ajna** | Third Eye | Manager | **M4 Pro Mac Mini**, 64GB RAM, 20-core GPU |
| **sahasrara** | Crown | Manager | **M4 Pro Mac Mini**, 64GB RAM, 20-core GPU |

### Network Topology

1. **Thunderbolt Mesh (Low Latency)**: The 5 compute nodes are connected via a daisy-chained Thunderbolt bridge (IP range `192.168.123.x`). This is enabling future distributed training research.
2. **Ethernet/WiFi (Control Plane)**: Standard 1GbE/WiFi for SSH and internet access (IP range `10.61.106.x`).

---

## 2. Benchmark Reference for Reproducibility

### Baseline Hardware (This Experiment)

- **Device**: Apple Mac Studio (M3 Ultra)
- **RAM**: 512 GB Unified (Note: Experiment uses <30GB)
- **OS**: macOS Sonoma 14.x
- **Framework**: JAX (Metal Backend)

### Minimum Consumer Requirements

Research can be reproduced on consumer hardware. We aim to validate this in Phase 7.

- **Recommended**: NVIDIA RTX 3090 / 4090 (24GB VRAM)
- **Minimum**: NVIDIA RTX 3060 / 4060 (12GB VRAM)
- **Apple Alternative**: M2/M3 Max MacBook Pro (32GB+ RAM)

### Current Utilization Profile (M3 Ultra)

### Current Utilization (During Experiment)

```
Memory: ~25 GB used / 512 GB available (5% utilization)
CPU: 150-200% (using 1.5-2 cores of 32)
GPU: Fully utilized (via Metal/MPS)
```

**Key Finding**: This workload is **GPU-bound**, not memory or CPU bound. The 512GB RAM is vastly underutilized.

---

## 2. Workload Characteristics

### What the Experiment Does

```
Per Generation (50 individuals):
├── For each individual:
│   ├── Forward pass: Genotype → Phenotype (GPU)
│   ├── Brax simulation: 200 timesteps × 2 rollouts (GPU)
│   └── Fitness aggregation
├── Selection + Mutation (CPU)
└── Repeat
```

### Bottleneck Analysis

| Component | Hardware | Utilization | Bottleneck? |
|-----------|----------|-------------|-------------|
| Genotype forward pass | GPU | High | **YES** |
| Phenotype inference | GPU | High | **YES** |
| Brax physics sim | GPU | High | **YES** |
| SNES optimization | CPU | Low | No |
| Memory (weights) | RAM | Very Low | No |
| Network (distributed) | N/A | N/A | N/A (single-node) |

**Conclusion**: This is a pure **GPU compute** workload. More GPU cores = faster evolution.

---

## 3. Alternative Hardware Recommendations

### Option A: M4 Pro Mac Mini Cluster (Recommended Budget Option)

| Spec | M4 Pro Mac Mini | vs M3 Ultra |
|------|-----------------|-------------|
| GPU Cores | 20 | 26% of M3 Ultra |
| Memory | 64 GB | Sufficient |
| Memory BW | 273 GB/s | 34% of M3 Ultra |
| Price | ~$2,000 | ~10% of Mac Studio |
| Cluster of 4 | 80 GPU cores | Similar to 1 M3 Ultra |

**Estimated Performance**:

- Single M4 Pro: ~25% speed of M3 Ultra
- 4× M4 Pro cluster: Comparable to single M3 Ultra
- **Cost**: ~$8,000 vs ~$7,000 (single Mac Studio)

**Verdict**: Cost-effective for reproduction. Slower but viable.

### Option B: NVIDIA RTX 4090

| Spec | RTX 4090 | vs M3 Ultra |
|------|----------|-------------|
| CUDA Cores | 16,384 | Different architecture |
| Memory | 24 GB VRAM | Sufficient (model < 10GB) |
| FP32 TFLOPs | 82.6 | Higher raw compute |
| Price | ~$2,000 | Excellent value |

**Requires**:

- Porting JAX Metal backend → JAX CUDA
- Should be straightforward (JAX supports both)

**Estimated Performance**:

- Likely **faster** than M3 Ultra for pure compute
- Memory limit (24GB) is fine for this experiment
- Most labs have NVIDIA access

**Verdict**: Best performance/cost. Recommended for academic reproduction.

### Option C: Cloud (AWS/GCP)

| Service | GPU | Hourly Cost | Est. Time/20 Gen |
|---------|-----|-------------|------------------|
| AWS p4d.24xlarge | 8× A100 | $32.77/hr | ~30 min |
| AWS g5.xlarge | 1× A10G | $1.01/hr | ~4 hrs |
| GCP a2-highgpu-1g | 1× A100 | $3.67/hr | ~1 hr |

**Estimated Cost for Full Experiment (9 runs × 1000 gens)**:

- AWS g5.xlarge: ~$400
- GCP A100: ~$150

**Verdict**: Accessible for one-time reproduction. Higher ongoing cost.

---

## 4. Code Portability

### Current Stack

```python
# Apple Silicon specific:
jax[metal]          # JAX with Metal backend
torch[mps]          # PyTorch with Metal Performance Shaders
mlx                 # Apple MLX (not used in core experiment)

# Platform-agnostic:
evotorch            # Works everywhere
brax                # Works everywhere  
flax                # Works everywhere
```

### Porting to NVIDIA

```bash
# Remove Apple dependencies
pip uninstall jax-metal

# Install CUDA versions
pip install jax[cuda12]
pip install torch  # Auto-detects CUDA
```

Code changes needed: **ZERO** (JAX abstracts hardware)

### Porting to CPU-only

```bash
pip install jax  # CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Performance**: 10-50× slower, but works for verification.

---

## 5. Optimizing M3 Ultra Utilization

### Current Issue

We're only using ~150-200% CPU (1.5-2 cores) while GPU is saturated.

### Potential Optimizations

1. **Parallel Population Evaluation**

   ```python
   # Current: Sequential evaluation
   for individual in population:
       fitness = evaluate(individual)
   
   # Better: Batched GPU evaluation
   fitnesses = jax.vmap(evaluate)(population)
   ```

   Status: EvoTorch may already do this. Verify with profiling.

2. **Larger Batch Sizes**
   - Population 50 → 200 could better saturate GPU
   - Trade-off: More memory, may change optimization dynamics

3. **Multiple Parallel Runs**
   - With 512GB RAM, could run 10+ experiments simultaneously
   - Each uses ~25GB, total capacity ~20 runs

### Recommendation for This Experiment

**Don't optimize further.** Reasons:

1. Current speed is acceptable for science
2. Changing batch size affects optimization dynamics
3. Changing parallelism affects reproducibility
4. Focus on results, not micro-optimization

---

## 6. Reproducibility Recommendations

### For Paper Publication

Include in supplementary materials:

```
1. Hardware: Apple M3 Ultra Mac Studio (512GB)
2. Software: Python 3.12, JAX 0.4.x, PyTorch 2.x
3. Estimated runtime: ~1 hour per 20 generations
4. Minimum requirements: 16GB VRAM GPU, 32GB RAM
5. Alternative: NVIDIA RTX 3090+ with CUDA JAX
6. Cloud cost estimate: ~$50-150 on GCP A100
```

### For Open Source Release

```bash
# Docker container for reproducibility
docker pull genomic-thesis/experiment:v1.0
docker run --gpus all genomic-thesis/experiment \
    --strategy hierarchical --seed 42 --generations 20
```

---

## 7. Summary

| Aspect | Status |
|--------|--------|
| GPU Utilization | ✅ Fully utilized |
| RAM Utilization | ⚠️ 5% used (by design) |
| CPU Utilization | ⚠️ Low (by design) |
| Workload Type | GPU-bound compute |
| Portability | ✅ Easy (JAX abstracts hardware) |
| Cheapest Reproduction | NVIDIA RTX 4090 (~$2K) |
| Fastest Reproduction | Cloud 8×A100 (~$30/hr) |
| Minimum Viable | M4 Mac Mini 64GB (~$1.5K) |

**Key Message for Paper**: This experiment can be reproduced on any modern GPU system, not just Apple Silicon. The M3 Ultra cluster provides convenience, not unique capability.

---

## 8. Future Directions

### Consumer GPU Validation (RTX 5090)

The user plans to validate results on an NVIDIA RTX 5090 (32GB VRAM). This comparison would:

1. **Verify result reproducibility** across hardware platforms
2. **Benchmark timing** - single RTX 5090 may outperform 5-node M3 Ultra cluster
3. **Validate JAX portability** - confirm zero code changes needed

**Expected outcome**: RTX 5090 likely 2-5× faster due to:

- ~21,000 CUDA cores vs 76 Apple GPU cores per node
- No distributed overhead (single card vs multi-node)
- Mature CUDA/cuDNN optimization

### Why This Experiment is Unique

| Workload | VRAM Required | Best Platform |
|----------|---------------|---------------|
| LLM Inference (70B+) | 40-140 GB | Apple Silicon (unified memory) |
| LLM Training | 100+ GB | Multi-GPU clusters |
| **This Experiment** | **~10 GB** | **Consumer GPU (RTX 3090+)** |

Unlike LLM workloads that require the massive unified memory of Apple Silicon, this experiment evolves a **small hypernetwork** (0.6-5M params). This makes it:

- Reproducible by grad students with gaming PCs
- Validatable on cloud spot instances cheaply
- Accessible for broad scientific verification

### Optimization Roadmap

1. **Current**: Single-node execution, Gen 0 JIT overhead
2. **Short-term**: Batched population evaluation for better GPU saturation
3. **Medium-term**: Multi-GPU data parallelism (multiple seeds simultaneously)
4. **Future**: NVIDIA CUDA port for benchmark comparison

---

## 9. NVIDIA Setup Guide (For Future Reference)

```bash
# 1. Create virtual environment
python3 -m venv ~/.venv/genomic_cuda
source ~/.venv/genomic_cuda/bin/activate

# 2. Install CUDA JAX (instead of Metal)
pip install jax[cuda12]
pip install torch  # Auto-detects CUDA
pip install evotorch brax flax optax

# 3. Clone experiment code
git clone <experiment_repo>
cd genomic_evo

# 4. Run with same seed for reproducibility
python main.py --strategy hierarchical --seed 42 --generations 20 --single-node

# 5. Compare results with Apple Silicon baseline
diff results_hierarchical_42.json ../apple_baseline/results_hierarchical_42.json
```

**Expected**: Identical fitness values (within floating-point tolerance), faster runtime.
