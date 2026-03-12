# Bandwidth Efficiency Analysis: 10GbE vs TB5 RDMA

## Simulation Requirements

### Brax Ant-v4 Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Observation Space** | 27 dimensions | Joint angles, angular velocities, torso orientation |
| **Action Space** | 8 dimensions | Continuous torques for 8 actuators (4 legs × 2 joints) |
| **Episode Length** | 1000 timesteps | ~20 seconds at 50 Hz physics |
| **Parallel Environments** | 32 per node | Vectorized across GPU cores |

### Node Requirements

**Current Setup**: 5 nodes (muladhara, svadhisthana, manipura, anahata, vishuddha)

**Rationale**:

- Each M3 Ultra has 128 GB unified memory
- Brax environment state: ~(27 + 8) × 32 envs × 1000 steps × 4 bytes ≈ **4.5 MB per rollout batch**
- With JAX JIT compilation, memory usage peaks at ~8 GB per node for 32 parallel envs
- **5 nodes is sufficient** with headroom for other processes

**Scaling**: Could run on 1-2 nodes for prototyping, but 5-node distributed evolution provides:

- **5× faster generation turnover** (population sharded across nodes)
- **Genetic diversity** (independent random seeds per node)

---

## Bandwidth Analysis

### Data Transfer Volume per Generation

#### SNES Population Parameters

| Strategy | G-net Params | Phenotype Params | Total per Individual |
|----------|-------------|------------------|---------------------|
| **Flat** | ~10,000 | 10,000 (direct weights) | **10,000 floats** |
| **Hierarchical** | ~65,000 | 8 × 128 = 1,024 (blocks) | **65,000 floats** |
| **Topological** | ~260,000 | 256 × 256 = 65,536 (adjacency) | **260,000 floats** |

**Population Size**: 200 individuals (typical for SNES)

**Data per Generation**:

- **Flat**: 200 × 10K × 4 bytes = **8 MB**
- **Hierarchical**: 200 × 65K × 4 bytes = **52 MB**
- **Topological**: 200 × 260K × 4 bytes = **208 MB**

### Network Transfer Requirements

Each generation requires:

1. **Fitness Sync** (dominant cost):
   - Each node evaluates 200/5 = 40 individuals locally
   - Sends 40 fitness values (40 × 4 bytes = **160 bytes**)
   - **All-gather** across 5 nodes: 160 × 5 = **800 bytes total**
   - **BUT**: Population weights must also be shared for SNES update step

2. **Population Distribution** (if using global SNES):
   - Coordinator broadcasts new population: **8-208 MB per generation**
   - Each node receives full population to shard locally

**Effective Transfer per Generation**: **8-208 MB** (depends on strategy)

### Bandwidth Comparison

| Metric | 10GbE | TB5 RDMA | Speedup |
|--------|-------|----------|---------|
| **Raw Bandwidth** | 10 Gbps (1.25 GB/s) | 120 Gbps (15 GB/s) | **12×** |
| **Latency** | ~100 μs (switched) | ~1 μs (direct P2P) | **100×** |
| **Flat Strategy (8 MB/gen)** | 6.4 ms | 0.5 ms | 12× |
| **Hierarchical (52 MB/gen)** | 42 ms | 3.5 ms | 12× |
| **Topological (208 MB/gen)** | 166 ms | 14 ms | 12× |

### Total Overhead per Generation

**Computation Time** (dominant):

- Brax Ant rollout (1000 steps × 32 envs): ~100-500 ms per individual (GPU-accelerated)
- Population of 40 individuals per node: ~4-20 seconds

**Communication Time**:

- **10GbE**: 6-166 ms per generation
- **TB5**: 0.5-14 ms per generation

**Overhead Percentage**:

- **10GbE**: 0.3% - 4% of total time (compute-bound)
- **TB5**: 0.025% - 0.3% of total time (negligible)

### Efficiency Loss on 10GbE

For **5000 generations**:

- **Flat**: 32 seconds (10GbE) vs 2.5 seconds (TB5) → **29.5 sec slower**
- **Hierarchical**: 3.5 minutes vs 17.5 seconds → **3.3 min slower**
- **Topological**: 13.8 minutes vs 70 seconds → **12.6 min slower**

**Total Experiment Runtime** (assuming 10 sec/gen for fitness evaluation):

- Compute: 5000 gen × 10 sec = **13.9 hours**
- 10GbE overhead: +13.8 minutes (1.6% slowdown)
- TB5 overhead: +70 seconds (0.1% slowdown)

---

## Conclusion

### 10GbE is **Acceptable** for This Workload

**Reason**: The experiment is **compute-bound** (fitness evaluation dominates), not **communication-bound**.

**Impact**:

- TB5 RDMA would be **12× faster** for network transfers
- But only reduces **total runtime by 1.5%** (13 minutes out of 14 hours)

**When TB5 Matters**:

- **Larger populations** (>1000 individuals): Communication scales linearly
- **Simpler fitness functions** (<10 ms/eval): Network becomes bottleneck
- **Sharded model inference** (tensor parallelism): TB5's low latency critical

**Bottom Line**:

- ✅ **Proceed with 10GbE** for this experiment (ring backend)
- 🔧 **Fix TB5/JACCL** as a nice-to-have optimization (12% faster generations, not game-changing)
- 🎯 **Focus on fitness environment quality** (Brax Ant) for scientifically valid results
