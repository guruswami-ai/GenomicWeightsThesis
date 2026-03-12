# Experimental Design: Protocol A

## The Zador-Krakauer In Silico Bottleneck

---

## 1. Hypothesis Under Test

### 1.1 Core Claim (from info.md §7.1)

> "Compressing a neural network through a 'Genomic Bottleneck' spontaneously generates 'Topological Weights' (modular/TAD-like structures)."

### 1.2 Testable Predictions

| ID | Prediction | Observable Metric | Success Criterion |
|----|------------|-------------------|-------------------|
| **P1** | Bottleneck compression forces emergence of modular structure | Spectral modularity (Q) of phenotype weight adjacency | Q_topological > Q_flat at p < 0.05 |
| **P2** | Smaller genotypes achieve comparable fitness through better compression | Fitness per genotype parameter | (Fitness/Params)_topological > (Fitness/Params)_flat |
| **P3** | Hierarchical strategies produce TAD-like block-diagonal patterns | Insulation score distribution | Matches biological Hi-C statistical profile |

### 1.3 Null Hypothesis (Equally Valid Outcome)

**H₀**: Compression strategy has no effect on emergent structure or fitness efficiency. All strategies perform equivalently when normalized for parameter count.

> **Important**: A null result would suggest that topological constraints are NOT a necessary consequence of genomic bottleneck compression. This would refute a core claim of the thesis and is a scientifically valuable outcome.

---

## 2. Experimental Design

### 2.1 Independent Variable: Compression Strategy

| Strategy | Description | Genotype→Phenotype Mapping | Parameter Count |
|----------|-------------|---------------------------|-----------------|
| **Flat** (Control) | Direct 1:1 encoding | Genotype emits raw weight vector | ~5.2M |
| **Hierarchical** | Bottleneck + modular blocks | Genotype emits 8 TAD-like blocks + projection | ~1.2M |
| **Topological** | Bottleneck + graph structure | Genotype emits adjacency matrix + projection | ~0.6M |

### 2.2 Dependent Variables

1. **Fitness** (Primary): Mean episode reward in Brax Ant-v4 physics environment
2. **Modularity** (Secondary): Spectral modularity Q of phenotype adjacency
3. **Efficiency** (Secondary): Fitness normalized by genotype parameter count
4. **Convergence Rate**: Generations to reach fitness threshold

### 2.3 Control Variables (Held Constant)

- Environment: Brax Ant-v4 (27-dim obs, 8-dim action, 200 timesteps)
- Evolution algorithm: SNES (EvoTorch) with identical hyperparameters
- Population size: 50 individuals per generation
- Number of generations: 1000 (or until convergence)
- Hardware: M3 Ultra Mac Studio (identical across all runs)
- Random seed: Varied for replicates, recorded for reproducibility

---

## 3. Code-to-Hypothesis Mapping

### 3.1 How Code Implements the Thesis

```
info.md §7.1 Requirement          │ Code Implementation
──────────────────────────────────┼─────────────────────────────────────────
"Phenotype Network (large ANN)"   │ PhenotypeNet in phenotype_net.py
                                  │ - Inputs: 27-dim observation
                                  │ - Outputs: 8-dim action
                                  │ - Weights: Derived from genotype output
──────────────────────────────────┼─────────────────────────────────────────
"Genotype Network (Hypernetwork)" │ FlatCompressor, HierarchicalCompressor,
                                  │ TopologicalCompressor in genotype_nets.py
                                  │ - Input: 128-dim latent z
                                  │ - Output: Phenotype weight structure
──────────────────────────────────┼─────────────────────────────────────────
"Topological Cost penalizing      │ chromatin_loss() in topo_loss.py
long-range connections"           │ - Penalizes connections where |i-j| is large
                                  │ - Mimics chromatin loop cost
──────────────────────────────────┼─────────────────────────────────────────
"Evolve using Evolutionary        │ SNES algorithm in distributed.py
Strategies over thousands of      │ - Population-based optimization
generations"                      │ - Fitness = Brax Ant episode reward
──────────────────────────────────┼─────────────────────────────────────────
"Apply TAD-calling algorithms     │ analysis.py (to be implemented)
to the weight matrix"             │ - Insulation score calculation
                                  │ - Spectral modularity Q
                                  │ - Hi-C-style visualization
```

### 3.2 Genotype Architecture Details

**Flat (Control)**:

```python
# Direct encoding: 128 → 512 → 10000 weights
# No compression bottleneck
# Baseline for comparison
```

**Hierarchical (TAD-like)**:

```python
# Modular encoding: 128 → 8 blocks × 128 dims each
# Bottleneck ratio: 1024 → 8192 (8× compression)
# Mimics TAD domain structure
```

**Topological (Chromatin-like)**:

```python
# Graph encoding: 128 → 64×64 adjacency + 64×8 projection
# Bottleneck ratio: 128 → 4096 + 512 (36× compression vs flat)
# Distance-biased adjacency (local connections cheaper)
```

---

## 4. Experimental Protocol

### 4.1 Phase 1: Comparative Pilot (20 Generations)

**Purpose**: Validate code works, establish baseline learning curves

| Run | Strategy | Node | Generations | Replicates |
|-----|----------|------|-------------|------------|
| 1.1 | Flat | muladhara | 20 | 4 (completed) |
| 1.2 | Hierarchical | svadhisthana | 20 | 3 |
| 1.3 | Topological | manipura | 20 | 3 |

**Analysis**: Plot fitness vs generation for all strategies. Early divergence indicates effect; overlap suggests null result.

### 4.2 Phase 2: Full Convergence Study (1000 Generations)

**Purpose**: Run until convergence to measure final fitness and structure

| Run | Strategy | Generations | Replicates | Seeds |
|-----|----------|-------------|------------|-------|
| 2.1 | Flat | 1000 | 3 | 42, 123, 456 |
| 2.2 | Hierarchical | 1000 | 3 | 42, 123, 456 |
| 2.3 | Topological | 1000 | 3 | 42, 123, 456 |

**Analysis**:

- ANOVA across strategies
- Post-hoc Tukey HSD for pairwise comparisons
- Effect size (Cohen's d) for magnitude

### 4.3 Phase 3: Structural Analysis

**Purpose**: Apply TAD-calling algorithms to evolved phenotype networks

**Metrics**:

1. **Spectral Modularity (Q)**: Measures block-diagonal structure
2. **Insulation Score**: Measures boundary strength between modules
3. **Contact Probability Decay**: P(s) ~ s^-γ (biological: γ ≈ 1)

---

## 5. Statistical Analysis Plan

### 5.1 Primary Analysis

**Question**: Do compression strategies differ in final fitness?

```
H₀: μ_flat = μ_hierarchical = μ_topological
H₁: At least one mean differs

Test: One-way ANOVA (α = 0.05)
     If significant: Post-hoc Tukey HSD
     
Power: With 3 replicates per strategy, we can detect 
       effect size d ≥ 2.0 with 80% power
```

### 5.2 Secondary Analysis

**Question**: Does compression improve efficiency?

```
Metric: Fitness / Parameters (normalized efficiency)

Expected (if thesis holds):
  Topological: ~0.6M params, high fitness → highest efficiency
  Flat: ~5.2M params, moderate fitness → lowest efficiency
```

### 5.3 Handling Null Results

If ANOVA p > 0.05:

1. **Calculate equivalence bounds**: Can we conclude strategies are equivalent (TOST procedure)?
2. **Report effect sizes**: Even non-significant effects may show trends
3. **Discuss biological implications**: Perhaps genomic bottleneck requires additional constraints not captured in our model

---

## 6. Reproducibility Requirements

### 6.1 Code Versioning

```bash
# Record exact code state
git log --oneline -1  # Commit hash
pip freeze > requirements.txt
```

### 6.2 Random Seed Recording

```python
# Every run logs its seed
SEED = int(os.environ.get('RANDOM_SEED', 42))
jax.random.PRNGKey(SEED)
torch.manual_seed(SEED)
print(f"Run started with seed={SEED}")
```

### 6.3 Data Outputs

For each run, save:

- `fitness_curve.csv`: Generation, avg_fitness, max_fitness, std
- `final_weights.npz`: Evolved genotype parameters
- `phenotype_structure.npz`: Generated phenotype weights for analysis
- `run_config.json`: All hyperparameters and seeds

### 6.4 Reproduction Instructions

```bash
# Clone repository
git clone <repo> && cd genomic_evo

# Install dependencies
pip install -r requirements.txt

# Run specific strategy with seed
python main.py --strategy hierarchical --generations 1000 \
               --seed 42 --single-node > results_hier_s42.log

# Verify identical results
# (Should match within floating-point tolerance)
```

---

## 7. Expected Outcomes

### 7.1 If Thesis is Supported

| Observation | Interpretation |
|-------------|----------------|
| Topological fitness > Flat fitness | Compression improves generalization |
| Topological has higher modularity Q | Structure emerges from bottleneck |
| Topological matches Hi-C patterns | Biological analogy confirmed |

### 7.2 If Thesis is Refuted (Null Result)

| Observation | Interpretation |
|-------------|----------------|
| All strategies have equal fitness | Compression provides no advantage |
| No modular structure in topological | TAD-like patterns don't emerge naturally |
| Random structure in all conditions | Topological weights not a bottleneck consequence |

> **Both outcomes are scientifically valuable.** A null result would indicate that the Zador bottleneck conjecture requires additional biological constraints (e.g., specific topological costs, developmental dynamics) not captured in our simplified model.

---

## 8. Current Status

### 8.1 Completed

- [x] Architecture implementation: All 3 compression strategies
- [x] Architecture validation: All strategies pass forward-pass test
- [x] Flat baseline (Phase 1): 4 replicates, Gen 0-20 completed
  - Gen 0 → Gen 20 fitness improvement confirmed
  - Proves evolution algorithm works

### 8.2 In Progress

- [ ] Fix hierarchical/topological initialization (Flax param issue - RESOLVED)
- [ ] Launch comparative 20-gen pilots for hierarchical/topological
- [ ] Sync fixed code to all cluster nodes

### 8.3 Next Steps

1. Launch hierarchical 20-gen pilot (svadhisthana, manipura)
2. Launch topological 20-gen pilot (anahata, vishuddha)
3. Compare learning curves across all 3 strategies
4. If divergence observed: Proceed to Phase 2 (1000 generations)
5. If convergence: Report null result and analyze implications

---

## 9. References to info.md

This experiment implements:

- **§7.1 Protocol A**: The Zador-Krakauer In Silico Bottleneck
- **§3.2**: The Bottleneck as a Regularizer
- **§4.1**: Polymer Physics and the Partition Function (simplified)
- **§6.1**: The Relational Manifold of the Genome (adjacency matrix)

The topological loss function (chromatin_loss) implements:

- **§4.1**: Distance penalty (local connections cheaper)
- **§4.3**: Loop extrusion cost approximation
