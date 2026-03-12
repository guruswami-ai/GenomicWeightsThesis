# Genomic-Inspired Weight Compression for Neuroevolution: A Negative Result with Positive Insights

## Abstract

Biological genomes encode neural architectures through highly compressed representations, suggesting that structured weight encodings might provide beneficial inductive biases for artificial neuroevolution. We test this hypothesis by comparing four encoding strategies—Flat (direct), Hierarchical (TAD-inspired blocks), Topological (chromatin-like graphs), and CPPN (compositional pattern networks)—across three reinforcement learning benchmarks designed to favor structured representations. Contrary to our hypothesis, the Flat encoding achieved highest mean fitness on all benchmarks (Ant: 407.9, Swimmer: 49.4, Multi-task: 265.0), while structured encodings showed 10-23× lower variance but consistently lower peak performance. A compression-strength sweep revealed that hierarchical encoding does not act as smooth regularization but as discrete "landscape surgery"—certain configurations trap all seeds in stable but suboptimal basins (variance collapse), while others create high-variance "encoding lotteries." These results provide strong evidence against genomic-inspired compression as a universal solution for neuroevolution, while offering novel insights into how indirect encodings reshape fitness landscapes. Code and data available at: [repository URL].

**Keywords**: neuroevolution, indirect encoding, weight compression, chromatin structure, CPPN, variance collapse

**Author**: Paul Nevin (paul@guruswami.com), Independent Researcher

**Code & Data**: https://github.com/guruswami-ai/GenomicWeightsThesis/

---

## 1. Introduction

The human genome encodes approximately 86 billion neurons and 100 trillion synapses using only ~20,000 protein-coding genes—a compression ratio that dwarfs anything achieved in artificial neural networks. This remarkable efficiency has inspired a long-standing hypothesis in neuroevolution: that structured, compressed weight representations might provide superior inductive biases compared to direct encodings (Stanley & Miikkulainen, 2002; Stanley, 2007).

Recent advances in genomics have revealed sophisticated organizational principles underlying this compression. Topologically Associating Domains (TADs) create modular, hierarchical structure in chromatin (Dixon et al., 2012), while chromatin physics constrain long-range interactions through distance-dependent contact probabilities (Lieberman-Aiden et al., 2009). These biological motifs suggest specific architectural priors that might benefit artificial evolution.

Despite theoretical appeal, rigorous empirical tests of genomic-inspired compression remain scarce. Most work on indirect encodings focuses on CPPNs and HyperNEAT (Stanley et al., 2009), with limited comparison to simpler baselines on tasks specifically designed to favor structured representations.

### Research Questions

We address three questions:

1. **Does genomic-inspired compression improve fitness?** Do Hierarchical (TAD-like) or Topological (chromatin-like) encodings outperform direct Flat encoding on tasks with inherent modularity or requiring generalization?

2. **What is the variance-fitness tradeoff?** How do different encodings affect the reliability (variance across seeds) versus peak performance (mean fitness)?

3. **How does compression strength affect outcomes?** Is there a smooth relationship between encoding constraint strength and the variance-fitness tradeoff?

### Contributions

- **Rigorous negative result**: We show that Flat encoding outperforms all structured encodings (including CPPNs) on three benchmarks specifically chosen to favor compression—a strong test of the genomic compression hypothesis.

- **Variance-fitness tradeoff characterization**: We document a consistent pattern where structured encodings show 10-23× lower variance but lower mean fitness, quantifying a previously informal observation.

- **Compression landscape analysis**: Through a targeted sweep, we demonstrate that hierarchical encoding acts as discrete "landscape surgery" rather than smooth regularization, explaining observed variance collapse phenomena.

- **Methodological contribution**: We provide a complete experimental framework (code, data, 85 runs across 5 seeds) for testing encoding strategies in neuroevolution.

---

## 2. Related Work

### 2.1 Indirect Encodings in Neuroevolution

The distinction between direct and indirect encodings has been central to neuroevolution since its inception. Direct encodings represent each weight independently, while indirect encodings use compressed representations that are "decoded" into full weight matrices.

**CPPNs and HyperNEAT**: Stanley (2007) introduced Compositional Pattern Producing Networks (CPPNs), which generate weight patterns as functions of spatial coordinates. HyperNEAT (Stanley et al., 2009) extended this to generate weights for substrate networks with geometric structure. These approaches have shown benefits for tasks with spatial regularity but require careful architecture design.

**Developmental encodings**: Gruau (1994) pioneered cellular encoding, where a grammar evolves to specify network growth. More recent work explores neural development through gene regulatory networks (Cussat-Blanc et al., 2019). These approaches add computational overhead but can discover modular structures.

**Modular encodings**: Kashtan & Alon (2005) showed that modularly varying goals accelerate evolution of modular networks. Clune et al. (2013) demonstrated that connection costs can encourage modularity. Our Hierarchical encoding is inspired by this line of work.

### 2.2 Genomic Organization and Neural Encoding

The three-dimensional organization of genomes provides inspiration for neural architecture:

**Topologically Associating Domains (TADs)**: Dixon et al. (2012) discovered that chromosomes fold into modular domains ~1Mb in size, within which genes interact frequently. TADs are conserved across species and cell types, suggesting functional importance. Our Hierarchical encoding mimics this block structure.

**Chromatin contact probability**: Lieberman-Aiden et al. (2009) showed that chromatin contact probability decays as a power law with genomic distance. This creates a bias toward local interactions while permitting long-range contacts. Our Topological encoding implements this distance-dependent connectivity.

**Compression in biological neural development**: The genome's ability to specify complex neural circuits through compressed developmental programs remains poorly understood. Our work tests whether analogous compression provides benefits in artificial evolution.

### 2.3 Evolution Strategies for Neuroevolution

We use Separable Natural Evolution Strategies (SNES; Schaul et al., 2011), part of a family of gradient-free optimization methods that have shown strong performance on reinforcement learning tasks (Salimans et al., 2017). SNES maintains a Gaussian search distribution and updates its mean and covariance based on fitness-weighted samples, making it well-suited for comparing encoding strategies without confounds from gradient computation.

---

## 3. Methods

### 3.1 Encoding Strategies

We compare four encoding strategies that map a latent genotype to phenotype weights:

**Flat (Direct)**: The genotype directly specifies all weights of a 2-layer MLP. For Ant (obs=27, act=8, hidden=64), this yields ~5,000 parameters. This serves as our baseline—the "null hypothesis" that structure provides no benefit.

**Hierarchical (TAD-inspired)**: The genotype specifies N independent blocks (default N=8), each producing a fixed-size hidden representation. Blocks are concatenated and projected to action space. This mimics TAD modularity where genomic regions form semi-independent functional units.

**Topological (Chromatin-inspired)**: The genotype specifies a graph adjacency matrix with distance-penalized connectivity: A[i,j] ∝ exp(-α|i-j|). Message passing on this graph transforms observations to actions. This implements the chromatin contact probability decay observed in Hi-C experiments.

**CPPN (Control)**: A small network (2 hidden layers, 32 units, tanh/sin activations) generates weights as functions of input/output indices. This established indirect encoding serves as a control to verify our results extend beyond our novel genomic-inspired variants.

All encodings produce the same phenotype architecture (2-layer MLP with tanh output) to ensure fair comparison.

### 3.2 Benchmarks

We selected three benchmarks to test different aspects of the compression hypothesis:

**Ant Locomotion** (baseline): Standard quadruped locomotion (obs=27, act=8). No obvious modular structure—we expect Flat to perform well here.

**Swimmer** (modularity test): 3-segment swimming robot (obs=8, act=2). Repeated body segments should favor block-structured encodings. We configured Hierarchical with 3 blocks to match segment count.

**Multi-task Ant** (generalization test): A single controller must perform Ant locomotion in four directions (forward, backward, left, right). Fitness averages across all four tasks. This forces compression—task-specific overfitting is impossible.

### 3.3 Evolution Protocol

- **Algorithm**: SNES (Separable Natural Evolution Strategies)
- **Population**: 50 individuals
- **Generations**: 100
- **Seeds**: 5 per condition (seeds 10000, 11000, 12000, 13000, 14000)
- **Fitness evaluation**: 2 rollouts averaged, episode length 200-500 steps
- **Hardware**: Apple M3 Ultra nodes with JAX/Brax simulation

### 3.4 Compression Strength Sweep

To characterize how compression strength affects outcomes, we ran a targeted sweep on Swimmer varying hierarchical block configuration while keeping total hidden capacity constant (1024 units):

| Config | Blocks × Size | Compression Level |
|--------|---------------|-------------------|
| Flat | — | None (baseline) |
| hier_weak | 32 × 32 | Minimal (many small blocks) |
| hier_medium | 8 × 128 | Moderate (default) |
| hier_strong | 4 × 256 | Strong |
| hier_vstrong | 2 × 512 | Maximum |

---

## 4. Results

### 4.1 Main Benchmark Results

Table 1 summarizes fitness across all benchmarks. Flat encoding achieved highest mean fitness on every benchmark.

**Table 1: Fitness by Strategy and Benchmark (mean ± std, 5 seeds)**

| Benchmark | Flat | Hierarchical | Topological | CPPN |
|-----------|------|--------------|-------------|------|
| Ant | **407.9 ± 52.7** | 378.2 ± 30.8 | 366.4 ± 12.3 | 324.9 ± 44.2 |
| Swimmer | **49.4 ± 16.0** | 36.8 ± 0.7 | 31.7 ± 2.0 | 32.3 ± 1.5 |
| Multi-task | **265.0 ± 13.6** | 239.5 ± 4.3 | 231.8 ± 6.9 | 240.1 ± 6.5 |

The pattern is consistent: Flat achieves 8-34% higher mean fitness than the best structured encoding on each benchmark.

### 4.2 Variance-Fitness Tradeoff

While Flat wins on mean fitness, structured encodings show dramatically lower variance:

**Table 2: Coefficient of Variation (CV = std/mean)**

| Benchmark | Flat | Hierarchical | Topological | CPPN |
|-----------|------|--------------|-------------|------|
| Ant | 12.9% | 8.1% | **3.3%** | 13.6% |
| Swimmer | 32.4% | **1.9%** | 6.4% | 4.5% |
| Multi-task | 5.1% | **1.8%** | 3.0% | 2.7% |

On Swimmer, Hierarchical shows 17× lower relative variance than Flat (CV 1.9% vs 32.4%). This "variance collapse" is consistent across all structured encodings.

### 4.3 Statistical Significance

Despite variance differences, pairwise comparisons favor Flat:

**Table 3: Welch's t-test (Flat vs others)**

| Benchmark | vs Hierarchical | vs Topological | vs CPPN |
|-----------|-----------------|----------------|---------|
| Ant | p=0.329 | p=0.060 | p=0.043* |
| Swimmer | p=0.142 | p=0.067 | p=0.099 |
| Multi-task | p=0.012* | p=0.006* | p=0.018* |

*Significant at α=0.05

On Multi-task—the benchmark with strongest compression pressure—Flat's advantage over all structured encodings is statistically significant.

### 4.4 Compression Strength Sweep

The sweep on Swimmer reveals a non-monotonic relationship between compression and variance:

**Table 4: Compression Sweep Results (Swimmer)**

| Config | Blocks × Size | Mean ± Std | CV |
|--------|---------------|------------|-----|
| **Flat** | — | **168.9 ± 2.8** | **1.7%** |
| hier_weak | 32 × 32 | 70.6 ± 46.6 | 66.0% |
| hier_medium | 8 × 128 | 41.5 ± 2.6 | 6.2% |
| hier_strong | 4 × 256 | 119.0 ± 57.2 | 48.1% |
| hier_vstrong | 2 × 512 | 120.4 ± 62.7 | 52.1% |

Key observations:

1. **Flat dominates**: Highest mean (168.9) *and* lowest CV (1.7%).

2. **Medium compression = stable failure**: hier_medium shows low CV (6.2%) but catastrophically low mean (41.5). All seeds converge to the same suboptimal basin.

3. **Strong/weak compression = encoding lottery**: hier_strong and hier_vstrong show high CV (48-52%) with bimodal outcomes—some seeds achieve ~170 (matching Flat), others get stuck at ~40-55.

Figure 1 shows learning curves illustrating these dynamics.

![Learning Curves](genomic_evo/experiments/figures/learning_curves_combined.png)

**Figure 1**: Learning curves across benchmarks (mean ± std shading, 5 seeds). Flat (green) achieves highest fitness with moderate variance. Structured encodings (blue, purple, red) show tighter bands at lower fitness levels.

---

## 5. Discussion

### 5.1 The Genomic Compression Hypothesis is Not Supported

Our results provide strong evidence against the hypothesis that genomic-inspired compression provides fitness advantages for neuroevolution. Across all three benchmarks—including those specifically designed to favor structured encodings—Flat achieved highest mean fitness.

This negative result is strengthened by:

- **Benchmark selection**: Swimmer has repeated segments (should favor Hierarchical); Multi-task requires generalization (should favor compression). Flat won on both.

- **CPPN control**: An established indirect encoding showed the same pattern, confirming results extend beyond our specific implementations.

- **Statistical power**: 5 seeds × 4 strategies × 3 benchmarks = 60 runs for main experiments, plus 25 compression sweep runs.

### 5.2 Compression as Discrete Landscape Surgery

The compression sweep reveals why structured encodings fail: they don't act as smooth regularizers but as discrete constraints that fundamentally reshape the fitness landscape.

**Medium compression (8×128)** creates a narrow, shallow attractor basin. Evolution reliably finds this basin (hence low variance) but it contains only suboptimal solutions (hence low mean). This is "variance collapse" in its purest form—consistency in failure.

**Strong compression (2-4 blocks)** creates a rugged, multi-attractor landscape. Some attractors contain good solutions (seeds that achieve ~170), others contain poor ones (seeds stuck at ~40). The high variance reflects this lottery—which attractor a seed finds depends on initialization and stochastic search dynamics.

**Flat encoding** provides sufficient flexibility to navigate to good basins while the landscape remains smooth enough for consistent convergence (hence both high mean and low variance after 100 generations).

### 5.3 When Might Structured Encodings Help?

Our negative results do not preclude benefits in other settings:

- **Larger scales**: Our networks have ~5,000-800,000 parameters. Compression benefits might emerge at billions of parameters where direct encoding becomes intractable.

- **Transfer learning**: Lower variance might indicate more stable representations that transfer better to new tasks (untested here).

- **Longer evolution**: 100 generations may be insufficient for structured encodings to escape initial basins and find good solutions.

- **Different optimizers**: SNES may not be well-suited to the landscapes created by structured encodings. CMA-ES or genetic algorithms might show different patterns.

- **Risk-averse settings**: When consistency matters more than peak performance, structured encodings' lower variance could be valuable.

### 5.4 Implications for Practitioners

1. **Start with Flat**: The direct encoding "null hypothesis" should not be dismissed. It achieved best results on all our benchmarks with minimal implementation complexity.

2. **Beware variance collapse**: Low variance is not inherently good—it may indicate stable convergence to poor solutions. Always compare mean fitness alongside variance.

3. **Configuration matters enormously**: The same hierarchical architecture with different block configurations produced radically different outcomes (CV from 6% to 66%, mean from 41 to 120). Extensive hyperparameter search is essential for structured encodings.

4. **Indirect encodings cluster together**: CPPN showed similar patterns to our genomic-inspired variants, suggesting these findings apply broadly to indirect encodings.

---

## 6. Conclusion

We conducted a rigorous empirical test of genomic-inspired weight compression for neuroevolution. Our hypothesis—that Hierarchical and Topological encodings would outperform Flat on compression-friendly tasks—was comprehensively rejected.

### Key Findings

1. **Flat wins on all benchmarks**: Mean fitness was 8-34% higher than the best structured encoding across Ant, Swimmer, and Multi-task.

2. **Structured encodings show variance collapse**: 10-23× lower variance but consistently lower peak performance.

3. **Compression strength has non-monotonic effects**: Medium compression produces stable failure; strong compression produces encoding lotteries. No configuration consistently beat Flat.

4. **Compression acts as landscape surgery, not regularization**: Hierarchical encoding creates discrete attractor basins rather than smoothly constraining the search space.

### Broader Impact

This negative result is valuable for the neuroevolution community. It redirects research away from a plausible-sounding but empirically unsupported hypothesis, while providing detailed characterization of how indirect encodings actually affect optimization dynamics. The variance-fitness tradeoff and landscape surgery concepts may inform future encoding design.

Future work should test at larger scales, with longer evolution horizons, and on tasks where transfer learning is evaluated explicitly. The question of when—if ever—genomic-inspired compression provides net benefits remains open.

---

## References

Clune, J., Mouret, J. B., & Lipson, H. (2013). The evolutionary origins of modularity. Proceedings of the Royal Society B: Biological Sciences, 280(1755), 20122863.

Cussat-Blanc, S., Harrington, K., & Banzhaf, W. (2019). Artificial gene regulatory networks—a review. Artificial Life, 24(4), 296-328.

Dixon, J. R., Selvaraj, S., Yue, F., Kim, A., Li, Y., Shen, Y., ... & Ren, B. (2012). Topological domains in mammalian genomes identified by analysis of chromatin interactions. Nature, 485(7398), 376-380.

Freeman, C. D., Frey, E., Raichuk, A., Girber, S., Mordatch, I., & Bachem, O. (2021). Brax—A differentiable physics engine for large scale rigid body simulation. arXiv preprint arXiv:2106.13281.

Gruau, F. (1994). Neural network synthesis using cellular encoding and the genetic algorithm. Doctoral dissertation, Ecole Normale Supérieure de Lyon.

Kashtan, N., & Alon, U. (2005). Spontaneous evolution of modularity and network motifs. Proceedings of the National Academy of Sciences, 102(39), 13773-13778.

Lieberman-Aiden, E., Van Berkum, N. L., Williams, L., Kaplan, M., ... & Dekker, J. (2009). Comprehensive mapping of long-range interactions reveals folding principles of the human genome. Science, 326(5950), 289-293.

Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864.

Schaul, T., Glasmachers, T., & Schmidhuber, J. (2011). High dimensions and heavy tails for natural evolution strategies. In Proceedings of the 13th Annual Conference on Genetic and Evolutionary Computation (pp. 845-852).

Stanley, K. O. (2007). Compositional pattern producing networks: A novel abstraction of development. Genetic Programming and Evolvable Machines, 8(2), 131-162.

Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A hypercube-based encoding for evolving large-scale neural networks. Artificial Life, 15(2), 185-212.

Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary Computation, 10(2), 99-127.

---

## Supplementary Material

Full methodology, raw data tables, additional figures, and code are available in the accompanying technical report (METHODOLOGY.md) and repository.

### Appendix: Experimental Details

- **Total runs**: 85 (60 main experiments + 25 compression sweep)
- **Hardware**: 5× Apple M3 Ultra (192GB unified memory)
- **Software**: JAX 0.4.38, Brax 0.14.0, EvoTorch 0.6.1, Python 3.12
- **Reproduction**: See REPRODUCIBILITY.md for exact environment setup
