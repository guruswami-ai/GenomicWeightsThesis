# Paper Review Plan: The Genomic Weight Thesis

This plan outlines the structure and key points for the review of the "Genomic Weight Thesis" paper.

## Proposed Changes

### [NEW] Review of Genomic Weight Thesis

I will create a comprehensive review document that evaluates the paper across several dimensions:

#### 1. Conceptual Rigor

- Analyze the transition from the "Scaling Hypothesis" critique to the "Relational Compression" solution.
- Evaluate the mapping of LLM token relationships to gene-regulatory relationships.

#### 2. Technical Grounding

- Verify the use of the Zador Paradox and Rao 2014 Hi-C findings.
- Check the feasibility of the proposed 3D folding metrics (P(s)~s⁻¹).

#### 3. Experimental Design

- Critique the "9 Total Runs" protocol for detecting "Chromatin Signatures."
- Assess the validity of "Spectral Clustering" as a proxy for TAD detection in artificial networks.
- Evaluate the "Generalization Gap" metric in the context of the proposed environments.

#### 4. Practical Feasibility

- Confirm that the M3 Ultra Cluster with RDMA/JACCL is optimized for this specific workload (population-parallel evolutionary runs).

### [MODIFY] [distributed.py](file:///Users/paul/genomic_evo/distributed.py)

Implemented sharded population evaluation using a custom `ShardedProblem` subclass. Fitnesses are synchronized across the cluster using JACCL/MLX `all_gather`.

### [MODIFY] [phenotype_net.py](file:///Users/paul/genomic_evo/phenotype_net.py)

Refined for parameter-free execution, using weights emitted by the g-net. Added robust shape handling and padding to ensure compatibility with various environment observations.

## Verification Plan

### Automated Tests

- [x] Cluster connectivity test using `mlx.launch` and `test_mlx.py`.
- [x] Single-generation prototype run on `muladhara` coordinator node.
- [x] Sharded evaluation and global fitness synchronization verified via logs.

### Manual Verification

- Verified Hi-C style heatmap generation and basic fitness logging on `muladhara`.

## Experimental Execution

### 1. Cluster Setup

- [x] Create `genomic_evo` conda environment on all compute nodes.
- [x] Install `jax`, `mlx`, `evotorch`, `flax`, `optax`, `wandb`.
- [x] Sync source code to all 5 compute nodes.

### 2. Prototype Launch

- [x] Run `run_experiment.sh` to trigger the sharded-batch prototype across the cluster.
- [x] Verify sharding, cross-node sync, and fitness logging.

### 3. Full 9-Run Suite

- [/] Launch the full suite on the 5-node M3 cluster using RDMA/JACCL.
- [ ] Monitor logs and resource utilization (GPU, Unified Memory).
