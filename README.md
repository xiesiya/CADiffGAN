# CADiffGAN: Causal Attention-Guided Diffusion WGAN with a Meta-Graph for Drug–miRNA Association Prediction

**Predicting drug–miRNA response associations on a miRNA–lncRNA–drug heterogeneous
graph through a causal attention meta-graph encoder, an attention-guided diffusion
regularizer, and WGAN-based adversarial training.**

## Availability

- **Source code:** <https://github.com/xiesiya/CADiffGAN>
- **Dataset:** <https://github.com/xiesiya/mixDR>
- **License:** MIT (see [LICENSE](LICENSE)).
- **Contact:** siya.xie@mpu.edu.mo

## Overview

CADiffGAN is a deep learning framework for drug–miRNA association (DMA) prediction,
built on a self-constructed miRNA–lncRNA–drug heterogeneous graph (the *mixDR*
benchmark). It combines three components:

1. **Causal Attention Meta-Graph Encoder**
   - Discovers optimal information-aggregation paths in the heterogeneous graph
     through DARTS-style bilevel optimization, without hand-crafted meta-paths.
   - Applies causal attention: at each meta-graph state, historical path choices are
     attended to with a causal mask (future states hidden), augmented by edge-type
     embeddings so the model learns which relation types are causally relevant.
   - Produces a per-node causal attention score used to guide diffusion.

2. **Attention-Guided Diffusion**
   - A DDPM forward process (linear noise schedule, `T = 1000`) applied to miRNA
     embeddings.
   - Instead of one shared random timestep for all nodes, the per-node causal
     attention score determines its noise level: high-attention (informative,
     well-connected) nodes receive less noise, while low-attention (sparse) nodes
     receive more noise for stronger regularization.

3. **WGAN-based Adversarial Training with a Cost-Sensitive Discriminator**
   - Wasserstein discriminator on `[adjacency_row, embedding]` pairs; weight clipping
     is used by default, and `--use_gp` switches to the WGAN-GP gradient penalty.
   - Cost-sensitive weighting (default `cFN:cFP = 1:1`, configurable via
     `--cost_fn` / `--cost_fp`) to address class imbalance.

## Dataset

The self-constructed mixDR graph integrates five public databases: ncDR, SM2miR,
RNAInter, NoncoRNA, and ncRNADrug. The dataset is available at
<https://github.com/xiesiya/mixDR>.

### Statistics

| Category | Count |
|----------|-------|
| miRNA nodes | 547 |
| lncRNA nodes | 1,203 |
| drug nodes | 1,326 |
| Total nodes | 3,076 |
| miRNA–drug associations | 7,435 |
| lncRNA–drug associations | 16,504 |
| miRNA–lncRNA associations | 18,399 |
| Total associations | 42,338 |

### Files

| File | Description |
|------|-------------|
| `miRNA_drug_associations.xlsx` | miRNA–drug associations |
| `lncRNA_drug_associations.xlsx` | lncRNA–drug associations |
| `miRNA_lncRNA_associations.xlsx` | miRNA–lncRNA interactions |
| `miRNA_sequences.xlsx` | miRNA sequence information |
| `lncRNA_sequences.xlsx` | lncRNA sequence information |
| `drug_smiles.xlsx` | Drug SMILES representations |
| `miRNA_similarity_matrix.xlsx` | miRNA–miRNA functional similarity matrix |
| `lncRNA_similarity_matrix.xlsx` | lncRNA–lncRNA functional similarity matrix |

### Feature extraction

- **miRNA:** k-mer sequence features (64-d) + secondary structure features (4-d) +
  functional similarity features, fused to 128-d.
- **lncRNA:** k-mer sequence features (64-d) + secondary structure features (4-d) +
  sequence similarity features, fused to 128-d.
- **drug:** Morgan + MACCS + Daylight molecular fingerprints, attention-fused to
  128-d.

If `biopython` / `rdkit` are unavailable, the loader falls back to identity /
one-hot features so the pipeline still runs.

## Installation

Requires Python 3.10 (NumPy < 2).

```bash
git clone https://github.com/xiesiya/CADiffGAN.git
cd CADiffGAN
pip install -r requirements.txt
```

Download the dataset from <https://github.com/xiesiya/mixDR> and place its contents
under `./dataset/`.

## Quick Start

```bash
# Train with the default configuration (uses ./dataset, saves to ./weights)
python main.py

# Use a custom dataset directory
python main.py --data_dir /path/to/dataset
```

## Evaluation Protocol

- **Five-fold cross-validation with ensemble.** All known positive miRNA–drug
  associations are randomly partitioned into five equal folds; in each round one
  fold is held out as the test set, and the remaining four folds are split into
  training (90%) and validation (10%). After five rounds every association has been
  used for testing exactly once. Final predictions are the average of the five fold
  models.
- **Sampling settings.** Results are reported under four positive:negative ratios:
  balanced 1:1, imbalanced 1:10, extremely imbalanced 1:20, and very extreme 1:30.
- **Metrics:** AUC, AUPR, ACC, F1, MCC.

## Reported Results

Numbers below are as reported in the manuscript; they correspond to the released
code configuration (300 epochs, batch size 64, five-fold CV + ensemble,
NVIDIA RTX 5090).

### CADiffGAN across sampling ratios

| Ratio | AUC | AUPR | ACC | F1 | MCC |
|-------|------|------|------|------|------|
| 1:1 | 0.9856 | 0.9834 | 0.9495 | 0.9512 | 0.9013 |
| 1:10 | 0.9837 | 0.8536 | 0.9568 | 0.7773 | 0.7554 |
| 1:20 | 0.9830 | 0.7592 | — | 0.7004 | — |
| 1:30 | 0.9831 | 0.7016 | — | 0.6487 | — |

Under 1:10, CADiffGAN outperforms the strongest baseline (SubKNet, AUPR 0.8416) by
1.4% absolute in AUPR. AUC stays near 0.983 across all ratios.

### Ablation (balanced 1:1)

| Variant | AUC | Relative ΔAUC |
|---------|------|---------------|
| Full CADiffGAN | 0.9856 | — |
| w/o Causal Attention | — | −1.0% |
| w/o Attention-guided Diffusion | 0.9739 | −1.2% |
| w/o BCE Loss | 0.9716 | −1.4% |
| w/o Multi-scale | 0.9786 | −0.7% |

Variant flags:

1. **w/o Causal Attention (fixed paths):** fix the meta-graph paths instead of
   learning them via causal attention and architecture search.
2. **w/o Attention-guided Diffusion:** `--attn_diff_weight 0` (single random
   timestep for all nodes).
3. **w/o BCE Loss:** `--bce_weight 0`.
4. **w/o Multi-scale:** `--n_cells 1`.

## File Structure

```
CADiffGAN/
├── config.py                 # All hyperparameters (argparse defaults)
├── data_loader.py            # Heterogeneous graph loader, five-fold CV split
├── causal_encoder.py         # Causal attention meta-graph encoder (DARTS-style)
├── diffusion.py              # DDPM schedule + attention-guided timestep
├── model.py                  # Generator + Discriminator
├── train.py                  # WGAN + bilevel training loop, five-fold CV + ensemble
├── utils.py                  # Metrics, seed, utilities
├── main.py                   # Entry point
├── compute_stats.py          # Dataset statistics computation
├── compute_all_sparsity.py   # Sparsity comparison across datasets
├── requirements.txt          # Python dependencies
├── dataset/                  # Dataset directory (populated from the mixDR repository)
└── weights/                  # Saved checkpoints and ensemble predictions (generated at runtime)
```

## Key Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `./dataset` | Path to dataset directory |
| `--n_hid` | 64 | Hidden dimension |
| `--n_steps` | 3 | Meta-graph state transitions per cell |
| `--n_cells` | 2 | Number of meta-graph cells (multi-scale) |
| `--attn_dim` | 32 | Causal attention internal dimension |
| `--edge_emb_dim` | 32 | Edge type embedding dimension |
| `--dropout` | 0.2 | Dropout rate |
| `--timesteps` | 1000 | DDPM diffusion steps T |
| `--beta_min` / `--beta_max` | 1e-4 / 2e-2 | Linear noise schedule bounds |
| `--attn_diff_weight` | 1.0 | Attention-to-diffusion guidance (0 = random t) |
| `--g_lr` / `--d_lr` | 1e-4 | Generator / discriminator learning rate |
| `--arch_lr` | 3e-4 | Architecture (alpha) learning rate |
| `--g_steps` / `--d_steps` | 5 / 2 | GAN updates per epoch |
| `--batch_size` | 64 | miRNA rows per batch |
| `--epochs` | 300 | Training epochs |
| `--alpha` | 0.1 | Reconstruction (MSE) loss weight |
| `--bce_weight` | 1.0 | Auxiliary BCE loss weight |
| `--cost_fn` / `--cost_fp` | 1.0 / 1.0 | Discriminator cost weights (cFN / cFP) |
| `--use_gp` | off | Use WGAN-GP gradient penalty instead of weight clipping |
| `--weight_clip` | 0.01 | Weight clipping bound (when `--use_gp` is off) |
| `--grad_clip` | 5.0 | Gradient clipping max norm (generator) |
| `--eps_start` / `--eps_decay` | 0.8 / 0.95 | Epsilon-greedy path sampling schedule |
| `--arch_warmup` | 30 | Warmup epochs with fixed argmax architecture |
| `--test_ratio` / `--val_ratio` | 0.2 / 0.1 | Hold-out / validation split ratios |
| `--n_folds` | 5 | Number of cross-validation folds |
| `--sim_top_k` | 5 | Nearest neighbours for similarity graphs |
| `--seed` | 2026 | Random seed |

## Reproducibility Notes

- Training checkpoints (`weights/`) and analysis outputs are generated locally at
  runtime and are not part of the released source archive.
- Random seed is fixed (`--seed 2026`); data splits use `KFold` with the same seed.
- CPU inference is available via `--device cpu`; training on CPU is not recommended.

## Citation

If you use this code or dataset, please cite:

```
@article{cadiffgan,
  title   = {CADiffGAN: Causal Attention-Guided Diffusion WGAN with Meta-Graph for Drug--miRNA Association Prediction},
  author  = {Xie, Siya and Law, K. L. Eddie},
  journal = {Bioinformatics},
  year    = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
