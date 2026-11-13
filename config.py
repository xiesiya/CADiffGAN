import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description='CADiffGAN: Causal Attention-guided Diffusion GAN')

    parser.add_argument('--data_dir', type=str,
                        default='./dataset',
                        help='Path to dataset directory containing association '
                             'and feature files')
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='Output feature dimension per node type')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Fraction of miRNA-drug positives held out as test set')
    parser.add_argument('--sim_top_k', type=int, default=5,
                        help='Number of nearest neighbours for similarity '
                             'graphs (miRNA/lncRNA/drug)')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of training positives used for validation')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation (1 = single run)')

    parser.add_argument('--n_hid', type=int, default=64,
                        help='Hidden dimension for node embeddings')
    parser.add_argument('--n_steps', type=int, default=3,
                        help='Number of state transitions per meta-graph cell')
    parser.add_argument('--n_cells', type=int, default=2,
                        help='Number of meta-graph cells (multi-scale)')
    parser.add_argument('--attn_dim', type=int, default=32,
                        help='Causal attention internal dimension')
    parser.add_argument('--edge_emb_dim', type=int, default=32,
                        help='Edge type embedding dimension for causal attention')
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--timesteps', type=int, default=1000,
                        help='DDPM diffusion timesteps T')
    parser.add_argument('--beta_min', type=float, default=1e-4)
    parser.add_argument('--beta_max', type=float, default=2e-2)
    parser.add_argument('--attn_diff_weight', type=float, default=1,
                        help='Lambda: how strongly attention guides diffusion timestep. '
                             '0 = random timestep.')

    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--arch_lr', type=float, default=3e-4,
                        help='Learning rate for meta-graph architecture params (alphas)')
    parser.add_argument('--g_steps', type=int, default=5,
                        help='Generator updates per epoch')
    parser.add_argument('--d_steps', type=int, default=2,
                        help='Discriminator updates per epoch')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight of reconstruction (MSE) loss in generator')
    parser.add_argument('--bce_weight', type=float, default=1,
                        help='Weight of auxiliary BCE loss on full predictions')
    parser.add_argument('--cost_fn', type=float, default=1.0,
                        help='Cost for false negatives in discriminator (cFN)')
    parser.add_argument('--cost_fp', type=float, default=1.0,
                        help='Cost for false positives in discriminator (cFP)')
    parser.add_argument('--use_gp', action='store_true', default=False,
                        help='Use WGAN-GP (gradient penalty) instead of weight clipping')
    parser.add_argument('--gp_lambda', type=float, default=10.0,
                        help='Gradient penalty coefficient for WGAN-GP')
    parser.add_argument('--weight_clip', type=float, default=0.01,
                        help='Weight clipping bound (only used if --use_gp not set)')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping max norm for generator')
    parser.add_argument('--lr_decay', type=float, default=0.0,
                        help='Learning rate decay per epoch (0 = no decay)')

    parser.add_argument('--pro_pm', type=int, default=50,
                        help='Num false-positive-mask negatives per miRNA row')
    parser.add_argument('--pro_zr', type=int, default=50,
                        help='Num zero-row negatives per miRNA row')

    parser.add_argument('--eps_start', type=float, default=0.8,
                        help='Initial epsilon for epsilon-greedy path sampling')
    parser.add_argument('--eps_decay', type=float, default=0.95,
                        help='Epsilon decay per epoch')
    parser.add_argument('--arch_warmup', type=int, default=30,
                        help='Epochs to use fixed argmax architecture before search')

    parser.add_argument('--save_dir', type=str, default='./weights')
    parser.add_argument('--print_every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    return parser.parse_args()
