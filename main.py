import os
import warnings
import torch

from config import get_config
from train import train
from utils import set_seed

warnings.filterwarnings('ignore')


def main():
    cfg = get_config()
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('[Info] GPU available:', torch.cuda.get_device_name(0))
    else:
        print('[Info] Using CPU (training will be slow)')

    print('\n[1/2] Setup complete (data loaded per fold)')
    print(f'      Data dir: {os.path.abspath(cfg.data_dir)}')
    print(f'\n[2/2] Training with {cfg.n_folds}-fold CV + ensemble...')

    mean_auc, final_metrics = train(cfg, None)

    print('\n[Done]')
    print(f'  Best model saved to: {os.path.abspath(cfg.save_dir)}')


if __name__ == '__main__':
    main()
