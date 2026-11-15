# CADiffGAN

Code for the manuscript *"CADiffGAN: Causal Attention-Guided Diffusion WGAN with a Meta-Graph for Drug–miRNA Association Prediction"*, currently under review.

## Availability

- **License:** MIT (see [LICENSE](LICENSE)).
- **Contact:** siya.xie@mpu.edu.mo

## Installation

Requires Python 3.10 (NumPy < 2).

```bash
git clone https://github.com/xiesiya/CADiffGAN.git
cd CADiffGAN
pip install -r requirements.txt
```

Download the dataset from <https://github.com/xiesiya/mixDR> and place its contents under `./dataset/`.

## Quick Start

```bash
# Train with the default configuration (uses ./dataset, saves to ./weights)
python main.py

# Use a custom dataset directory
python main.py --data_dir /path/to/dataset
```

## Citation

> Manuscript under review; the formal citation will be added after acceptance.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
