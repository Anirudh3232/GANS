# Basic GAN (PyTorch)
A minimal, beginner‑friendly GAN on MNIST/FashionMNIST. The goal is clarity over cleverness.

## Why this repo?
- **Short & readable**: ~200 lines of core training code.
- **Sane defaults**: Works out of the box on CPU or a single GPU.
- **Learn-by-doing**: Each function is commented with *why* not just *what*.


## Quickstart
```bash
# 1) Setup
python -m venv .venv && source .venv/bin/activate # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt


# 2) Train on MNIST (28x28 digits)
python -m src.train_gan --dataset mnist --epochs 10 --batch-size 128


# 3) See results
ls samples/ # generated images appear every epoch