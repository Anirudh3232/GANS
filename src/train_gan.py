# train_gan.py
# Minimal GAN trainer with clear logs + forced sample/marker so you SEE outputs.
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# NOTE: make sure you have an empty src/__init__.py so these imports work
from src.models import Generator, Discriminator, Z_DIM
from src.data import get_loader
from src.utils import set_seed, save_sample_grid

torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser(description="Train a minimal GAN on MNIST/FashionMNIST")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"], help="dataset to use")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    p.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 (DCGAN default)")
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-every", type=int, default=1, help="epochs between sample dumps")
    p.add_argument("--samples-dir", default="samples")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # --- sanity prints so it's obvious where files go ---
    print("[INFO] CWD:", Path().cwd())
    print("[INFO] args:", vars(args))

    # Ensure samples dir exists and drop a marker file you can see right away
    Path(args.samples_dir).mkdir(parents=True, exist_ok=True)
    marker = Path(args.samples_dir) / "sanity.txt"
    marker.write_text("hello from GAN\n")
    print("[INFO] Wrote marker:", marker.resolve())

    set_seed(args.seed)

    device = torch.device(args.device)
    loader, _ = get_loader(args.dataset, args.batch_size)
    try:
        print("[INFO] num batches:", len(loader))
    except TypeError:
        pass

    G = Generator().to(device)
    D = Discriminator().to(device)

    # Losses/opts
    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    fixed_noise = torch.randn(64, Z_DIM, device=device)  # consistent snapshots

    # --- force-save a sample immediately so you SEE an image even before training ---
    with torch.no_grad():
        samples0 = G(torch.randn(64, Z_DIM, device=device)).cpu()
    save_sample_grid(samples0, args.samples_dir, "epoch_0000_forced.png", nrow=8)
    print("[INFO] Wrote forced sample:", (Path(args.samples_dir) / "epoch_0000_forced.png").resolve())

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for real, _ in pbar:
            real = real.to(device)
            bs = real.size(0)

            # --- Train Discriminator: maximize log(D(x)) + log(1 - D(G(z))) ---
            opt_D.zero_grad()

            # real -> 1
            logits_real = D(real)
            loss_real = criterion(logits_real, torch.ones(bs, device=device))

            # fake -> 0
            z = torch.randn(bs, Z_DIM, device=device)
            with torch.no_grad():
                fake = G(z)
            logits_fake = D(fake)
            loss_fake = criterion(logits_fake, torch.zeros(bs, device=device))

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # --- Train Generator: maximize log(D(G(z))) ---
            opt_G.zero_grad()
            z = torch.randn(bs, Z_DIM, device=device)
            fake = G(z)
            logits = D(fake)
            loss_G = criterion(logits, torch.ones(bs, device=device))
            loss_G.backward()
            opt_G.step()

            pbar.set_postfix({"loss_D": f"{loss_D.item():.3f}", "loss_G": f"{loss_G.item():.3f}"})

        # Save sample grid each epoch (or as configured)
        print("[INFO] Finished epoch. Saving samples/checkpoints.")
        if epoch % args.sample_every == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
            save_sample_grid(samples, args.samples_dir, f"epoch_{epoch:04d}.png", nrow=8)
            print("[INFO] Wrote:", (Path(args.samples_dir) / f"epoch_{epoch:04d}.png").resolve())

    # Save final weights
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(G.state_dict(), "checkpoints/generator.pt")
    torch.save(D.state_dict(), "checkpoints/discriminator.pt")
    print("[INFO] Saved checkpoints to:", Path("checkpoints").resolve())
    print("[INFO] Done. Samples dir:", Path(args.samples_dir).resolve())


if __name__ == "__main__":
    main()
