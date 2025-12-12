import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .dataset import PennFudanYOLODataset
from .model import TinyYOLOv1
from .loss import YOLOv1Loss

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    train_ds = PennFudanYOLODataset(cfg.data_root, cfg.S, cfg.C, cfg.img_size, train=True)
    val_ds   = PennFudanYOLODataset(cfg.data_root, cfg.S, cfg.C, cfg.img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyYOLOv1(S=cfg.S, C=cfg.C).to(device)
    crit = YOLOv1Loss(cfg.lambda_coord, cfg.lambda_noobj).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / cfg.checkpoint_name

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]")
        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            pred = model(imgs)  # (B,S,S,5+C)
            loss = crit(pred, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss = total / max(1, len(train_loader))

        # quick val
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                tgt = batch["target"].to(device, non_blocking=True)
                pred = model(imgs)
                vtotal += crit(pred, tgt).item()
        val_loss = vtotal / max(1, len(val_loader))

        print(f"[INFO] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )
            print("[SAVE] best checkpoint ->", ckpt_path)

    print("[DONE] best_val_loss =", best_val)

if __name__ == "__main__":
    main()
