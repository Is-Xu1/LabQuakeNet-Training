import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from seisbench.models import VariableLengthPhaseNet

# --- Constants ---
SAMPLING_RATE = 5000000
PHASE_LABELS = {"noise": 0, "p": 1}
torch.manual_seed(42)
np.random.seed(42)

def apply_gaussian_mask(length, index, std):
    x = np.arange(length)
    return np.exp(-0.5 * ((x - index) / std) ** 2)

class SlidingWindowDataset(Dataset):
    def __init__(self, root_dir, label_csv, window_size=50000, stride=20000, gauss_std=200,
                 amp_scale_range=(0.8, 1.2)):
        self.window_size = window_size
        self.stride = stride
        self.gauss_std = gauss_std
        self.amp_scale_range = amp_scale_range
        self.data = []
        self.waveform_cache = {}

        print(f"🔍 Loading picks from: {label_csv}")
        label_df = pd.read_csv(label_csv)
        print(f"📑 Loaded {len(label_df)} pick entries from {label_csv}")
        self.label_dict = label_df.set_index("Name")["marked_point"].to_dict()

        self._cache_and_window_traces(root_dir)
        print(f"✅ Cached {len(self.waveform_cache)} traces into memory.")

        if len(self.data) == 0:
            raise ValueError("❌ No valid windows generated. Check data folder and labels.")

        pick_windows = sum(pick_idx >= 0 for _, _, pick_idx in self.data)
        noise_windows = len(self.data) - pick_windows

        print(f"🪟 Generated {len(self.data)} sliding windows.")
        print(f"🔵 Pick windows: {pick_windows} ({pick_windows / len(self.data) * 100:.2f}%)")
        print(f"⚪ Noise windows: {noise_windows} ({noise_windows / len(self.data) * 100:.2f}%)")

    def _cache_and_window_traces(self, root_dir):
        mseed_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(root_dir)
            for f in files if f.endswith(".mseed")
        ]
        discarded = 0
        for path in mseed_paths:
            try:
                stream = read(path)
                for i, tr in enumerate(stream):
                    name = self._build_name(path, i)
                    waveform = tr.data.astype(np.float32)
                    if len(waveform) < self.window_size:
                        pad_len = self.window_size - len(waveform)
                        waveform = np.concatenate([waveform, np.zeros(pad_len, dtype=np.float32)])
                    self.waveform_cache[name] = waveform
                    pick_idx = self.label_dict.get(name, -1)  # -1 if no pick
                    discarded += self._generate_windows(name, waveform, pick_idx)
            except Exception as e:
                print(f"⚠️ Skipped {path}: {e}")
        print(f"❌ Discarded {discarded} windows with edge picks.")

    def _build_name(self, path, trace_index):
        parts = path.split(os.sep)
        exp = next(p for p in parts if p.startswith("Exp_")).replace("Exp_", "")
        run = next((p for p in parts if p.startswith("Run")), "RunX")
        event = os.path.basename(path).split("_WindowSize")[0]
        return f"p_picks_Exp_{exp}_{run}_{event}_trace{trace_index + 1}"

    def _generate_windows(self, name, waveform, pick_idx):
        L = len(waveform)
        discarded = 0
        for start in range(0, L - self.window_size + 1, self.stride):
            # Check if pick is inside this window
            if start <= pick_idx < start + self.window_size:
                if start + 2000 <= pick_idx < start + self.window_size - 2000:
                    self.data.append((name, start, pick_idx))  # good pick window
                else:
                    discarded += 1  # pick is near edge → discard
            elif pick_idx == -1:
                self.data.append((name, start, -1))  # no pick at all → noise window
            else:
                self.data.append((name, start, -1))  # pick exists, but outside this window → noise window
        return discarded


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, start, pick_idx = self.data[idx]
        waveform = self.waveform_cache[name]
        segment = waveform[start:start + self.window_size]
        if len(segment) < self.window_size:
            pad_len = self.window_size - len(segment)
            segment = np.concatenate([segment, np.zeros(pad_len, dtype=np.float32)])
        window = segment.copy()

        # Amplitude scaling
        scale = np.random.uniform(*self.amp_scale_range)
        window *= scale

        # Normalize
        window = (window - window.mean()) / (window.std() + 1e-6)

        # Label creation
        label = np.zeros((2, self.window_size), dtype=np.float32)
        if pick_idx >= 0 and start + 5000 <= pick_idx < start + self.window_size - 5000:
            local_idx = pick_idx - start
            label[1] = apply_gaussian_mask(self.window_size, local_idx, self.gauss_std)
            label[0] = np.clip(1 - label[1], 0, 1)
        else:
            label[0] = np.ones(self.window_size, dtype=np.float32)
            label[1] = np.zeros(self.window_size, dtype=np.float32)

        return torch.tensor(window).unsqueeze(0), torch.tensor(label)

class PhaseNetLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, weights=None):
        super().__init__()
        self.eps = eps
        self.weights = weights or torch.tensor([0.1, 1.0])

    def forward(self, prediction, target):
        weights = self.weights.to(prediction.device)
        loss = 0.0
        for c in range(prediction.shape[1]):
            pred = prediction[:, c].clamp(self.eps, 1 - self.eps)
            tgt = target[:, c]
            loss += weights[c] * torch.nn.functional.binary_cross_entropy(pred, tgt, reduction='mean')
        return loss

def train_model(data_dir, label_csv, checkpoint_path, log_csv, window_size, gauss_std, epochs=5, batch_size=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    dataset = SlidingWindowDataset(
        root_dir=data_dir,
        label_csv=label_csv,
        window_size=window_size,
        stride=20000,
        gauss_std=gauss_std,
        amp_scale_range=(0.8, 1.2)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VariableLengthPhaseNet(
        in_channels=1,
        classes=2,
        phases="NP",
        sampling_rate=SAMPLING_RATE,
        norm="std"
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = PhaseNetLoss().to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print("ℹ️ No checkpoint found, training from scratch.")

    if os.path.exists(log_csv):
        log_df = pd.read_csv(log_csv)
        log = log_df.to_dict("records")
        start_epoch = log_df["Epoch"].max() + 1
    else:
        log = []
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"[{checkpoint_path}] Epoch {epoch} Step {i+1}/{len(loader)} → Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"✅ [{checkpoint_path}] Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, checkpoint_path)
        print(f"💾 Saved checkpoint to {checkpoint_path}")

        log.append({"Epoch": epoch, "Avg Loss": avg_loss})
        pd.DataFrame(log).to_csv(log_csv, index=False)

if __name__ == "__main__":
    WINDOWSIZE = [40000,30000]
    GAUSS_STD = [100]
    configs = []
    for w, g in zip(WINDOWSIZE, GAUSS_STD):
        config = {
            "checkpoint": f"{w}w{g}gAN.pt",
            "log": f"log_{w}w{g}gAN.csv",
            "labels": "p_picks_training.csv",
            "window_size": w,
            "gauss_std": g
        }
        configs.append(config)

    for config in configs:
        print(f"\n🚀 Starting training for {config['checkpoint']}...\n")
        train_model(
            data_dir=r"f:\Data",
            label_csv=config["labels"],
            checkpoint_path=config["checkpoint"],
            log_csv=config["log"],
            window_size=config["window_size"],
            gauss_std=config["gauss_std"],
            epochs=60,
            batch_size=200
        )
