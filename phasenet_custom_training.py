import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from seisbench.models import VariableLengthPhaseNet

# --- Constants ---
WINDOW_SIZE = 50000
GAUSS_STD = 200
SAMPLING_RATE = 5000000
CHECKPOINT_PATH = "vlphasenet_checkpoint.pt"
LOG_CSV = "training_log.csv"
PHASE_LABELS = {"noise": 0, "p": 1}

# --- Gaussian Mask ---
def apply_gaussian_mask(length, index, std=GAUSS_STD):
    x = np.arange(length)
    return np.exp(-0.5 * ((x - index) / std) ** 2)

class SlidingWindowDataset(Dataset):
    def __init__(self, root_dir, label_csv, window_size=50000, stride=20000, gauss_std=200,
                 amp_scale_range=(0.8, 1.2), noise_std=0.05):
        self.window_size = window_size
        self.stride = stride
        self.gauss_std = gauss_std
        self.amp_scale_range = amp_scale_range
        self.noise_std = noise_std
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

        # Count pick vs noise windows
        pick_windows = sum(
            0 <= pick_idx and start + 2000 <= pick_idx < start + self.window_size - 2000
            for _, start, pick_idx in self.data
        )
        noise_windows = len(self.data) - pick_windows
        pct_pick = 100 * pick_windows / len(self.data)
        pct_noise = 100 * noise_windows / len(self.data)

        print(f"🪟 Generated {len(self.data)} sliding windows.")
        print(f"🔵 Pick windows: {pick_windows} ({pct_pick:.2f}%)")
        print(f"⚪ Noise windows: {noise_windows} ({pct_noise:.2f}%)")

        # Optional: count unique traces with picks
        unique_picks = set(name for name, _, pick_idx in self.data if pick_idx >= 0)
        print(f"📍 Unique traces with picks: {len(unique_picks)}")

    def _cache_and_window_traces(self, root_dir):
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".mseed"):
                    path = os.path.join(root, f)
                    try:
                        stream = read(path)
                        for i, tr in enumerate(stream):
                            name = self._build_name(path, i)
                            waveform = tr.data.astype(np.float32)
                            self.waveform_cache[name] = waveform
                            pick_idx = self.label_dict.get(name, -1)
                            self._generate_windows(name, waveform, pick_idx)
                    except Exception as e:
                        print(f"⚠️ Skipped {path}: {e}")

    def _build_name(self, path, trace_index):
        parts = path.split(os.sep)
        exp = next(p for p in parts if p.startswith("Exp_")).replace("Exp_", "")
        run = next((p for p in parts if p.startswith("Run")), "RunX")
        event = os.path.basename(path).split("_WindowSize")[0]
        return f"p_picks_Exp_{exp}_{run}_{event}_trace{trace_index + 1}"

    def _generate_windows(self, name, waveform, pick_idx):
        L = len(waveform)
        for start in range(0, L - self.window_size + 1, self.stride):
            if 0 <= pick_idx < L:
                # Only include if pick is not too close to edges
                if start + 2000 <= pick_idx < start + self.window_size - 2000:
                    self.data.append((name, start, pick_idx))
            else:
                # No pick in trace: safe to add as noise window
                self.data.append((name, start, pick_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, start, pick_idx = self.data[idx]
        waveform = self.waveform_cache[name]
        window = waveform[start:start + self.window_size].copy()

        # --- Amplitude scaling ---
        scale = np.random.uniform(*self.amp_scale_range)
        window *= scale

        # --- Add Gaussian noise ---
        noise = np.random.normal(0, self.noise_std, size=window.shape)
        window += noise.astype(np.float32)

        label = np.zeros((2, self.window_size), dtype=np.float32)

        if 0 <= pick_idx < len(waveform) and start + 2000 <= pick_idx < start + self.window_size - 2000:
            local_idx = pick_idx - start
            label[1] = apply_gaussian_mask(self.window_size, local_idx, self.gauss_std)
            label[0] = np.clip(1 - label[1], 0, 1)
        else:
            # Forcefully assign full noise label
            label[0] = np.ones(self.window_size, dtype=np.float32)
            label[1] = np.zeros(self.window_size, dtype=np.float32)

        waveform_tensor = torch.tensor(window).unsqueeze(0)
        label_tensor = torch.tensor(label)
        return waveform_tensor, label_tensor



# --- Loss ---
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

# --- Training ---
def train_model(data_dir, label_csv, epochs=5, batch_size=20):
    dataset = SlidingWindowDataset(
        root_dir=data_dir,
        label_csv=label_csv,
        window_size=WINDOW_SIZE,
        stride=30000,
        gauss_std=GAUSS_STD,
        amp_scale_range=(0.8, 1.2),
        noise_std=0.02
    )

    if len(dataset) == 0:
        raise ValueError("❌ Dataset is empty. Check if files are missing or naming mismatches.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VariableLengthPhaseNet(
        in_channels=1,
        classes=2,
        phases="NP",
        sampling_rate=SAMPLING_RATE,
        norm="std"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = PhaseNetLoss()

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print("ℹ️ No checkpoint found, training from scratch.")

    if os.path.exists(LOG_CSV):
        log_df = pd.read_csv(LOG_CSV)
        log = log_df.to_dict("records")
        start_epoch = log_df["Epoch"].max() + 1
        print(f"📈 Continuing log from epoch {start_epoch}")
    else:
        log = []
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"[Epoch {epoch}] Step {i+1}/{len(loader)} → Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, CHECKPOINT_PATH)
        print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")

        log.append({"Epoch": epoch, "Avg Loss": avg_loss})
        pd.DataFrame(log).to_csv(LOG_CSV, index=False)

# --- Entry Point ---
if __name__ == "__main__":
    train_model(
        data_dir=r"f:\Data",
        label_csv="p_picks_Data.csv",
        epochs=20,
        batch_size=200
    )