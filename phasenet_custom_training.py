import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from seisbench.models import VariableLengthPhaseNet

# --- Constants ---
WINDOW_SIZE = 50000
GAUSS_STD = 20
SAMPLING_RATE = 100
CHECKPOINT_PATH = "vlphasenet_checkpoint.pt"
LOG_CSV = "training_log.csv"
PHASE_LABELS = {"noise": 0, "p": 1, "s": 2}

# --- Gaussian Mask ---
def apply_gaussian_mask(length, index, std=GAUSS_STD):
    x = np.arange(length)
    return np.exp(-0.5 * ((x - index) / std) ** 2)

# --- Dataset ---
class GaussianLabelDataset(Dataset):
    def __init__(self, root_dir, label_csv):
        self.label_df = pd.read_csv(label_csv)
        self.mseed_paths = self._find_mseed_files(root_dir)
        self.trace_map = self._index_traces()

    def _find_mseed_files(self, base_dir):
        return [
            os.path.join(root, file)
            for root, _, files in os.walk(base_dir)
            for file in files if file.endswith(".mseed")
        ]

    def _index_traces(self):
        index = []
        for path in self.mseed_paths:
            stream = read(path)
            for trace_index in range(len(stream)):
                name = self._build_trace_name(path, trace_index)
                if name in set(self.label_df["Name"]):
                    index.append((path, trace_index, name))
        return index

    def _build_trace_name(self, mseed_path, trace_index):
        parts = mseed_path.split(os.sep)
        exp = [p for p in parts if p.startswith("Exp_")][0].replace("Exp_", "")
        run = [p for p in parts if p.startswith("Run")][0]
        event = os.path.basename(mseed_path).split("_WindowSize")[0]
        return f"p_picks_Exp_{exp}_{run}_{event}_trace{trace_index + 1}"

    def __len__(self):
        return len(self.trace_map)

    def __getitem__(self, idx):
        path, trace_index, name_key = self.trace_map[idx]
        picks = self.label_df[self.label_df["Name"] == name_key]
        pick = int(picks.iloc[0]["marked_point"])

        stream = read(path)
        tr = stream[trace_index]
        waveform = tr.data.astype(np.float32)
        full_len = len(waveform)

        start = max(0, pick - WINDOW_SIZE // 2)
        end = start + WINDOW_SIZE
        if end > full_len:
            start = full_len - WINDOW_SIZE
            end = full_len

        waveform = waveform[start:end]
        waveform = torch.tensor(waveform).unsqueeze(0)  # shape: (1, 50000)

        label_mask = np.zeros((3, WINDOW_SIZE), dtype=np.float32)

        local_idx = pick - start
        if 0 <= local_idx < WINDOW_SIZE:
            label_mask[PHASE_LABELS["p"]] = apply_gaussian_mask(WINDOW_SIZE, local_idx)

        # Noise = 1 - P - S
        label_mask[PHASE_LABELS["noise"]] = np.clip(1 - label_mask[1] - label_mask[2], 0, 1)

        label = torch.tensor(label_mask)
        return waveform, label

# --- Training ---
def train_model(data_dir, label_csv, epochs=5, batch_size=20):
    dataset = GaussianLabelDataset(data_dir, label_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VariableLengthPhaseNet(
        in_channels=1,
        classes=3,
        phases="NPS",
        sampling_rate=SAMPLING_RATE,
        norm="std"  # still enabled per model design, even if data is raw
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print("ℹ️  No checkpoint found, training from scratch.")

    log = []

    for epoch in range(epochs):
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
        data_dir=r"c:\Users\hiriy\Downloads\python\RTX2025\Data",
        label_csv="p_picks_Data.csv",
        epochs=1,
        batch_size=1
    )
