import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from seisbench.models import VariableLengthPhaseNet

# --- Constants ---
GAUSS_STD = 20
PHASE_LABELS = {"p": 1, "s": 2, "noise": 0}
SAMPLING_RATE = 100
CHECKPOINT_PATH = "vlphasenet_checkpoint.pt"

# --- Gaussian Mask ---
def apply_gaussian_mask(length, index, std=GAUSS_STD):
    x = np.arange(length)
    return np.clip(np.exp(-0.5 * ((x - index) / std) ** 2), 0, 1)

# --- Find all .mseed files ---
def find_mseed_files(base_dir):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(base_dir)
        for file in files if file.endswith(".mseed")
    ]

# --- Label name from path and trace index ---
def build_trace_name(mseed_path, trace_index):
    parts = mseed_path.split(os.sep)
    exp = [p for p in parts if p.startswith("Exp_")][0].replace("Exp_", "")
    run = [p for p in parts if p.startswith("Run")][0]
    event = os.path.basename(mseed_path).split("_WindowSize")[0]
    return f"p_picks_Exp_{exp}_{run}_{event}_trace{trace_index + 1}"

# --- Dataset Class ---
class PNoiseDataset(Dataset):
    def __init__(self, root_dir, label_csv):
        print(f"🔍 Loading dataset from: {root_dir}")
        self.label_df = pd.read_csv(label_csv)
        print(f"📑 Loaded {len(self.label_df)} pick entries from {label_csv}")
        self.mseed_paths = find_mseed_files(root_dir)
        print(f"📁 Found {len(self.mseed_paths)} .mseed files")

    def __len__(self):
        return sum(len(read(path)) for path in self.mseed_paths)

    def __getitem__(self, global_idx):
        trace_counter = 0
        for path in self.mseed_paths:
            stream = read(path)
            if global_idx < trace_counter + len(stream):
                trace_index = global_idx - trace_counter
                tr = stream[trace_index]
                full_waveform = tr.data.astype(np.float32)
                full_length = len(full_waveform)

                print(f"\n📄 File: {os.path.basename(path)}, Trace: {trace_index}")
                print(f"   → Trace length: {full_length} samples")

                name_key = build_trace_name(path, trace_index)
                picks = self.label_df[self.label_df["Name"] == name_key]

                if len(picks) == 0:
                    print(f"⚠️  No picks found for {name_key}, using center default.")
                    center = full_length // 2
                    min_pick = max_pick = center
                else:
                    print(f"✅ {len(picks)} picks found for {name_key}")
                    pick_indices = picks["marked_point"].astype(int).values
                    min_pick = np.min(pick_indices)
                    max_pick = np.max(pick_indices)

                window_size = 50000
                min_start = max(0, max_pick - window_size + 1)
                max_start = min(min_pick, full_length - window_size)
                if max_start < min_start:
                    start_idx = min_start
                else:
                    start_idx = np.random.randint(min_start, max_start + 1)

                end_idx = start_idx + window_size
                print(f"🪟 Cropping window: {start_idx} to {end_idx} (size: {window_size})")

                waveform = full_waveform[start_idx:end_idx]
                label_mask = np.zeros((3, window_size), dtype=np.float32)

                for _, row in picks.iterrows():
                    lbl = "p"  # assume all picks are P-wave
                    pick_idx = int(row["marked_point"])
                    if lbl in PHASE_LABELS and 0 <= pick_idx < full_length:
                        if start_idx <= pick_idx < end_idx:
                            local_idx = pick_idx - start_idx
                            mask = apply_gaussian_mask(window_size, local_idx)
                            label_mask[PHASE_LABELS[lbl]] += mask


                label_mask[0] = np.clip(1 - label_mask[1] - label_mask[2], 0, 1)

                waveform = torch.tensor(waveform).unsqueeze(0)
                label = torch.tensor(label_mask)
                return waveform, label

            trace_counter += len(stream)

        raise IndexError("Global index out of range")

# --- Training Function ---
def train_model(data_dir, label_csv, epochs=5, batch_size=2):
    dataset = PNoiseDataset(data_dir, label_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("🧠 Initializing VariableLengthPhaseNet...")
    model = VariableLengthPhaseNet(
        in_channels=1,
        classes=3,
        phases="NPS",
        sampling_rate=SAMPLING_RATE,
        norm="std"
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

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for i, (waveform, label) in enumerate(loader):
            output = model(waveform)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"[Epoch {epoch}] Step {i+1}/{len(loader)} → Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch} complete. Avg Loss: {running_loss / len(loader):.4f}")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, CHECKPOINT_PATH)
        print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")

# --- Entry Point ---
if __name__ == "__main__":
    train_model(
        data_dir=r"c:\Users\hiriy\Downloads\python\RTX2025\Data",
        label_csv="p_picks_Data.csv",
        epochs=1,
        batch_size=20
    )
