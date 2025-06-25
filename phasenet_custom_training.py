import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from seisbench.models import VariableLengthPhaseNet
from torch.optim.lr_scheduler import ReduceLROnPlateau


# --- Constants ---
GAUSS_STD = 20
PHASE_LABELS = {"p": 1, "s": 2, "noise": 0}
SAMPLING_RATE = 5000000
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

class PNoiseDataset(Dataset):
    def __init__(self, root_dir, label_csv):
        print(f"🔍 Loading picks from: {label_csv}")
        self.label_df = pd.read_csv(label_csv)  # ✅ define it before using
        print(f"📑 Loaded {len(self.label_df)} pick entries from {label_csv}")
        
        self.root_dir = root_dir
        self.mseed_paths = find_mseed_files(root_dir)

        # Only keep rows where pick is valid
        self.pick_rows = self.label_df[self.label_df["marked_point"] != -1].reset_index(drop=True)
        print(f"✅ Using {len(self.pick_rows)} individual picks for training.")


        # Cache all traces into memory
        self.traces = []  # List of (name_key, waveform, path)
        for path in self.mseed_paths:
            stream = read(path)
            for i, tr in enumerate(stream):
                name_key = build_trace_name(path, i)
                waveform = tr.data.astype(np.float32)
                self.traces.append((name_key, waveform, path))

        print(f"✅ Cached {len(self.traces)} traces into memory.")

    def __len__(self):
        return len(self.pick_rows)


    def __getitem__(self, idx):
        import re

        row = self.pick_rows.iloc[idx]
        name_key = row["Name"]
        pick_idx = int(row["marked_point"])

        # --- Parse name_key using regex ---
        match = re.match(r"p_picks_Exp_(.+?)_(Run.+)_(EventID_\d+)_trace(\d+)", name_key)
        if not match:
            raise ValueError(f"❌ Invalid Name format: {name_key}")

        exp = match.group(1)
        run = match.group(2)
        event = match.group(3)
        trace_index = int(match.group(4)) - 1  # 1-based to 0-based

        # --- Get corresponding waveform from cached self.traces ---
        try:
            matching_trace = next(t for t in self.traces if t[0] == name_key)
        except StopIteration:
            # Pick exists but corresponding waveform is missing
            return self.__getitem__((idx + 1) % len(self))

        full_waveform = matching_trace[1]
        full_length = len(full_waveform)

        # --- Crop window around pick ---
        window_size = 50000
        half_window = window_size // 2
        min_start = max(0, pick_idx - half_window)
        max_start = min(full_length - window_size, pick_idx - int(0.2 * window_size))
        start_idx = np.random.randint(min_start, max_start + 1) if max_start > min_start else min_start
        end_idx = start_idx + window_size

        #print(f"\n📄 {name_key} | Pick: {pick_idx} | Crop: {start_idx}–{end_idx}")

        waveform = full_waveform[start_idx:end_idx]
        label_mask = np.zeros((3, window_size), dtype=np.float32)

        if pick_idx == -1:
            label_mask[0] = 1.0  # Entire window is noise
        else:
            local_idx = pick_idx - start_idx
            if 0 <= local_idx < window_size:
                mask = apply_gaussian_mask(window_size, local_idx)
                label_mask[1] = mask
            label_mask[0] = np.clip(1 - label_mask[1] - label_mask[2], 0, 1)

        waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]
        label = torch.tensor(label_mask)                # [3, T]
        return waveform, label




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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

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

        avg_loss = running_loss / len(loader)
        print(f"✅ Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

        # Step the scheduler AFTER each epoch using average loss
        scheduler.step(avg_loss)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, CHECKPOINT_PATH)
        print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")


# --- Entry Point ---
if __name__ == "__main__":
    train_model(
        data_dir=r"f:\Data",
        label_csv="p_picks_Data.csv",
        epochs=200,
        batch_size=200
    )
