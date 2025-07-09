import os
import numpy as np
import pandas as pd
import torch
from obspy import read
from seisbench.models import VariableLengthPhaseNet
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Config ---
WINDOW_SIZE = 50000
STRIDE = 20000
SAMPLING_RATE = 5000000
THRESHOLD = 0.5
TOLERANCE = 10
CHECKPOINT_PATH = "checkpoint_50000w100g.pt"
DATA_DIR = "f:/Data"
VALIDATION_CSV = "p_picks_validation.csv"
OUTPUT_CSV = "validation_results.csv"

# --- CUDA support ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📟 Using device: {DEVICE}")

# --- Load model ---
model = VariableLengthPhaseNet(
    in_channels=1,
    classes=2,
    phases="NP",
    sampling_rate=SAMPLING_RATE,
    norm="std"
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()

# --- Helper functions ---
def parse_name(name):
    parts = name.split("_")
    trace_index = int(parts[-1].replace("trace", "")) - 1
    event = f"{parts[-3]}_{parts[-2]}"
    exp = f"{parts[2]}_{parts[3]}"
    run_parts = parts[4:-3]
    run = "_".join(run_parts)
    return exp, run, event, trace_index

def load_waveform(name_key):
    exp, run, event, trace_index = parse_name(name_key)
    base_folder = os.path.join(DATA_DIR, exp)
    folder = os.path.join(base_folder, run)
    if not os.path.exists(folder):
        alt_run = run.replace("_Traces", "") if "_Traces" in run else run + "_Traces"
        folder = os.path.join(base_folder, alt_run)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"❌ Neither '{run}' nor fallback '{alt_run}' exists under {base_folder}")
    filename = f"{event}_WindowSize_0.05s_Data.mseed"
    full_path = os.path.join(folder, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ File not found: {full_path}")
    stream = read(full_path)
    return stream[trace_index].data.astype(np.float32), full_path

# --- Inference and CSV generation ---
df = pd.read_csv(VALIDATION_CSV)
results = []
prediction_cache = {}

for _, row in df.iterrows():
    name = row["Name"]
    pick_idx = int(row["marked_point"])
    try:
        waveform, _ = load_waveform(name)
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        continue

    probs = np.zeros((2, len(waveform)))
    count = np.zeros(len(waveform))

    for start in range(0, len(waveform) - WINDOW_SIZE + 1, STRIDE):
        segment = waveform[start:start + WINDOW_SIZE]
        input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)[0].detach().cpu().numpy()
        probs[:, start:start + WINDOW_SIZE] += output
        count[start:start + WINDOW_SIZE] += 1

    last_start = len(waveform) - WINDOW_SIZE
    if last_start % STRIDE != 0:
        remaining = waveform[last_start:]
        padded = np.zeros(WINDOW_SIZE, dtype=np.float32)
        padded[:len(remaining)] = remaining
        input_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)[0].detach().cpu().numpy()
        probs[:, last_start:] += output[:, :len(remaining)]
        count[last_start:] += 1

    count[count == 0] = 1
    probs /= count

    probs_p = gaussian_filter1d(probs[1], sigma=15)
    probs_noise = gaussian_filter1d(probs[0], sigma=15)

    peak_indices, _ = find_peaks(probs_p, height=THRESHOLD, prominence=0.05, distance=1000)

    matched = False
    tp, fp = 0, 0
    for p in peak_indices:
        if abs(p - pick_idx) <= TOLERANCE and not matched:
            tp += 1
            matched = True
        else:
            fp += 1
    fn = 1 if not matched else 0

    results.append({
        "Name": name,
        "True Pick": pick_idx,
        "Predicted Picks": peak_indices.tolist(),
        "Correct": bool(tp),
        "True Positive": tp,
        "False Positive": fp,
        "False Negative": fn
    })

    prediction_cache[name] = (waveform, probs_p, probs_noise)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

tp_total = results_df["True Positive"].sum()
fp_total = results_df["False Positive"].sum()
fn_total = results_df["False Negative"].sum()

print(f"✅ Saved results to {OUTPUT_CSV}")
print(f"📊 True Positives: {tp_total}")
print(f"📊 False Positives: {fp_total}")
print(f"📊 False Negatives: {fn_total}")


# --- Tkinter GUI with Toolbar and Probabilities ---
class Visualizer:
    def __init__(self, master, results_df, cache):
        self.master = master
        self.master.title("Validation Viewer")
        self.df = results_df
        self.cache = cache
        self.index = 0
        self.total = len(self.df)

        self.frame = tk.Frame(master)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(master)
        btn_frame.pack()
        tk.Button(btn_frame, text="Previous", command=self.prev).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Next", command=self.next).pack(side=tk.LEFT)

        self.plot()

    def plot(self):
        self.ax1.clear()
        self.ax2.clear()

        row = self.df.iloc[self.index]
        name = row["Name"]
        true_pick = row["True Pick"]
        try:
            waveform, probs_p, probs_noise = self.cache[name]
            self.ax1.plot(waveform, color="black", label="Waveform")
            self.ax1.axvline(x=true_pick, color="green", label="True Pick")
            preds = eval(row["Predicted Picks"]) if isinstance(row["Predicted Picks"], str) else row["Predicted Picks"]
            for p in preds:
                self.ax1.axvline(x=p, color="red", linestyle="--", alpha=0.7)
            self.ax1.set_title(f"{name} ({self.index + 1}/{self.total})")
            self.ax1.legend()

            self.ax2.plot(probs_p, label="P Probability", color="blue", alpha=0.8)
            self.ax2.plot(probs_noise, label="Noise Probability", color="gray", alpha=0.8)
            self.ax2.set_title("Model Output Probabilities")
            self.ax2.legend()

            self.canvas.draw()
        except Exception as e:
            self.ax1.set_title(f"⚠️ {e}")
            self.canvas.draw()

    def next(self):
        self.index = (self.index + 1) % self.total
        self.plot()

    def prev(self):
        self.index = (self.index - 1) % self.total
        self.plot()

# --- Exit properly on window close ---
def on_close():
    print("👋 Exiting...")
    root.destroy()

root = tk.Tk()
root.geometry("1400x700")
app = Visualizer(root, results_df, prediction_cache)
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()

