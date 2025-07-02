import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from obspy import read
import os
import re
import pandas as pd

# --- Constants ---
WINDOW_SIZE = 50000
GAUSS_STD = 200
AMP_SCALE_RANGE = (0.8, 1.2)
NOISE_STD = 0.01
PICK_CSV = "p_picks_Data.csv"

# --- Gaussian Mask ---
def apply_gaussian_mask(length, index, std=GAUSS_STD):
    x = np.arange(length)
    return np.exp(-0.5 * ((x - index) / std) ** 2)

# --- Trace Name Builder ---
def build_trace_name(path, trace_index):
    parts = re.split(r"[\\/]", path)  # handles / and \

    exp_match = [p for p in parts if p.startswith("Exp_")]
    exp = exp_match[0].replace("Exp_", "") if exp_match else "UnknownExp"

    run_match = [p for p in parts if p.startswith("Run")]
    run = run_match[0] if run_match else "UnknownRun"

    filename = os.path.basename(path)
    event = filename.split("_WindowSize")[0]

    trace_name = f"p_picks_Exp_{exp}_{run}_{event}_trace{trace_index + 1}"
    print(f"✅ Built trace name: {trace_name}")
    return trace_name

# --- Load and Visualize ---
def visualize_processing():
    file_path = filedialog.askopenfilename(filetypes=[("MiniSEED Files", "*.mseed")])
    if not file_path:
        return

    # Load picks
    try:
        label_df = pd.read_csv(PICK_CSV)
        label_dict = label_df.set_index("Name")["marked_point"].to_dict()
    except Exception as e:
        print(f"❌ Failed to load pick CSV: {e}")
        return

    stream = read(file_path)
    tr = stream[0]
    waveform = tr.data.astype(np.float32)

    name = build_trace_name(file_path, 0)
    pick_idx = label_dict.get(name, -1)

    # Augment waveform
    scale = np.random.uniform(*AMP_SCALE_RANGE)
    noise = np.random.normal(0, NOISE_STD, size=waveform.shape).astype(np.float32)
    augmented = waveform * scale + noise

    # Create label mask
    label = np.zeros((2, len(waveform)), dtype=np.float32)
    if 0 <= pick_idx < len(waveform):
        label[1] = apply_gaussian_mask(len(waveform), pick_idx, GAUSS_STD)
        label[0] = np.clip(1 - label[1], 0, 1)
    else:
        label[0] = np.ones(len(waveform), dtype=np.float32)

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(waveform, color='black')
    axes[0].set_title("Original Waveform")
    if pick_idx >= 0:
        axes[0].axvline(pick_idx, color='r', linestyle='--', label='P Pick')
        axes[0].legend()

    axes[1].plot(augmented, color='blue')
    axes[1].set_title("Augmented Waveform")

    axes[2].plot(label[1], label="P Prob", color='red')
    axes[2].plot(label[0], label="Noise Prob", color='gray')
    axes[2].set_title("Label Probabilities")
    axes[2].legend()

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()

    # Clear old widgets
    for widget in frame.winfo_children():
        widget.destroy()

    # Embed plot
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Add navigation toolbar
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    toolbar.pack()

# --- Main Window ---
root = tk.Tk()
root.title("Waveform Preprocessing Visualizer")
root.geometry("1200x900")

# Exit cleanly on window close
def on_closing():
    print("🛑 Window closed. Exiting...")
    root.destroy()
    exit()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Layout
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

btn = tk.Button(root, text="📂 Load .mseed File", command=visualize_processing)
btn.pack(pady=5)

# Launch GUI
root.mainloop()
