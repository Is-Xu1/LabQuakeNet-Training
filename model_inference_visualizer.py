import os
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
from obspy import read
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from seisbench.models import VariableLengthPhaseNet

# --- Constants ---
WINDOW_SIZE = 50000
STRIDE = 20000
SAMPLING_RATE = 5000000
CHECKPOINT_PATH = "checkpoint_50000w75g_training.pt"

# --- Load model ---
model = VariableLengthPhaseNet(
    in_channels=1,
    classes=2,
    phases="NP",
    sampling_rate=SAMPLING_RATE,
    norm="std"
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model"])
model.eval()

# --- GUI class ---
class PhaseNetVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("PhaseNet Inference Viewer")

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.toolbar_frame = tk.Frame(master)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        control_frame = tk.Frame(master)
        control_frame.pack(fill=tk.X)

        self.button = tk.Button(control_frame, text="Load .mseed File", command=self.load_file)
        self.button.pack(side=tk.LEFT, padx=10, pady=5)

        tk.Label(control_frame, text="Trace Index:").pack(side=tk.LEFT)
        self.trace_index_entry = tk.Entry(control_frame, width=5)
        self.trace_index_entry.pack(side=tk.LEFT)
        self.trace_index_entry.insert(0, "0")

        self.status_label = tk.Label(control_frame, text="")
        self.status_label.pack(side=tk.LEFT, padx=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MSEED files", "*.mseed")])
        if not file_path:
            return

        try:
            stream = read(file_path)
            trace_idx = int(self.trace_index_entry.get())
            trace = stream[trace_idx]
            waveform = trace.data.astype(np.float32)
            x = np.arange(len(waveform))

            # --- Slide over waveform and predict ---
            probs_p = np.zeros(len(waveform))
            probs_noise = np.zeros(len(waveform))
            count = np.zeros(len(waveform))

            for start in range(0, len(waveform) - WINDOW_SIZE + 1, STRIDE):
                window = waveform[start:start + WINDOW_SIZE]
                input_tensor = torch.tensor(window).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                with torch.no_grad():
                    output = model(input_tensor).squeeze(0).numpy()  # [2, T]
                probs_noise[start:start + WINDOW_SIZE] += output[0]
                probs_p[start:start + WINDOW_SIZE] += output[1]
                count[start:start + WINDOW_SIZE] += 1

            count[count == 0] = 1  # prevent divide-by-zero
            probs_noise /= count
            probs_p /= count

            # --- Plot ---
            self.ax1.clear()
            self.ax1.plot(x, waveform, label="Waveform")
            self.ax1.set_ylabel("Amplitude")
            self.ax1.legend()

            self.ax2.clear()
            self.ax2.plot(x, probs_noise, label="Noise Prob", alpha=0.6)
            self.ax2.plot(x, probs_p, label="P Prob", alpha=0.8)
            self.ax2.set_ylabel("Probability")
            self.ax2.set_xlabel("Sample Index")
            self.ax2.set_ylim(0, 1.05)
            self.ax2.legend()

            self.fig.suptitle(f"{os.path.basename(file_path)} | Trace {trace_idx}")
            self.canvas.draw()
            self.status_label.config(text="✅ Loaded and visualized.")

        except Exception as e:
            self.status_label.config(text=f"❌ Error: {e}")

# --- Launch GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PhaseNetVisualizer(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()