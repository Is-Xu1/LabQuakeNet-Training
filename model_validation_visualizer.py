import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import torch
from obspy import read
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
from seisbench.models import VariableLengthPhaseNet

# Constants 
WINDOW_SIZE = 50000
CHECKPOINT_PATH = "vlphasenet_checkpoint.pt"

#Load model 
def load_model():
    model = VariableLengthPhaseNet(in_channels=1, classes=2, sampling_rate=5000000, norm="std")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

# Run inference and stitch output 
def infer_full_trace(model, waveform):
    waveform = waveform.astype(np.float32)
    total_length = len(waveform)
    outputs = []

    n_chunks = int(np.ceil(total_length / WINDOW_SIZE))

    for i in range(n_chunks):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE

        chunk = waveform[start:end]
        chunk = (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-6)
        if len(chunk) < WINDOW_SIZE:
            chunk = np.pad(chunk, (0, WINDOW_SIZE - len(chunk)))  # Pad right
            


        input_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            pred = model(input_tensor)[0].numpy()  # [2, T]

        outputs.append(pred[:, :min(WINDOW_SIZE, total_length - start)])  # Trim to real size

    stitched = np.concatenate(outputs, axis=1)
    return stitched[:, :total_length]

# GUI 
class FullTraceViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Full Trace P-Pick Visualizer")

        self.model = load_model()
        self.waveform = None
        self.probs = None

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.toolbar_frame = tk.Frame(root)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.load_btn = tk.Button(self.control_frame, text="Load .mseed", command=self.load_file)
        self.load_btn.pack(side=tk.RIGHT, padx=10, pady=5)

        self.status_label = tk.Label(self.control_frame, text="No file loaded")
        self.status_label.pack(side=tk.LEFT, padx=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MiniSEED files", "*.mseed")])
        if not file_path:
            return

        st = read(file_path)
        self.waveform = st[0].data.astype(np.float32)
        self.probs = infer_full_trace(self.model, self.waveform)
        self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
        self.update_plot()

    def update_plot(self):
        if self.waveform is None or self.probs is None:
            return

        x = np.arange(len(self.waveform))
        wf = self.waveform
        noise_prob = self.probs[0]
        p_prob = self.probs[1]

        self.ax1.clear()
        self.ax1.plot(x, wf, label="Waveform")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.legend(loc="upper right")

        self.ax2.clear()
        self.ax2.plot(x, noise_prob, label="Noise Prob", alpha=0.6)
        self.ax2.plot(x, p_prob, label="P Prob", alpha=0.8)
        self.ax2.set_ylim(0, 1.05)
        self.ax2.set_ylabel("Probability")
        self.ax2.set_xlabel("Sample Index")
        self.ax2.legend(loc="upper right")

        self.fig.suptitle("Full Trace Prediction")
        self.canvas.draw()

# --- Launch ---
if __name__ == "__main__":
    root = tk.Tk()

    def on_close():
        root.quit()
        root.destroy()

    app = FullTraceViewer(root)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
