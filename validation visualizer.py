import os
import re
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from obspy import read
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Load validation results CSV
validation_df = pd.read_csv("validation_results.csv")

class ValidationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("PhaseNet Validation Viewer")

        self.current_index = 0
        self.data_root = ""

        self.fig, (self.ax_wave, self.ax_probs) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.label = tk.Label(root, text="", font=("Arial", 12))
        self.label.pack()

        nav_frame = tk.Frame(root)
        nav_frame.pack()

        tk.Button(nav_frame, text="Previous", command=self.prev_waveform).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next", command=self.next_waveform).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Select Data Directory", command=self.select_data_root).pack(side=tk.LEFT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.plot_current()

    def on_close(self):
        self.root.quit()
        self.root.destroy()

    def select_data_root(self):
        self.data_root = filedialog.askdirectory(title="Select Root Data Directory")
        self.plot_current()

    def get_waveform_path(self, name):
        print(f"[INFO] Resolving path for: {name}")
        match = re.match(r"p_picks_(Exp_T\d+)_((Run\d+)(_Traces)?)_EventID_(\d+)_trace(\d+)", name)
        if not match:
            print("[ERROR] Regex failed to parse the name.")
            return None, None
        exp = match.group(1)
        run = match.group(2)
        event_id = match.group(5)
        trace_idx = int(match.group(6)) - 1
        filename = f"EventID_{event_id}_WindowSize_0.05s_Data.mseed"
        return os.path.join(self.data_root, exp, run, filename), trace_idx

    def plot_current(self):
        self.ax_wave.clear()
        self.ax_probs.clear()

        if not self.data_root:
            self.label.config(text="Please select a data root directory.")
            return

        row = validation_df.iloc[self.current_index]
        name = row["Name"]
        true_pick = int(row["True Pick (sample)"])
        pred_pick = int(row["Predicted Pick"]) if row["Predicted Pick"] != -1 else None
        probs_file = row["Probs File"]

        print(f"\n[INFO] Plotting {self.current_index}: {name}")
        mseed_path, trace_idx = self.get_waveform_path(name)

        if not os.path.exists(mseed_path):
            self.label.config(text=f"Missing waveform: {name}")
            return

        try:
            stream = read(mseed_path)
            trace = stream[trace_idx]
            waveform = trace.data
            sr = trace.stats.sampling_rate
            times = np.arange(len(waveform)) / sr
        except Exception as e:
            self.label.config(text=f"Error reading waveform: {e}")
            return

        probs_path = os.path.join("prediction_probs", probs_file)
        if not os.path.isfile(probs_path):
            self.label.config(text=f"Missing probs: {probs_file}")
            return

        try:
            probs = np.load(probs_path)
            noise_prob = probs[0]
            p_prob = probs[1]
        except Exception as e:
            self.label.config(text=f"Error loading probs: {e}")
            return

        min_len = min(len(waveform), len(p_prob))
        waveform = waveform[:min_len]
        noise_prob = noise_prob[:min_len]
        p_prob = p_prob[:min_len]
        times = times[:min_len]

        self.ax_wave.plot(times, waveform, color='black', label="Waveform")

        if 0 <= true_pick < len(times):
            self.ax_wave.axvline(true_pick / sr, color='green', linestyle='--', label='True Pick')

        if pred_pick is not None and 0 <= pred_pick < len(times):
            self.ax_wave.axvline(pred_pick / sr, color='red', linestyle='--', label='Predicted Pick')

        crossings = np.where((p_prob[:-1] < 0.5) & (p_prob[1:] >= 0.5))[0]
        for c in crossings:
            self.ax_wave.axvline(times[c], color='orange', linestyle=':', alpha=0.7)

        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.legend(loc='upper right')

        self.ax_probs.plot(times, noise_prob, label="Noise", alpha=0.7)
        self.ax_probs.plot(times, p_prob, label="P-phase", alpha=0.9)
        self.ax_probs.set_ylabel("Probability")
        self.ax_probs.set_xlabel("Time (s)")
        self.ax_probs.legend()

        self.label.config(text=f"{name}")
        self.canvas.draw()

    def next_waveform(self):
        if self.current_index < len(validation_df) - 1:
            self.current_index += 1
            self.plot_current()

    def prev_waveform(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.plot_current()

# Run
if __name__ == "__main__":
    root = tk.Tk()
    app = ValidationViewer(root)
    root.mainloop()
