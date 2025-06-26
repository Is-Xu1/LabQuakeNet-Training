import os
import re
import torch
import pandas as pd
import numpy as np
from obspy import read
from seisbench.models import VariableLengthPhaseNet

# --- Constants ---
CHECKPOINT_PATH = "vlphasenet_checkpoint.pt"
VALIDATION_CSV = "p_picks_validation.csv"
DATA_DIR = r"c:\Users\hiriy\Downloads\python\RTX2025\Data"
WINDOW_SIZE = 50000
SAMPLING_RATE = 100
TOLERANCE = 20  # samples
OUTPUT_CSV = "validation_results.csv"
PROB_DIR = "prediction_probs"

os.makedirs(PROB_DIR, exist_ok=True)

# --- Helper Functions ---
def parse_name(name):
    match = re.match(r"p_picks_Exp_(.+?)_(Run.+?)_(EventID_\d+)_trace(\d+)", name)
    if not match:
        raise ValueError(f"Invalid name format: {name}")
    exp, run, event, trace_num = match.groups()
    trace_index = int(trace_num) - 1
    return exp, run, event, trace_index

def load_waveform(name_key):
    exp, run, event, trace_index = parse_name(name_key)
    folder = os.path.join(DATA_DIR, f"Exp_{exp}", run)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Run folder not found: {folder}")
    mseed_files = [f for f in os.listdir(folder) if event in f and f.endswith(".mseed")]
    if not mseed_files:
        raise FileNotFoundError(f"No .mseed file found for event {event} in {folder}")
    stream = read(os.path.join(folder, mseed_files[0]))
    return stream[trace_index].data.astype(np.float32), os.path.join(folder, mseed_files[0])

# --- Main Validation Function ---
def validate():
    print("🔍 Loading model...")
    model = VariableLengthPhaseNet(in_channels=1, classes=3, phases="NPS", sampling_rate=SAMPLING_RATE, norm="std")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    df = pd.read_csv(VALIDATION_CSV)
    results = []
    total_picks = 0
    successful_picks = 0
    pick_errors = []

    for i, row in df.iterrows():
        name_key = row["Name"]
        pick = int(row["marked_point"])
        if pick == -1:
            continue

        try:
            waveform, _ = load_waveform(name_key)
        except Exception as e:
            print(f"⏩ {name_key} - {e}")
            continue

        original_length = len(waveform)
        pad_len = (WINDOW_SIZE - (original_length % WINDOW_SIZE)) % WINDOW_SIZE
        padded_waveform = np.pad(waveform, (0, pad_len), mode="constant")
        num_windows = len(padded_waveform) // WINDOW_SIZE

        all_noise = []
        all_p = []

        for j in range(num_windows):
            start = j * WINDOW_SIZE
            end = start + WINDOW_SIZE
            chunk = padded_waveform[start:end]
            x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                y_hat = model(x)
                y_hat = torch.softmax(y_hat, dim=1)
                y_hat_np = y_hat.squeeze(0).cpu().numpy()

            all_noise.append(y_hat_np[0])
            all_p.append(y_hat_np[1])

        noise_full = np.concatenate(all_noise)[:original_length]
        p_full = np.concatenate(all_p)[:original_length]

        probs = np.stack([noise_full, p_full])
        prob_filename = os.path.join(PROB_DIR, f"probs_{i}.npy")
        np.save(prob_filename, probs)

        over_threshold = np.where(p_full >= 0.5)[0]
        if len(over_threshold) > 0:
            pred_idx = int(over_threshold[0])
            abs_error = abs(pred_idx - pick)
            success = abs_error <= TOLERANCE
        else:
            pred_idx = -1
            abs_error = "-"
            success = False

        results.append({
            "Index": i,
            "Name": name_key,
            "True Pick (sample)": pick,
            "Predicted Pick": pred_idx,
            "Abs Error": abs_error,
            "Success": success,
            "Probs File": os.path.basename(prob_filename)
        })

        total_picks += 1
        if success:
            successful_picks += 1
        if abs_error != "-":
            pick_errors.append(abs_error)

        print(f"[{i+1}/{len(df)}] {name_key} → Error: {abs_error} → {'✅' if success else '❌'}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\n📊 Evaluation Summary")
    print(f"Total Picks: {total_picks}")
    print(f"Pick Accuracy (±{TOLERANCE} samples): {successful_picks / total_picks:.2%}")
    if pick_errors:
        print(f"MAE: {np.mean(pick_errors):.2f}")
        print(f"RMSE: {np.sqrt(np.mean(np.square(pick_errors))):.2f}")
    print(f"📁 Results saved to {OUTPUT_CSV}")
    print(f"📁 Prediction probs saved in: {PROB_DIR}")

if __name__ == "__main__":
    validate()
