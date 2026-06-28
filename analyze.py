"""
Vibrato Emotion Analysis v2
- Model 1: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (Arousal/Valence/Dominance)
- Model 2: speechbrain/emotion-recognition-wav2vec2-IEMOCAP (neutral/happy/sad/angry)
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import pipeline, AutoConfig, AutoProcessor, AutoModelForAudioClassification

# ── paths ──────────────────────────────────────────────────────────────────────
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "results", "data")
os.makedirs(DATA_DIR, exist_ok=True)

SUSTAIN_SEC = 0.60   # last N seconds of each clip to analyse
SR_TARGET   = 16000  # both models expect 16kHz

# ── helpers ────────────────────────────────────────────────────────────────────
def load_sustain(path, duration=SUSTAIN_SEC, sr=SR_TARGET):
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = int(duration * sr)
    return y[-n:] if len(y) >= n else y


def list_files():
    files = []
    for fname in sorted(os.listdir(AUDIO_DIR)):
        if not fname.lower().endswith(".m4a"):
            continue
        cond = "Vibrato" if fname.startswith("V_") else "No Vibrato"
        files.append({"file": fname, "condition": cond,
                       "path": os.path.join(AUDIO_DIR, fname)})
    return files


# ── Model 1: dimensional (A/V/D) ───────────────────────────────────────────────
def run_model1(files):
    """
    Direct model inference — avoids pipeline softmax which collapses
    all three regression outputs to ~0.333.
    Raw logits are the actual continuous regression values.
    """
    MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    print("[Model 1] Loading wav2vec2 dimensional model (direct) …")
    config    = AutoConfig.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()

    # label order from config
    id2label = config.id2label   # {0: 'arousal', 1: 'dominance', 2: 'valence'} (order may vary)
    print(f"  Label order: {id2label}")

    rows = []
    for f in files:
        audio = load_sustain(f["path"])
        # normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        inputs = processor(audio, sampling_rate=SR_TARGET,
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits  # shape (1, 3) — raw regression values
        vals = logits[0].cpu().numpy()
        scores = {id2label[i]: float(vals[i]) for i in range(len(vals))}
        rows.append({
            "file":      f["file"],
            "condition": f["condition"],
            "arousal":   scores.get("arousal",   np.nan),
            "valence":   scores.get("valence",   np.nan),
            "dominance": scores.get("dominance", np.nan),
        })
        print(f"  {f['file']:12s}  A={scores.get('arousal',0):.4f}  "
              f"V={scores.get('valence',0):.4f}  D={scores.get('dominance',0):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "model1_avd.csv"), index=False)
    print("[Model 1] Done. Saved model1_avd.csv\n")
    return df


# ── Model 2: categorical – superb/wav2vec2-base-superb-er (IEMOCAP-based) ─────
def run_model2(files):
    print("[Model 2] Loading superb/hubert-large-superb-er …")
    # Load from local HF cache to avoid network errors
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--superb--hubert-large-superb-er"
        "/snapshots/ef1a2ebfd7cfc424dc7f0fbcdc406e8b794d63bb"
    )
    pipe = pipeline(
        "audio-classification",
        model=cache_path,
        device="cpu",
    )
    # label mapping: hap/neu/sad/ang
    rows = []
    for f in files:
        audio = load_sustain(f["path"])
        out = pipe({"array": audio, "sampling_rate": SR_TARGET}, top_k=None)
        scores = {item["label"]: item["score"] for item in out}
        row = {
            "file":      f["file"],
            "condition": f["condition"],
            "neutral":   scores.get("neu", np.nan),
            "happy":     scores.get("hap", np.nan),
            "sad":       scores.get("sad", np.nan),
            "angry":     scores.get("ang", np.nan),
        }
        rows.append(row)
        top = max(scores, key=scores.get)
        print(f"  {f['file']:12s}  top={top}  "
              f"hap={row['happy']:.3f}  neu={row['neutral']:.3f}  "
              f"sad={row['sad']:.3f}  ang={row['angry']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "model2_categorical.csv"), index=False)
    print("[Model 2] Done. Saved model2_categorical.csv\n")
    return df


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    files = list_files()
    print(f"Found {len(files)} files\n")
    df1 = run_model1(files)
    df2 = run_model2(files)
    # merge for fusion analysis
    merged = pd.merge(df1, df2[["file", "neutral", "happy", "sad", "angry"]], on="file")
    merged.to_csv(os.path.join(DATA_DIR, "merged.csv"), index=False)
    print("Saved merged.csv")
