"""
Full pipeline replicating the original repo:
  Step 1 – preprocess.py  : m4a → wav, silence trim, RMS normalise to 0.08
  Step 2 – segment_sustain: energy-based endpoint, last 0.60s + 0.10s pad, fade
  Step 3 – run_emotion_ai : 16 kHz direct inference (raw logits for M1, softmax for M2)
"""

import os
import subprocess
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from pathlib import Path
from transformers import AutoConfig, AutoProcessor, AutoFeatureExtractor, AutoModelForAudioClassification
from scipy.special import softmax as scipy_softmax
import warnings
warnings.filterwarnings("ignore")

# ── dirs ───────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
RAW_DIR   = BASE / "audio"
PROC_DIR  = BASE / "data" / "processed"
SEG_DIR   = BASE / "data" / "segments"
DATA_DIR  = BASE / "results" / "data"
for d in [PROC_DIR, SEG_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── constants (matching original) ─────────────────────────────────────────────
SR_PROC     = 44100
SR_SEG      = 22050
SR_INFER    = 16000
TARGET_RMS  = 0.08
SEG_LEN     = 0.60
POST_PAD    = 0.10


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – preprocess  (preprocess.py)
# ══════════════════════════════════════════════════════════════════════════════
def preprocess():
    print("=== Step 1: preprocess ===")
    m4a_files = sorted(RAW_DIR.glob("*.m4a"))
    for src in m4a_files:
        tmp_wav = PROC_DIR / (src.stem + "_tmp.wav")
        out_wav = PROC_DIR / (src.stem + ".wav")
        if out_wav.exists():
            continue
        # ffmpeg: m4a → 44100 Hz mono wav
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR_PROC), str(tmp_wav)],
            check=True, capture_output=True
        )
        # load, trim silence, RMS normalise
        y, _ = librosa.load(tmp_wav, sr=SR_PROC, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms > 0:
            y = y * (TARGET_RMS / rms)
        y = np.clip(y, -1.0, 1.0)
        sf.write(out_wav, y, SR_PROC)
        tmp_wav.unlink(missing_ok=True)
        print(f"  preprocessed {src.name}")
    print(f"  {len(list(PROC_DIR.glob('*.wav')))} files in {PROC_DIR}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – segment sustain  (segment_sustain.py)
# ══════════════════════════════════════════════════════════════════════════════
def active_end_time(y, sr):
    rms    = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times  = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    if len(rms) == 0 or np.max(rms) <= 0:
        return librosa.get_duration(y=y, sr=sr)
    threshold = max(np.percentile(rms, 60), 0.12 * float(np.max(rms)))
    active = np.where(rms > threshold)[0]
    if len(active) == 0:
        return librosa.get_duration(y=y, sr=sr)
    return float(times[active[-1]])

def segment_sustain():
    print("=== Step 2: segment sustain ===")
    rows = []
    for path in sorted(PROC_DIR.glob("*.wav")):
        out_path = SEG_DIR / path.name
        y, sr = librosa.load(path, sr=SR_SEG, mono=True)
        dur  = librosa.get_duration(y=y, sr=sr)
        end  = min(dur, active_end_time(y, sr) + POST_PAD)
        start = max(0.0, end - SEG_LEN)
        seg  = y[int(start * sr): int(end * sr)]
        # fade in/out 15 ms
        fade_n = min(int(0.015 * sr), len(seg) // 4)
        if fade_n > 0:
            fade = np.linspace(0, 1, fade_n)
            seg[:fade_n]  *= fade
            seg[-fade_n:] *= fade[::-1]
        sf.write(out_path, seg, sr)
        rows.append({
            "file": path.name,
            "condition": "no_vibrato" if path.stem.startswith("NV_") else "vibrato",
            "seg_start": round(start, 3),
            "seg_end":   round(end, 3),
            "seg_dur":   round(librosa.get_duration(y=seg, sr=sr), 3),
        })
        print(f"  {path.name:12s}  start={start:.2f}s  end={end:.2f}s")
    pd.DataFrame(rows).to_csv(DATA_DIR / "segment_metadata.csv", index=False)
    print(f"  {len(rows)} segments saved to {SEG_DIR}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – inference  (run_emotion_ai.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_audio_16k(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR_INFER, mono=True)
    if np.max(np.abs(audio)) > 0:
        audio = audio / max(1.0, np.max(np.abs(audio)))
    return audio.astype(np.float32)

def condition_from_name(stem: str) -> str:
    return "No Vibrato" if stem.startswith("NV_") else "Vibrato"

def run_model1():
    MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    print("[Model 1] wav2vec2-MSP-dim …")
    config = AutoConfig.from_pretrained(MODEL)
    try:
        processor = AutoProcessor.from_pretrained(MODEL)
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(MODEL)
    model = AutoModelForAudioClassification.from_pretrained(MODEL)
    model.eval()
    id2label = {int(k): str(v).lower() for k, v in config.id2label.items()}
    print(f"  labels: {id2label}")

    rows = []
    for path in sorted(SEG_DIR.glob("*.wav")):
        audio  = load_audio_16k(path)
        inputs = processor(audio, sampling_rate=SR_INFER, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze().detach().cpu().numpy()
        logits = np.atleast_1d(logits)
        scores = {id2label[i]: float(logits[i]) for i in range(len(logits))}
        rows.append({
            "file":      path.name,
            "condition": condition_from_name(path.stem),
            "arousal":   scores.get("arousal",   np.nan),
            "valence":   scores.get("valence",   np.nan),
            "dominance": scores.get("dominance", np.nan),
        })
        print(f"  {path.name:12s}  A={scores.get('arousal',0):+.4f}  "
              f"V={scores.get('valence',0):+.4f}  D={scores.get('dominance',0):+.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "model1_avd.csv", index=False)
    print("[Model 1] Done.\n")
    return df

def run_model2():
    cache_path = Path.home() / (
        ".cache/huggingface/hub/models--superb--hubert-large-superb-er"
        "/snapshots/ef1a2ebfd7cfc424dc7f0fbcdc406e8b794d63bb"
    )
    print("[Model 2] HuBERT-IEMOCAP …")
    config = AutoConfig.from_pretrained(str(cache_path))
    try:
        processor = AutoProcessor.from_pretrained(str(cache_path))
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(str(cache_path))
    model = AutoModelForAudioClassification.from_pretrained(str(cache_path))
    model.eval()
    id2label = {int(k): str(v).lower() for k, v in config.id2label.items()}
    print(f"  labels: {id2label}")

    rows = []
    for path in sorted(SEG_DIR.glob("*.wav")):
        audio  = load_audio_16k(path)
        inputs = processor(audio, sampling_rate=SR_INFER, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze().detach().cpu().numpy()
        probs = scipy_softmax(np.atleast_1d(logits))
        scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        rows.append({
            "file":      path.name,
            "condition": condition_from_name(path.stem),
            "neutral":   scores.get("neu", np.nan),
            "happy":     scores.get("hap", np.nan),
            "sad":       scores.get("sad", np.nan),
            "angry":     scores.get("ang", np.nan),
        })
        top = max(scores, key=scores.get)
        print(f"  {path.name:12s}  top={top}  "
              f"hap={rows[-1]['happy']:.3f}  sad={rows[-1]['sad']:.3f}  "
              f"neu={rows[-1]['neutral']:.3f}  ang={rows[-1]['angry']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "model2_categorical.csv", index=False)
    print("[Model 2] Done.\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    preprocess()
    segment_sustain()
    df1 = run_model1()
    df2 = run_model2()
    merged = pd.merge(df1, df2[["file","neutral","happy","sad","angry"]], on="file")
    merged.to_csv(DATA_DIR / "merged.csv", index=False)
    print("Saved merged.csv - pipeline complete.")
