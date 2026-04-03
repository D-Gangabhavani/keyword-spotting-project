import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pickle


MODEL_PATH = "model/cnn_bilstm_model.h5"
LABELS_PATH = "model/labels.pkl"
SR = 16000
N_MFCC = 40
MAX_LEN = 100
WINDOW_SIZE = 1.0       
HOP_SIZE = 0.1          
CONF_THRESHOLD = 0.7

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)  


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    if len(mfcc) > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad = MAX_LEN - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
    return mfcc

def merge_segments(segments, gap=0.2):
    if len(segments) == 0:
        return []
    merged = []
    current = segments[0]
    for seg in segments[1:]:
        if seg["start"] <= current["end"] + gap:
            current["end"] = max(current["end"], seg["end"])
            current["confidence"] = max(current["confidence"], seg["confidence"])
            current["label"] = seg["label"]
        else:
            merged.append(current)
            current = seg
    merged.append(current)
    return merged


def test_audio(audio_path, keyword=None):
    print("\nLoading audio...")
    audio, _ = librosa.load(audio_path, sr=SR)
    window_len = int(WINDOW_SIZE * SR)
    hop_len = int(HOP_SIZE * SR)

    detections = []

    for i in range(0, len(audio) - window_len + 1, hop_len):
        chunk = audio[i:i + window_len]
        mfcc = extract_mfcc(chunk)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)

        pred = model.predict(mfcc, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)

        if confidence >= CONF_THRESHOLD:
            
            chunk_mid = (i + window_len / 2) / SR
            detections.append({
                "time": chunk_mid,
                "conf": float(confidence),
                "label": keyword if keyword else labels[class_id]
            })

    segments = []
    if detections:
        start = detections[0]["time"]
        end = detections[0]["time"]
        best_conf = detections[0]["conf"]
        label = detections[0]["label"]

        for d in detections[1:]:
            if d["time"] <= end + HOP_SIZE + 0.05:
                end = d["time"]
                best_conf = max(best_conf, d["conf"])
            else:
                segments.append({"start": start, "end": end, "label": label, "confidence": best_conf})
                start = d["time"]
                end = d["time"]
                best_conf = d["conf"]
                label = d["label"]
        segments.append({"start": start, "end": end, "label": label, "confidence": best_conf})
    if not segments:
        print(" Keyword Not Found")
    else:
        print("\n === KEYWORD DETECTED ===\n")
        for seg in segments:
            print(f"Start: {seg['start']:.3f}s | End: {seg['end']:.3f}s | {seg['label']} | Confidence: {seg['confidence']:.3f}")


if __name__ == "__main__":
    audio_path = input("Enter audio path: ").strip()
    keyword = input("Enter keyword : ").strip()
    if keyword == "":
        keyword = None
    if not os.path.exists(audio_path):
        print(" File not found!")
    else:
        test_audio(audio_path, keyword)