import os
import librosa
import numpy as np

BASE_DIR = r"IISC_KWS/ALL_ANNOTATIONS_PHASE_1"
OUT_DIR = "dataset"

os.makedirs(OUT_DIR, exist_ok=True)
SR = 16000
N_MFCC = 40
MAX_LEN = 100

def find_audio(folder):

    for root, dirs, files in os.walk(folder):

        for f in files:

            if f.lower().endswith((".wav", ".mp3", ".flac")):
                return os.path.join(root, f)

    return None


# =============================
# EXTRACT MFCC
# =============================

def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC
    )

    mfcc = mfcc.T

    # Pad / Trim
    if len(mfcc) > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]

    else:
        pad = MAX_LEN - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))

    return mfcc


# =============================
# MAIN
# =============================

X = []
Y = []

label_map = {}
label_id = 0

count = 0

print("\nScanning dataset...\n")


# =============================
# LOOP MEMBERS
# =============================

for member in os.listdir(BASE_DIR):

    member_path = os.path.join(BASE_DIR, member)

    if not os.path.isdir(member_path):
        continue


    ann_dir = os.path.join(member_path, "ANNOTATIONS")

    if not os.path.isdir(ann_dir):
        print("No ANNOTATIONS:", member)
        continue


    audio_file = find_audio(member_path)

    if audio_file is None:
        print("No audio:", member)
        continue


    print("Using:", audio_file)


    # Load audio
    audio, _ = librosa.load(audio_file, sr=SR)


    # =============================
    # READ ANNOTATIONS
    # =============================

    for file in os.listdir(ann_dir):

        if not file.endswith(".txt"):
            continue


        path = os.path.join(ann_dir, file)


        with open(path, errors="ignore") as f:

            for line in f:

                line = line.strip()

                if line == "":
                    continue


                parts = line.split()

                if len(parts) < 3:
                    continue


                # SAFE CONVERSION
                try:
                    start = float(parts[0])
                    end   = float(parts[1])
                except:
                    continue   # skip bad timestamp


                label = parts[2]


                # Convert to samples
                s = int(start * SR)
                e = int(end   * SR)


                segment = audio[s:e]


                # Skip very small segments
                if len(segment) < SR * 0.1:
                    continue


                mfcc = extract_mfcc(segment)


                # Label mapping
                if label not in label_map:
                    label_map[label] = label_id
                    label_id += 1


                X.append(mfcc)
                Y.append(label_map[label])

                count += 1


# =============================
# SAVE DATA
# =============================

X = np.array(X)
Y = np.array(Y)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "Y.npy"), Y)
np.save(os.path.join(OUT_DIR, "labels.npy"), label_map)


print("\nDONE ")
print("Samples:", len(X))
print("Classes:", len(label_map))