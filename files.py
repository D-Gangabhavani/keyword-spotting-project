import os

BASE_DIR = r"IISC_KWS/ALL_ANNOTATIONS_PHASE_1"

total_members = 0
total_annotations = 0
total_audios = 0

print("Scanning base directory...")
print("BASE_DIR =", BASE_DIR)
print("-" * 50)

for member in os.listdir(BASE_DIR):

    member_path = os.path.join(BASE_DIR, member)

    if not os.path.isdir(member_path):
        continue

    annotations_dir = os.path.join(member_path, "ANNOTATIONS")
    datasets_dir = os.path.join(member_path, "DATASETS")

    if not os.path.isdir(annotations_dir) or not os.path.isdir(datasets_dir):
        continue

    total_members += 1

    ann_files = [
        f for f in os.listdir(annotations_dir)
        if f.lower().endswith(".txt")
    ]

    audio_files = [
        f for f in os.listdir(datasets_dir)
        if f.lower().endswith((".wav", ".mp3", ".flac"))
    ]

    total_annotations += len(ann_files)
    total_audios += len(audio_files)

    print(f" {member}")
    print(f"   Annotations: {len(ann_files)}")
    print(f"   Audios     : {len(audio_files)}")

print("\n" + "=" * 50)
print("FINAL COUNT")
print("Valid members found :", total_members)
print("Total annotations   :", total_annotations)
print("Total audios        :", total_audios)
print("=" * 50)