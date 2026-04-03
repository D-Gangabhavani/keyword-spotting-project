import os
from collections import Counter

BASE_DIR = r"IISC_KWS/ALL_ANNOTATIONS_PHASE_1"

X = []

member_count = Counter()
label_set = set()   

print("BASE_DIR EXISTS:", os.path.exists(BASE_DIR))

members = sorted(os.listdir(BASE_DIR))
print("Total Members Found:", len(members))


for member in members:

    ann_dir = os.path.join(BASE_DIR, member, "ANNOTATIONS")

    if not os.path.isdir(ann_dir):
        continue


    for file in os.listdir(ann_dir):

        if not file.endswith(".txt"):
            continue

        path = os.path.join(ann_dir, file)


        with open(path, "r", errors="ignore") as f:

            for line in f:

                parts = line.strip().split("\t")

                if len(parts) != 3:
                    continue

                try:
                    float(parts[0])
                    float(parts[1])
                except:
                    continue


                label = parts[2]          
                label_set.add(label)     

                X.append(member)
                member_count[member] += 1


print("\n==============================")
print("FINAL DATASET SUMMARY")
print("==============================")

print("Total Samples :", len(X))
print("Total Members :", len(member_count))
print("Unique Labels :", len(label_set))   


print("\n==============================")
print("SAMPLES PER MEMBER")
print("==============================\n")

for member, count in member_count.most_common():
    print(f"{member}  -->  {count}")


if len(X) == 0:
    raise ValueError(" Dataset is EMPTY — annotation parsing failed")