"""
Rename BD dataset images to BHP/BPP convention
Healthy -> BHP001_hand_spiral_01.jpg, BHP002_hand_spiral_01.jpg, ...
Parkinson -> BPP001_hand_spiral_01.jpg, ...
"""
from pathlib import Path
import os

BD_ROOT = Path(r"D:/Final Semester/Thesis Work/Codes/Dataset/Spiral_Handwriting/BD_Dataset")

if not BD_ROOT.exists():
    print("BD dataset folder not found:", BD_ROOT)
    raise SystemExit(1)

for cls, prefix in [("Healthy", "BHP"), ("Parkinson", "BPP")]:
    folder = BD_ROOT / cls
    if not folder.exists():
        print("Missing folder:", folder)
        continue
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    counter = 1
    for f in files:
        new_name = f"{prefix}{counter:03d}_hand_spiral_01{f.suffix.lower()}"
        new_path = folder / new_name
        print(f"Renaming {f.name} -> {new_name}")
        os.rename(f, new_path)
        counter += 1

print("BD dataset renaming complete")
