import os
import shutil

# ============================================================
#  PATHS - already set for your machine!
# ============================================================
UBNORMAL_ROOT = r"C:\Users\khanm\Desktop\lab_project\PLOVAD\UBnormal"
PLOVAD_LIST = r"C:\Users\khanm\Desktop\lab_project\Plovad\src\list\ubnormal"
OUTPUT_ROOT   = r"C:\Users\khanm\Desktop\lab_project\PLOVAD\videos"

# ============================================================
#  LIST FILES from PLOVAD
# ============================================================
list_files = {
    "train": os.path.join(PLOVAD_LIST, "ub-vit-train.list"),
    "test":  os.path.join(PLOVAD_LIST, "ub-vit-test.list"),
}

# ============================================================
#  Create output folders
# ============================================================
for split in ["train", "test", "validation"]:
    for label in ["abnormal", "normal"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, label), exist_ok=True)

print("✅ Output folders created!")

# ============================================================
#  Build a map of all .mp4 files in UBnormal
#  { "abnormal_scene_3_scenario_4": full_path_to_.mp4 }
# ============================================================
print("\n🔍 Scanning UBnormal for .mp4 files...")
video_map = {}
for root, dirs, files in os.walk(UBNORMAL_ROOT):
    for file in files:
        if file.endswith(".mp4"):
            name = os.path.splitext(file)[0]  # remove .mp4
            video_map[name] = os.path.join(root, file)

print(f"✅ Found {len(video_map)} .mp4 files in UBnormal")

# ============================================================
#  Read .list files and copy videos to correct folders
# ============================================================
not_found = []

for split, list_path in list_files.items():
    print(f"\n📋 Processing {split} list: {list_path}")
    
    with open(list_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # e.g. "train/abnormal/abnormal_scene_27_scenario_8.npy"
        parts = line.replace("\\", "/").split("/")
        
        if len(parts) < 3:
            continue

        label    = parts[1]   # "abnormal" or "normal"
        npy_name = parts[2]   # "abnormal_scene_27_scenario_8.npy"
        vid_name = os.path.splitext(npy_name)[0]  # remove .npy

        # Find the .mp4 in our map
        if vid_name in video_map:
            src = video_map[vid_name]
            dst_folder = os.path.join(OUTPUT_ROOT, split, label)
            dst = os.path.join(dst_folder, vid_name + ".mp4")

            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  ✅ Copied: {vid_name}.mp4 → {split}/{label}/")
            else:
                print(f"  ⏭️  Already exists: {vid_name}.mp4")
        else:
            print(f"  ❌ NOT FOUND: {vid_name}.mp4")
            not_found.append(vid_name)

# ============================================================
#  Summary
# ============================================================
print("\n" + "="*50)
print("✅ DONE! Summary:")
for split in ["train", "test"]:
    for label in ["abnormal", "normal"]:
        folder = os.path.join(OUTPUT_ROOT, split, label)
        count = len([f for f in os.listdir(folder) if f.endswith(".mp4")])
        print(f"  {split}/{label}: {count} videos")

if not_found:
    print(f"\n⚠️  {len(not_found)} videos not found in UBnormal:")
    for v in not_found:
        print(f"   - {v}")
else:
    print("\n🎉 All videos found and copied successfully!")

print("\n👉 Next step: Run extract_frames_1.py")