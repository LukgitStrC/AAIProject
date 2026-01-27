from ultralytics.data.split import autosplit
import os
import shutil



# autosplit(
#     path="images",
#     weights=(0.85, 0.1, 0.05),  # (train, validation, test) fractional splits
#     annotated_only=False,  # split only images with annotation file when True
# )

# create folders
os.makedirs("images/train", exist_ok=True)
os.makedirs("images/val", exist_ok=True)
os.makedirs("images/test", exist_ok=True)

os.makedirs("labels/train", exist_ok=True)
os.makedirs("labels/val", exist_ok=True)
os.makedirs("labels/test", exist_ok=True)


# Mapping of txt files to target folders
txt_to_folder = {
    "autosplit_train.txt": ("images/train", "labels/train"),
    "autosplit_val.txt": ("images/val", "labels/val"),
    "autosplit_test.txt": ("images/test", "labels/test"),
}

# Move files based on txt lists
for txt_file, (img_target_dir, label_target_dir) in txt_to_folder.items():
    with open(txt_file, "r") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue

            path = path.lstrip("./")  # remove leading ./
            base_name = os.path.splitext(os.path.basename(path))[0]  # without extension

            # Move image (any common image extension)
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
                img_src = os.path.join("images", base_name + ext)
                if os.path.exists(img_src):
                    shutil.move(img_src, os.path.join(img_target_dir, base_name + ext))
                    break

            # Move label
            label_src = os.path.join("labels", base_name + ".txt")
            if os.path.exists(label_src):
                shutil.move(label_src, os.path.join(label_target_dir, base_name + ".txt"))
            else:
                print(f"⚠️ Label nicht gefunden: {label_src}")