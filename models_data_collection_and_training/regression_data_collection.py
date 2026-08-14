"""Interactive dataset collection utility for pairing camera frames with target workspace coordinates."""

import csv
import glob
import os
import cv2

os.makedirs("coord_dataset/images", exist_ok=True)
csv_file = "coord_dataset/labels.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "x_coord", "y_coord"])

existing_files = glob.glob("coord_dataset/images/img_*.jpg")
if existing_files:
    ids = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in existing_files]
    img_id = max(ids) + 1
else:
    img_id = 0

cap = cv2.VideoCapture(0)

print("=== DATA COLLECTION ===")
print("Position target object, then enter ground-truth X,Y coordinates in terminal.")
print("Enter 'q' to exit.")

while True:
    for _ in range(5):
        ret, frame = cap.read()

    if not ret:
        print("[ERROR] Camera stream interrupted.")
        break

    cv2.imshow("Webcam Live Feed", frame)
    cv2.waitKey(1)

    val = input(f"\n[Img ID: {img_id}] Enter 'X,Y' coordinates (or 'q' to quit): ")
    if val.lower().strip() == "q":
        break

    try:
        x, y = map(float, val.split(","))
        img_name = f"img_{img_id}.jpg"
        img_path = os.path.join("coord_dataset/images", img_name)

        cv2.imwrite(img_path, frame)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([img_name, x, y])

        print(f"Logged: {img_name} -> X: {x:.1f}, Y: {y:.1f}")
        img_id += 1
    except ValueError:
        print("[ERROR] Invalid format. Expected format: 150,200")

cap.release()
cv2.destroyAllWindows()