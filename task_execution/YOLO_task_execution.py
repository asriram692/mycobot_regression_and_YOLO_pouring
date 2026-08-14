"""
YOLO Perception and Robotic Pouring Control Pipeline for MyCobot 280.

This module processes live webcam feed using a fine-tuned YOLO model to detect 
target vessels (cups, bowls), transforms pixel coordinates to robot workspace 
coordinates, and executes adaptive robotic positioning and pouring tasks.
"""

import math
import time
import cv2
import numpy as np
from ultralytics import YOLO
from controller import MyCobotController

# --- Workspace & Safety Config ---
ROBOT_ORIGIN_X, ROBOT_ORIGIN_Y = 200.0, 0.0  # Base offsets (mm)
PIXEL_SCALE = 0.26                            # mm per pixel
APPROACH_OFFSET_X, SPOUT_OFFSET_Y = -5.0, 30.0

K_FOCAL = 28000.0       # Focal length scaling for height-based depth
MIN_SAFE_Z, MAX_SAFE_Z = 110.0, 280.0
CLEARANCE_OFFSET = 20.0 # Joint 6 roll clearance (mm)

THRESHOLD_CUP, THRESHOLD_BOWL = 0.8, 0.7
LOCK_FRAMES = 10        # Required consecutive frames for target lock


def get_zone_orientation(cy: float) -> tuple[float, float, float]:
    """Return end-effector rotation angles (RX, RY, RZ) based on image Y-coordinate."""
    if cy > 700:
        return -78.94, -12.0, -70.0
    elif cy < 300:
        return -79.0, -20.0, -9.0
    return -78.94, -20.0, -48.49


# --- Setup & Initialization ---
robot = MyCobotController(enable_robot=True)
ROBOT_ENABLED = False
robot.mc.send_angles([0, 0, 0, 0, 0, 0], 30)

try:
    robot.mc.set_gripper_value(70, 70)
    time.sleep(2)
except Exception as e:
    print(f"[WARNING] Gripper init skipped: {e}")

model = YOLO("weights/best.pt")

cap = cv2.VideoCapture(0)
time.sleep(2)
if not cap.isOpened():
    raise RuntimeError("Failed to open camera stream.")

# State tracking
MODE = "IDLE"  # "IDLE" or "TEST"
lock_counter = 0
LOCKED_CLASS = None

total_frames = missed_frames = detected_frames = 0
fps_sum = fps_count = 0
prev_cx = prev_cy = None
stability_sum = stability_count = 0

cup_gt = cup_high_conf = bowl_gt = bowl_high_conf = 0
first_detection_made = False
frames_after_first = continuous_target_frames = current_streak = max_streak = 0

total_attempts = position_success_count = pour_success_count = 0
task_times = []


# --- Main Loop ---
while True:
    start_time = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cx0, cy0 = 960, 540  # Optical center reference

    results = model.predict(frame, conf=0.25, verbose=False)
    boxes = results[0].boxes

    total_frames += 1
    if len(boxes) == 0:
        missed_frames += 1
    else:
        detected_frames += 1

    target_label = target_cx = target_cy = target_box_h = None
    min_dist = 1e9

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_h = y2 - y1
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Log recall stats
        if label == "cup":
            cup_gt += 1
            if conf >= THRESHOLD_CUP:
                cup_high_conf += 1
        elif label == "bowl":
            bowl_gt += 1
            if conf >= THRESHOLD_BOWL:
                bowl_high_conf += 1

        if (label == "cup" and conf < THRESHOLD_CUP) or (label == "bowl" and conf < THRESHOLD_BOWL):
            continue

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Map to robot frame & find target closest to base
        rx_cand = ROBOT_ORIGIN_X + (cx - cx0) * 0.2
        ry_cand = ROBOT_ORIGIN_Y - (cy - cy0) * 0.2
        p_dist = rx_cand**2 + ry_cand**2

        if p_dist < min_dist:
            min_dist = p_dist
            target_label, target_cx, target_cy, target_box_h = label, cx, cy, box_h

        # Visual overlays
        cv2.circle(frame, (cx0, cy0), 10, (0, 0, 255), -1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Jitter tracking
    if target_cx is not None and prev_cx is not None:
        stability_sum += math.sqrt((target_cx - prev_cx)**2 + (target_cy - prev_cy)**2)
        stability_count += 1
    prev_cx, prev_cy = target_cx, target_cy

    # Detection streak tracking
    if target_label is not None:
        if not first_detection_made:
            first_detection_made = True
            LOCKED_CLASS = target_label

        if target_label == LOCKED_CLASS:
            current_streak += 1
            continuous_target_frames += 1
        else:
            max_streak = max(max_streak, current_streak)
            current_streak, LOCKED_CLASS = 1, target_label
    else:
        max_streak = max(max_streak, current_streak)
        current_streak = 0

    if first_detection_made:
        frames_after_first += 1

    # --- Robot Action Execution ---
    if ROBOT_ENABLED and MODE == "TEST" and target_label is not None:
        lock_counter = lock_counter + 1 if target_label == LOCKED_CLASS else 1

        if lock_counter >= LOCK_FRAMES:
            total_attempts += 1
            action_start = time.time()

            pour_x = ROBOT_ORIGIN_X + (target_cx - cx0) * PIXEL_SCALE + APPROACH_OFFSET_X
            pour_y = ROBOT_ORIGIN_Y - (target_cy - cy0) * PIXEL_SCALE + SPOUT_OFFSET_Y

            # Calculate dynamic Z height from box height
            if target_box_h and target_box_h > 0:
                raw_z = 275.0 - ((K_FOCAL / target_box_h) * 0.15)
                calculated_z = max(MIN_SAFE_Z, max(MIN_SAFE_Z, min(raw_z, MAX_SAFE_Z)) - CLEARANCE_OFFSET)
            else:
                calculated_z = 220.0

            rx_val, ry_val, rz_val = get_zone_orientation(target_cy)

            try:
                robot.move_to_target(pour_x, pour_y, calculated_z, rx=rx_val, ry=ry_val, rz=rz_val)
                position_success_count += 1
                time.sleep(2)

                POUR_ANGLE = 110.0
                pour_speed = 18 if LOCKED_CLASS == "cup" else 10

                try:
                    robot.pour(angle=POUR_ANGLE, speed=pour_speed)
                except TypeError:
                    robot.pour(POUR_ANGLE)

                pour_success_count += 1
                robot.upright()
                task_times.append(time.time() - action_start)

            except Exception as e:
                print(f"[ERROR] Task failed: {e}")
                robot.upright()

            time.sleep(1.5)
            lock_counter = 0

    elif target_label is None or MODE == "IDLE":
        lock_counter = 0

    # UI Feed
    fps = 1 / (time.perf_counter() - start_time)
    fps_sum += fps
    fps_count += 1

    cv2.putText(frame, f"FPS: {fps:.1f} | ROBOT: {'ON' if ROBOT_ENABLED else 'OFF'} | MODE: {MODE}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("YOLO Control Studio", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        ROBOT_ENABLED = True
        robot.set_enabled(True)
    elif key == ord('s'):
        ROBOT_ENABLED = False
        robot.set_enabled(False)
    elif key == ord('t'):
        MODE = "TEST"
    elif key == ord('i'):
        MODE = "IDLE"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# --- Metrics Summary ---
avg_fps = fps_sum / fps_count if fps_count > 0 else 0
miss_rate = missed_frames / total_frames if total_frames > 0 else 0
avg_stability = stability_sum / stability_count if stability_count > 0 else 0

total_gt = cup_gt + bowl_gt
total_det = cup_high_conf + bowl_high_conf
recall = (total_det / total_gt * 100) if total_gt > 0 else 0.0

pos_rate = (position_success_count / total_attempts * 100) if total_attempts > 0 else 0.0
pour_rate = (pour_success_count / total_attempts * 100) if total_attempts > 0 else 0.0
avg_time = sum(task_times) / len(task_times) if task_times else 0.0

print(f"\n--- EVALUATION SUMMARY ---")
print(f"Frames: {total_frames} | FPS: {avg_fps:.1f} | Recall: {(1-miss_rate)*100:.1f}%")
print(f"Instance Recall: {recall:.1f}% | Spatial Jitter: {avg_stability:.3f} px/frame")
print(f"Attempts: {total_attempts} | Positioning Acc: {pos_rate:.1f}% | Pour Acc: {pour_rate:.1f}% | Avg Time: {avg_time:.2f}s")