"""Execute vision-guided robotic pouring sequence using MobileNetV3 coordinate regression and MyCobot 280."""

import time
import cv2
import torch
import torch.nn as nn
from PIL import Image
from pymycobot import MyCobot
from torchvision import models, transforms

PORT = "/dev/tty.usbserial-56E30046201"
BAUD = 115200
HOME_ANGLES = [0, 0, 0, 0, 0, 0]
MODEL_PATH = "coord_predictor.pth"

#  Calibration Constants
K_FOCAL = 500.0
REAL_OBJECT_WIDTH_MM = 80.0
BBOX_WIDTH = 120.0
MIN_SAFE_Z = 160.0


def get_zone_orientation(pred_y: float) -> tuple[float, float, float]:
    """Return End-Effector Orientation (RX, RY, RZ) based on workspace target depth."""
    if pred_y < 25.0:
        return -78.94, -12.0, -70.0
    elif pred_y > 90.0:
        return -79.0, -20.0, -9.0
    return -68.0, -20.0, -48.49


def enforce_joint5_limits(angles_list: list) -> list:
    """Clamp Joint 5 within safe operational limits to prevent inverse kinematics singularity twists."""
    if isinstance(angles_list, list) and len(angles_list) == 6:
        if abs(angles_list[4]) > 10.0:
            print(f"[IK FIX] Correcting Joint 5 twist: {angles_list[4]:.1f}° -> 0.0°")
            angles_list[4] = 0.0
    return angles_list


def main():
    print(f"Connecting to MyCobot on {PORT}...")
    mc = MyCobot(PORT, BAUD)
    time.sleep(1.0)

    print("-> Initializing: Moving robot to safe ready position...")
    mc.send_angles(HOME_ANGLES, 20)
    time.sleep(3.0)

    print("-> Initializing Gripper...")
    try:
        mc.set_gripper_value(70, 70)
        time.sleep(2.0)
    except Exception as e:
        print(f"[WARNING] Gripper setup skipped: {e}")

    # Load Neural Network Architecture
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Capture Frame
    print("[VISION] Capturing live frame from camera...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("[ERROR] Could not grab frame from camera!")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Predict Target Coordinates
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        predictions = model(input_tensor)
        pred_x = predictions[0][0].item()
        pred_y = predictions[0][1].item()

    pred_x = max(-150.0, min(300.0, pred_x))
    pred_y = max(-100.0, min(250.0, pred_y))

    print(f"[AI PREDICTION] Target X: {pred_x:.1f} mm, Y: {pred_y:.1f} mm")

    rx, ry, rz = get_zone_orientation(pred_y)
    print(f"[ZONE ORIENTATION] Target Pose RX: {rx}, RY: {ry}, RZ: {rz}")

    # Depth & Height Estimation
    estimated_dist_mm = (K_FOCAL * REAL_OBJECT_WIDTH_MM) / BBOX_WIDTH
    raw_z = 320.0 - (estimated_dist_mm * 0.15)
    calculated_z = max(MIN_SAFE_Z, raw_z)
    print(f"[DEPTH MATH] Calculated Z Height: {calculated_z:.1f} mm")

    #  Confirmation Window
    print("\n" + "=" * 55)
    print("  Press [SPACEBAR] on the window to execute pour sequence.")
    print("  Press [Q] to quit script.")
    print("=" * 55)

    execute_motion = False
    while True:
        cv2.imshow("Target Frame - [SPACE] Start | [Q] Quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            print("\n[KEYBOARD CONTROL] Spacebar detected! Starting pour sequence...")
            execute_motion = True
            break
        elif key in (ord("q"), ord("Q")):
            print("\n[KEYBOARD CONTROL] Quit key pressed. Exiting safely...")
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not execute_motion:
        print("Exited cleanly.")
        return

    # Task Execution Sequence
    hover_coords = [pred_x, pred_y, 250.0, rx, ry, rz]
    target_coords = [pred_x, pred_y, calculated_z, rx, ry, rz]

    # Step 1: Hover
    print("-> Step 1: Hovering over target...")
    hover_angles = mc.solve_inv_kinematics(hover_coords, HOME_ANGLES)
    hover_angles = enforce_joint5_limits(hover_angles)

    if isinstance(hover_angles, list) and len(hover_angles) == 6:
        mc.send_angles(hover_angles, 20)
    else:
        mc.send_coords(hover_coords, 20, 0)
    time.sleep(3.0)

    # Step 2: Descend
    print("-> Step 2: Descending to target height...")
    curr_angles = mc.get_angles()
    if not isinstance(curr_angles, list) or len(curr_angles) != 6:
        curr_angles = hover_angles if isinstance(hover_angles, list) else HOME_ANGLES

    target_angles = mc.solve_inv_kinematics(target_coords, curr_angles)
    target_angles = enforce_joint5_limits(target_angles)

    if isinstance(target_angles, list) and len(target_angles) == 6:
        mc.send_angles(target_angles, 15)
    else:
        mc.send_coords(target_coords, 15, 0)
    time.sleep(2.5)

    # Step 3 & 4: Pour & Upright
    print("-> Step 3: Pouring...")
    pose_angles = mc.get_angles()

    if isinstance(pose_angles, list) and len(pose_angles) == 6:
        pour_angles = pose_angles.copy()
        pour_angles[5] += 160

        mc.send_angles(pour_angles, 20)
        time.sleep(2.5)

        print("-> Step 4: Returning wrist upright...")
        mc.send_angles(pose_angles, 20)
        time.sleep(2.0)

    # Step 5: Retract & Come back
    print("-> Step 5: Retracting back up...")
    if isinstance(hover_angles, list) and len(hover_angles) == 6:
        mc.send_angles(hover_angles, 20)
    else:
        mc.send_coords(hover_coords, 20, 0)
    time.sleep(2.5)

    print("-> Task Complete: Returning robot home...")
    mc.send_angles(HOME_ANGLES, 20)
    time.sleep(3.0)
    print("Done!")


if __name__ == "__main__":
    main()