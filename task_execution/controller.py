import time
from pymycobot.mycobot import MyCobot

PORT = "/dev/tty.usbserial-56E30046201"  # USB port for MyCobot 280

class MyCobotController:

    def __init__(self, enable_robot=False):
        self.enable_robot = enable_robot
        self.last_move_time = 0
        self.cooldown = 1.2

        print("[INIT] Connecting robot (Controller V2)...")
        self.mc = MyCobot(PORT, 115200)
        time.sleep(2)
        print(f"[INIT] Robot V2 Ready | ENABLED = {self.enable_robot}")

    def set_enabled(self, state: bool):
        self.enable_robot = state
        print(f"[STATE] Robot enabled = {self.enable_robot}")
    
    def move_to_target(self, x, y, z=260.0, rx=-78.94, ry=-20.0, rz=-48.49):
        """
        Moves robot to target X, Y, dynamic Z height, and dynamic RX, RY, RZ orientation.
        Applies IK checking and Joint 5 safety clamping.
        """
        print("!!!!!!!! NEW MOVE FUNCTION WITH DYNAMIC Z & ORIENTATION !!!!!!!!")
        print(f"[DEBUG] ENTERED move_to_target | X={x:.1f}, Y={y:.1f}, Z={z:.1f} | RX={rx:.1f}, RY={ry:.1f}, RZ={rz:.1f}")

        if not self.enable_robot:
            print("[SAFE MODE] Robot blocked")
            return

        now = time.time()
        if now - self.last_move_time < self.cooldown:
            return

        # Physical hardware safety limits
        MIN_SAFE_Z = 160.0
        MAX_SAFE_Z = 280.0

        safe_z = max(MIN_SAFE_Z, min(z, MAX_SAFE_Z))
        if safe_z != z:
            print(f"[SAFETY GUARD] Clamped Z from {z:.1f}mm to {safe_z:.1f}mm")

        coords = [x, y, safe_z, rx, ry, rz]
        print("[SENDING COORDS]", coords)

        # Inverse Kinematics & Joint 5 Clamping
        curr_angles = self.mc.get_angles()
        if not isinstance(curr_angles, list) or len(curr_angles) != 6:
            curr_angles = [0, 0, 0, 0, 0, 0]

        target_angles = self.mc.solve_inv_kinematics(coords, curr_angles)

        # Force Joint 5 back to 0.0° if IK tries to twist sideways
        if isinstance(target_angles, list) and len(target_angles) == 6:
            if abs(target_angles[4]) > 10.0:
                print(f"[IK FIX] Correcting Joint 5 twist from {target_angles[4]:.1f}° -> 0.0°")
                target_angles[4] = 0.0
            
            self.mc.send_angles(target_angles, 20)
        else:
            self.mc.send_coords(coords, 20, 0)

        print("[MOVE COMMAND SENT]")
        time.sleep(3.0)

        final = self.mc.get_coords()
        print("[FINAL COORDS]", final)
        self.last_move_time = time.time()

    def pour(self, angle=110, speed=18):
        if not self.enable_robot:
            return

        current = self.mc.get_angles()
        if current is None or not isinstance(current, list) or len(current) != 6:
            print("[ERROR] Could not read joints. Aborting pour.")
            return

    # Copy current joint angles
        pose = current.copy()
    
    # Apply rotation to Joint 6 to pour
        pose[5] = pose[5] + angle  

    # 1. Clamp ALL joints before sending initial pour command (-134.5 to +134.5)
        for i in range(6):
            pose[i] = max(-134.5, min(134.5, float(pose[i])))

        print(f"[POUR EXECUTION] Sending safe joint targets: {pose}")

        try:
            self.mc.send_angles(pose, int(speed))
        except Exception as e:
            print(f"[ERROR] Pour command failed: {e}")
            return

        time.sleep(2.5)

    # 2. Reset Joint 6 back to starting position
        pose[5] = current[5]
    
    # 3. Re-clamp ALL joints before sending reset command
        for i in range(6):
            pose[i] = max(-134.5, min(134.5, float(pose[i])))
        
        try:
            self.mc.send_angles(pose, 20)
        except Exception as e:
            print(f"[ERROR] Return to upright failed: {e}")
            return

        time.sleep(2.0)

def upright(self):
    # Your upright method continues here
    def upright(self):
        """Resets the robot back to safe home position."""
        print("[UPRIGHT] Resetting arm to safe home posture...")

        if not self.enable_robot:
            return

        self.mc.send_angles([0, 0, 0, 0, 0, 0], 20)
        time.sleep(2)

        try:
            self.mc.set_gripper_value(70, 70)
        except Exception as e:
            print(f"[WARNING] Gripper release in upright failed: {e}")
        time.sleep(1)