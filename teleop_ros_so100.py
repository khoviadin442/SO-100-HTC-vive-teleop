import threading
import openvr
import time
import numpy as np
import pinocchio as pin
from numpy.linalg import solve
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray

URDF = "/home/kyrylo/tst/so100.urdf"
ARM = ['Shoulder_Rotation', 'Shoulder_Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll']

SCALE, GAIN, DAMP, ERR_MAX, REACH, VEL_SCALE, RATE = 0.5, 0.8, 1e-3, 0.08, 0.14, 0.9, 100.0
W_POS, W_ORI = 1.0, 0.3
APPROACH_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])
DOWN_WORLD = np.array([0.0, 0.0, -1.0])
HOME_TIME = 3.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0]
AXIS_SIGN = np.array([-1.0, 1.0, 1.0])
AXIS_MAP = [0, 2, 1]

GRIP_OPEN, GRIP_CLOSE = 1.4, 0.0
TRIG_ID, TRIG_TRESH, TRIG_TIMEOUT, TRIG_HOLD = 8, 100, 0.25, 0.25
GRIP_ACTION = "/gripper_controller/gripper_cmd"

model = pin.buildModelFromUrdf(URDF)
data = model.createData()
q_lo, q_hi = model.lowerPositionLimit, model.upperPositionLimit
dq_max = model.velocityLimit * VEL_SCALE / RATE
ee = model.getFrameId("End_Effector")


shared = {"q": np.zeros(6), "target": None, "home": None, "engaged": False, "ref": None, "anchor": None, "ready": False, "trig_last": 0.0}
lock = threading.Lock()

def controller():
    vr = openvr.init(openvr.VRApplication_Other)   
    UNIVERSE = openvr.TrackingUniverseRawAndUncalibrated
    PAD = 1 << openvr.k_EButton_SteamVR_Touchpad   
    dev = None
    cur = None
    pad_was = False                                
    try:
        while True:
            poses = vr.getDeviceToAbsoluteTrackingPose(
                UNIVERSE, 0, openvr.k_unMaxTrackedDeviceCount)
            if dev is None or vr.getTrackedDeviceClass(dev) != openvr.TrackedDeviceClass_Controller:
                dev = next((i for i in range(openvr.k_unMaxTrackedDeviceCount)
                            if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller),
                           None)
            if dev is None:
                time.sleep(0.05)
                continue
 
            p = poses[dev]
            if p.bPoseIsValid:
                m = p.mDeviceToAbsoluteTracking
                cur = np.array([m[0][3], m[1][3], m[2][3]]) 
                with lock:
                    if shared["ready"] and shared["engaged"]:
                        d = (cur - shared["ref"])[AXIS_MAP]
                        newp = shared["anchor"] + SCALE * AXIS_SIGN * d
                        off = newp - shared["home"]
                        dd = np.linalg.norm(off)
                        if dd > REACH:
                            newp = shared["home"] + off * (REACH / dd)
                        shared["target"] = newp
 
            res, state = vr.getControllerState(dev)
            if res:
                pad = bool(state.ulButtonPressed & PAD)
                trig = state.rAxis[1].x                     
                if pad and not pad_was:                    
                    with lock:
                        if shared["ready"] and cur is not None:
                            if not shared["engaged"]:
                                shared["ref"] = cur.copy()
                                shared["anchor"] = shared["target"].copy()
                                shared["engaged"] = True
                                print("ENGAGED")
                            else:
                                shared["engaged"] = False
                                print("FROZEN")
                pad_was = pad
                if trig > 0.5:                         
                    with lock:
                        shared["trig_last"] = time.monotonic()
            time.sleep(1.0 / 250.0)
    finally:
        openvr.shutdown()

        
class Bridge(Node): 
    def __init__(self):
        super().__init__("vive_so100_bridge")
        self.phase = "wait"
        self.t_home = None
        self.pub = self.create_publisher(Float64MultiArray, "/arm_controller/commands", 10)
        self.create_subscription(JointState, "/joint_states", self.seed, 10)
        self.create_timer(1.0 / RATE, self.tick)
        self.grip = ActionClient(self, ParallelGripperCommand, GRIP_ACTION)
        self._grip_last = None
        self._held_since = None
        print("Bridge up. Waiting for robot...")

    def send_traj(self, positions):
        msg = Float64MultiArray()
        msg.data = [float(el) for el in positions]
        self.pub.publish(msg)

    def send_grip(self, pos):
        if not self.grip.server_is_ready():
            return False
        goal = ParallelGripperCommand.Goal()
        goal.command.name = ["Gripper"]
        goal.command.position = [float(pos)]
        self.grip.send_goal_async(goal)
        return True

    def seed(self, msg):
        if self.phase != "wait":
            return
        nm = dict(zip(msg.name, msg.position))
        if not all(j in nm for j in ARM):
            return
        self.q_start = np.array([nm[j] for j in ARM])
        self.t_home = self.get_clock().now()
        self.phase = "homing"
        print(f"Homing to zeros over {HOME_TIME}s (controller ignored)...")

    def tick(self):
        if self.phase == "wait":
            return
        if self.phase == "homing":
            el = (self.get_clock().now() - self.t_home).nanoseconds * 1e-9
            a = min(el / HOME_TIME, 1.0)
            q_cmd = (1.0 - a)* self.q_start + a * np.array(HOME_Q)
            self.send_traj(q_cmd)
            if a < 1.0:
                return
            q = np.zeros(6)
            for i in range(5):
                q[i] = HOME_Q[i] 
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            with lock:
                shared["q"] = q
                shared["home"] = data.oMf[ee].translation.copy()
                shared["target"] = shared["home"].copy()
                shared["anchor"] = shared["home"].copy()
                shared["engaged"] = False
                shared["ready"] = True
            self.phase = "teleop"
            print("Teleop ready. Click trackpad to engage.")
            return
        with lock:  
            q = shared["q"].copy()
            tgt = shared["target"].copy()
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        errp = tgt - data.oMf[ee].translation
        n = np.linalg.norm(errp)
        if n > ERR_MAX:
            errp = errp * (ERR_MAX / n)

        R = data.oMf[ee].rotation
        a_world = R @ APPROACH_AXIS_LOCAL
        erro = np.cross(a_world, DOWN_WORLD)
        Jf = pin.computeFrameJacobian(model, data, q, ee, pin.LOCAL_WORLD_ALIGNED)
        Jp, Jw = Jf[:3, :], Jf[3:, :]
        Jt = np.vstack([W_POS * Jp, W_ORI * Jw])
        et = np.concatenate([W_POS * errp, W_ORI * erro])
        v = Jt.T @ solve(Jt @ Jt.T + DAMP * np.eye(6), et)

        dq = np.clip(v * GAIN, -dq_max, dq_max)
        q = pin.integrate(model, q, dq)
        q = np.clip(q, q_lo, q_hi)
        with lock:
            shared["q"] = q
        self.send_traj(q[:5])

        now = time.monotonic()
        with lock:
            trig_last = shared["trig_last"]
        held = (now - trig_last) < TRIG_TIMEOUT
        if not held:
            self._held_since = None
        elif self._held_since is None:
            self._held_since = now
        want = held and (now - self._held_since) >= TRIG_HOLD
        if want != self._grip_last and self.send_grip(GRIP_CLOSE if want else GRIP_OPEN):
            self._grip_last = want

def main():
    threading.Thread(target=controller, daemon=True).start()
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.ok() and rclpy.shutdown()
if __name__ == "__main__":
    main()