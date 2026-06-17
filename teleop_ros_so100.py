import threading, subprocess, shlex
import time
import numpy as np
import pinocchio as pin
from numpy.linalg import solve
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionClient

URDF = "/home/kyrylo/tst/so100.urdf"
SURVIVE = "/home/kyrylo/libsurvive/build/survive-cli --record-stdout --v 0"
ARM = ['Shoulder_Rotation', 'Shoulder_Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll']

SCALE, GAIN, DAMP, ERR_MAX, REACH, VEL_SCALE, RATE = 0.5, 0.8, 1e-3, 0.08, 0.14, 0.9, 100.0
W_POS, W_ORI = 1.0, 0.3
APPROACH_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])
DOWN_WORLD = np.array([0.0, 0.0, -1.0])
HOME_TIME = 3.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0]
AXIS_SIGN = np.array([-1.0,-1.0,1.0])

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
    p = subprocess.Popen(shlex.split(SURVIVE), stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    cur = None
    for line in p.stdout:
        t = line.split()
        if len(t) < 3 or t[1] != "WW0":
            continue
        if t[2] == "POSE" and len(t) >= 6:
            cur = np.array(list(map(float, t[3:6])))
            with lock:
                if shared["ready"] and shared["engaged"]:
                    newp = shared["anchor"] + SCALE * AXIS_SIGN * (cur - shared["ref"])
                    off = newp - shared["home"]
                    dd = np.linalg.norm(off)
                    if dd > REACH:
                        newp = shared["home"] + off * (REACH / dd)
                    shared["target"] = newp
        elif t[2] == "BUTTON" and len(t) >= 5 and int(t[3]) == 3 and int(t[4]) == 1:
            with lock:
                if not shared["ready"] or cur is None:
                    continue
                if not shared["engaged"]:
                    shared["ref"] = cur.copy()
                    shared["anchor"] = shared["target"].copy()
                    shared["engaged"] = True
                    print("ENGAGED")
                else:
                    shared["engaged"] = False
                    print("FROZEN")
        elif t[2] == "BUTTON" and len(t) >= 5 and int(t[3]) == TRIG_ID and float(t[4]) > TRIG_TRESH:
            with lock:
                shared["trig_last"] = time.monotonic()
        
class Bridge(Node): 
    def __init__(self):
        super().__init__("vive_so100_bridge")
        self.phase = "wait"
        self.t_home = None
        self.pub = self.create_publisher(JointTrajectory, "/arm_controller/joint_trajectory", 10)
        self.create_subscription(JointState, "/joint_states", self.seed, 10)
        self.create_timer(1.0 / RATE, self.tick)
        self.grip = ActionClient(self, ParallelGripperCommand, GRIP_ACTION)
        self._grip_last = None
        self._held_since = None
        print("Bridge up. Waiting for robot...")

    def send_traj(self, positions, sec=0, nanosec=int(1e9 / RATE)):
        msg = JointTrajectory()
        msg.joint_names = ARM
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in positions]
        pt.time_from_start = Duration(sec=int(sec), nanosec=int(nanosec))
        msg.points = [pt]
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
        self.send_traj(HOME_Q, sec=int(HOME_TIME), nanosec=0)
        self.t_home = self.get_clock().now()
        self.phase = "homing"
        print(f"Homing to zeros over {HOME_TIME}s (controller ignored)...")

    def tick(self):
        if self.phase == "wait":
            return
        if self.phase == "homing":
            if (self.get_clock().now() - self.t_home).nanoseconds * 1e-9 < HOME_TIME:
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