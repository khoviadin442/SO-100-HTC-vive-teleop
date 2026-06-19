import threading
import openvr
import time
import numpy as np
import pinocchio as pin
from numpy.linalg import solve, svd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray

URDF = "/home/kyrylo/tst/so100.urdf"
ARM = ['Shoulder_Rotation', 'Shoulder_Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll']

SCALE, VEL_SCALE, RATE = 0.8, 0.9, 100.0
NUM_IK = 3
CO_LP = 0.0

SING_EPS, DAMP_MAX, DAMP_FLOOR = 0.05, 0.03, 1e-3
IK_ITERS, IK_TOL, IK_STEP = 80, 1e-4, 0.3

APPROACH_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])
PITCH_AXIS_LOCAL    = np.array([0.0, 0.0, 1.0])
WRIST_ROLL_SIGN, WRIST_FLEX_SIGN = 1.0, 1.0

REACH_FRAC = 0.97

AZ_GAIN = 2.0

HOME_TIME = 3.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0]

AXIS_SIGN = np.array([-1.0, 1.0, 1.0])
AXIS_MAP = [0, 2, 1]
M = np.zeros((3, 3))
for i in range(3):
    M[i, AXIS_MAP[i]] = AXIS_SIGN[i]

GRIP_OPEN, GRIP_CLOSE = 1.4, 0.0
TRIG_TIMEOUT, TRIG_HOLD = 0.15, 0.15
GRIP_ACTION = "/gripper_controller/gripper_cmd"

model = pin.buildModelFromUrdf(URDF)
data = model.createData()
q_lo, q_hi = model.lowerPositionLimit, model.upperPositionLimit
dq_max = model.velocityLimit * VEL_SCALE / RATE
ee = model.getFrameId("End_Effector")
wr = model.joints[model.getJointId("Wrist_Roll")].idx_q
i_wf = model.joints[model.getJointId("Wrist_Pitch")].idx_q
i_wr = wr
v_wf = model.joints[model.getJointId("Wrist_Pitch")].idx_v
v_wr = model.joints[model.getJointId("Wrist_Roll")].idx_v

pin.forwardKinematics(model, data, np.zeros(model.nq))
pin.updateFramePlacements(model, data)
SHOULDER = data.oMi[model.getJointId("Shoulder_Pitch")].translation.copy()


def compute_max_reach(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    lo_d, hi_d = 1e9,0.0
    q = np.zeros(model.nq)
    for s in rng.uniform(q_lo[:NUM_IK], q_hi[:NUM_IK],size = (n,NUM_IK)):
        q[:NUM_IK] = s
        pin.forwardKinematics(model,data,q)
        pin.updateFramePlacements(model,data)
        d = np.linalg.norm(data.oMf[ee].translation - SHOULDER)
        lo_d = min(lo_d, d)
        hi_d = max(hi_d, d)
    return lo_d, hi_d

MIN_REACH, MAX_REACH = compute_max_reach()
R_MIN = MIN_REACH + 0.2 * (MAX_REACH - MIN_REACH)
R_MAX = MIN_REACH + 0.85 * (MAX_REACH - MIN_REACH)

shared = {"q": np.zeros(6), "target": None, "home": None, "engaged": False, "ref": None,
          "anchor": None, "ready": False, "trig_last": 0.0,
          "Rc": np.eye(3), "q_meas": np.zeros(6)}
lock = threading.Lock()


def solve_ik_pos(target, q_seed):
    q = np.asarray(q_seed, dtype=float).copy()
    sm = 1.0
    for _ in range(IK_ITERS):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        err = target - data.oMf[ee].translation
        if np.linalg.norm(err) < IK_TOL:
            break
        J3 = pin.computeFrameJacobian(model, data, q, ee, pin.LOCAL_WORLD_ALIGNED)[:3, :NUM_IK]
        sm = svd(J3, compute_uv=False)[-1]
        damp = DAMP_FLOOR if sm >= SING_EPS else DAMP_FLOOR + (1.0 - (sm / SING_EPS) ** 2) * DAMP_MAX
        dq3 = J3.T @ solve(J3 @ J3.T + damp * np.eye(3), err)
        q[:NUM_IK] = np.clip(q[:NUM_IK] + np.clip(dq3, -IK_STEP, IK_STEP), q_lo[:NUM_IK], q_hi[:NUM_IK])
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return q[:NUM_IK].copy(), float(np.linalg.norm(target - data.oMf[ee].translation)), sm

def solve_wrist_ori(q_seed, R_des, iters = 20, step=0.3):
    q = np.asarray(q_seed,float).copy()
    for _ in range(iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model,data)
        e = pin.log3(R_des @ data.oMf[ee].rotation.T)
        if np.linalg.norm(e) < 1e-4:
            break
        J = pin.computeFrameJacobian(model,data,q,ee, pin.LOCAL_WORLD_ALIGNED)[3:6]
        Jw = J[:, [v_wf, v_wr]]
        dw = np.linalg.lstsq(Jw, e, rcond=None)[0]
        q[i_wf] = np.clip(q[i_wf]+ np.clip(dw[0], -step, step), q_lo[i_wf], q_hi[i_wf])
        q[i_wr] = np.clip(q[i_wr] + np.clip(dw[1], -step, step), q_lo[i_wr], q_hi[i_wr])
    return float(q[i_wf]),float(q[i_wr])

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
                mm = p.mDeviceToAbsoluteTracking
                cur = np.array([mm[0][3], mm[1][3], mm[2][3]])
                Rc = np.array([[mm[0][0], mm[0][1], mm[0][2]],
                               [mm[1][0], mm[1][1], mm[1][2]],
                               [mm[2][0], mm[2][1], mm[2][2]]])
                with lock:
                    shared["Rc"] = Rc
                    if shared["ready"] and shared["engaged"]:
                        d = (cur - shared["ref"])[AXIS_MAP]
                        newp = shared["anchor"] + SCALE * AXIS_SIGN * d
                        arel = shared["anchor"] - SHOULDER
                        rel0 = newp - SHOULDER
                        az0 = np.arctan2(arel[1],arel[0])
                        az = np.arctan2(rel0[1],rel0[0])
                        daz = (az - az0 + np.pi)% (2*np.pi) - np.pi
                        naz = az0 + AZ_GAIN * daz
                        rh = np.hypot(rel0[0],rel0[1])
                        newp = SHOULDER + np.array([rh*np.cos(naz), rh * np.sin(naz), rel0[2]])
                        rel = newp - SHOULDER
                        rr = np.linalg.norm(rel)
                        if rr > R_MAX:
                            newp = SHOULDER + rel * (R_MAX / rr)
                        elif rr < R_MIN:
                            newp = SHOULDER + rel * (R_MIN / rr)
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
        self._was_engaged = False
        self.Rc_ref = None
        self.R_anchor = None
        self.wrist_flex0 = 0.0
        self.wrist_roll0 = 0.0
        self.q_cmd = None
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
        nm = dict(zip(msg.name, msg.position))
        if not all(j in nm for j in ARM):
            return
        qm = np.zeros(model.nq)
        qm[:5] = [nm[j] for j in ARM]
        if "Gripper" in nm:
            qm[5] = nm["Gripper"]
        with lock:
            shared["q_meas"] = qm
        if self.phase == "wait":
            self.q_start = qm[:5].copy()
            self.t_home = self.get_clock().now()
            self.phase = "homing"
            print(f"Homing to zeros over {HOME_TIME}s (controller ignored)...")

    def tick(self):
        if self.phase == "wait":
            return

        if self.phase == "homing":
            el = (self.get_clock().now() - self.t_home).nanoseconds * 1e-9
            a = min(el / HOME_TIME, 1.0)
            q_cmd = (1.0 - a) * self.q_start + a * np.array(HOME_Q)
            self.send_traj(q_cmd)
            if a < 1.0:
                return
            q = np.zeros(6)
            for i in range(5):
                q[i] = HOME_Q[i]
            self.q_cmd = q.copy()
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
            q_meas = shared["q_meas"].copy()
            tgt = shared["target"].copy()
        q = (1 - CO_LP) * self.q_cmd + CO_LP * q_meas
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        with lock:
            Rc = shared["Rc"].copy()
            engaged = shared["engaged"]
        if engaged and not self._was_engaged:
            self.Rc_ref = Rc.copy()
            self.R_anchor = data.oMf[ee].rotation.copy()
            self.wrist_flex0 = float(q[i_wf])
            self.wrist_roll0 = float(q[i_wr])
        self._was_engaged = engaged

        ik3,pos_err,sm = solve_ik_pos(tgt,q)
        q[:NUM_IK] = ik3
        if engaged and self.Rc_ref is not None:
            dR = M @ (Rc @ self.Rc_ref.T) @ M.T
            R_des = dR @ self.R_anchor
            wf, wrr = solve_wrist_ori(q, R_des)
        else:
            wf, wrr = float(q[i_wf]), float(q[i_wr])
        q[i_wf] = wf
        q[i_wr] = wrr
        dq = np.clip(q - self.q_cmd, -dq_max, dq_max)
        q = np.clip(self.q_cmd + dq, q_lo, q_hi)
        self.q_cmd = q

        margin = np.minimum(q[:5] - q_lo[:5], q_hi[:5] - q[:5])
        print(f"sm={sm:.3f} err={pos_err * 1000:.1f}mm tgtz={tgt[2]:+.3f} "
              f"wf={np.degrees(wf):+.0f} wr={np.degrees(wrr):+.0f} "
              f"minmargin={margin.min():.2f}@{int(margin.argmin())}")

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