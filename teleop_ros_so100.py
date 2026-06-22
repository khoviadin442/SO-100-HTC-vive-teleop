import os
import threading
import openvr
import time
import numpy as np
import pinocchio as pin
import qpsolvers
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray
from pink import Configuration, solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.limits import ConfigurationLimit, VelocityLimit
from ament_index_python.packages import get_package_share_directory

URDF = "/home/kyrylo/tst/so100.urdf"
EE_FRAME = "End_Effector"
ARM = ['Shoulder_Rotation', 'Shoulder_Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll']
GRIPPER_JOINT = "Gripper"

RATE = 100.0
DT = 1.0 / RATE

POSITION_COST, ORIENTATION_COST, LM_DAMPING = 1.0, 0.15, 1e-3
TASK_GAIN, POSTURE_COST, VEL_SCALE = 0.8, 1e-2, 0.9

SCALE, AZ_GAIN = 0.6, 2.0
REACH_LO_FRAC, REACH_HI_FRAC = 0.20, 0.85

AXIS_SIGN = np.array([-1.0, 1.0, 1.0])
AXIS_MAP = [0, 2, 1]
M = np.zeros((3, 3))
for i in range(3):
    M[i, AXIS_MAP[i]] = AXIS_SIGN[i]

HOME_TIME = 3.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0]

GRIP_OPEN, GRIP_CLOSE = 1.4, 0.0
TRIG_TIMEOUT, TRIG_HOLD = 0.15, 0.15
GRIP_ACTION = "/gripper_controller/gripper_cmd"
ARM_CMD_TOPIC = "/arm_controller/commands"
JOINT_STATES_TOPIC = "/joint_states"

COLLISION_MARGIN = 0.01

RLIMITS = {"Wrist_Pitch": (-0.7, 0.7)}

def mesh_pkg_dirs():
    return [os.path.dirname(get_package_share_directory('so_arm_100_description'))]

def srdf_path():
    return os.path.join(get_package_share_directory('so_arm_100_moveit_config'), 'config', 'so_arm_100.srdf')

class PinkIK:
    def __init__(self, urdf_path, ee_frame, arm_joints, gripper_joint=None, position_cost=POSITION_COST, orientation_cost=ORIENTATION_COST,lm_damping=LM_DAMPING, gain=TASK_GAIN, posture_cost=POSTURE_COST, vel_scale=VEL_SCALE, solver=None, srdf_path=None, package_dirs=None, collision_margin=COLLISION_MARGIN):
        full = pin.buildModelFromUrdf(urdf_path)
        locked = []
        if gripper_joint and full.existJointName(gripper_joint):
            locked = [full.getJointId(gripper_joint)]
        self.geom = None
        self.blocked = False
        if srdf_path and package_dirs:
            geom_full = pin.buildGeomFromUrdf(full, urdf_path, pin.GeometryType.COLLISION, package_dirs=list(package_dirs))
            if locked:
                self.model, self.geom = pin.buildReducedModel(full, geom_full, locked, pin.neutral(full))
            else:
                self.model,self.geom = full, geom_full
            self.geom.addAllCollisionPairs()
            pin.removeCollisionPairs(self.model, self.geom, srdf_path, False)
            self.geom_data = self.geom.createData()
            for req in self.geom_data.collisionRequests:
                req.security_margin = float(collision_margin)
            self.col_data = self.model.createData()
        else:
            self.model = pin.buildReducedModel(full, locked, pin.neutral(full)) if locked else full 
        self.data = self.model.createData()
        if not self.model.existFrame(ee_frame):
            raise ValueError(f"EE frame '{ee_frame}' not found in URDF")
        self.ee = ee_frame
        self.arm_joints = list(arm_joints)
        self._qidx = {j: self.model.joints[self.model.getJointId(j)].idx_q for j in self.arm_joints}
        self.fix_limits(vel_scale)
        self.solver = solver or ("daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0])
        self.ee_task = FrameTask(ee_frame, position_cost=position_cost, orientation_cost=orientation_cost,lm_damping=lm_damping, gain=gain)
        self.posture = PostureTask(cost=posture_cost)
        self.configuration = Configuration(self.model, self.data, pin.neutral(self.model))
        self.posture.set_target(self.configuration.q)
        self.limits = [ConfigurationLimit(self.model), VelocityLimit(self.model)]

    def fix_limits(self, vel_scale):
        lo = np.array(self.model.lowerPositionLimit, float)
        hi = np.array(self.model.upperPositionLimit, float)
        lo[~np.isfinite(lo)] = -np.pi
        hi[~np.isfinite(hi)] = np.pi
        for j, (jlo,jhi) in RLIMITS.items():
            if j in self._qidx:
                qi = self._qidx[j]
                lo[qi] = max(lo[qi], float(jlo))
                hi[qi] = min(hi[qi], float(jhi))
        self.model.lowerPositionLimit, self.model.upperPositionLimit = lo, hi
        vl = np.array(self.model.velocityLimit, float)
        vl[~np.isfinite(vl) | (vl <= 0)] = np.pi
        self.model.velocityLimit = vl * float(vel_scale)

    def in_collision(self,q):
        if self.geom is None:
            return False
        return bool(pin.computeCollisions(self.model, self.col_data, self.geom, self.geom_data, np.asarray(q, float), True))

    def qindex(self, joint_name):
        return self._qidx[joint_name]

    def neutral(self):
        return pin.neutral(self.model)

    @property
    def q(self):
        return self.configuration.q.copy()

    def arm_positions(self):
        q = self.configuration.q
        return np.array([q[self._qidx[j]] for j in self.arm_joints], float)

    def fk_rotation(self):
        return self.configuration.get_transform_frame_to_world(self.ee).rotation.copy()

    def fk_translation(self):
        return self.configuration.get_transform_frame_to_world(self.ee).translation.copy()

    def reset_to(self, qf):
        self.configuration = Configuration(self.model, self.data, np.asarray(qf, float))
        self.posture.set_target(self.configuration.q)

    def step(self, target_pos, target_R, dt=DT):
        T = pin.SE3(np.asarray(target_R, float), np.asarray(target_pos, float))
        self.ee_task.set_target(T)
        q_prec = self.configuration.q.copy()
        q_new = q_prec
        try:
            v = solve_ik(self.configuration, [self.ee_task, self.posture], dt, solver=self.solver, limits=self.limits, safety_break=False)
            q_new = pin.integrate(self.model, q_prec, v*dt)
        except Exception as exc:
            print(f"[ik] solve skipped: {exc}")
        q_new = np.clip(q_new, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        self.blocked = self.in_collision(q_new)
        if self.blocked:
            q_new = q_prec
        if not np.array_equal(q_new, self.configuration.q):
            self.configuration = Configuration(self.model, self.data, q_new)
        return self.arm_positions()


ik = PinkIK(URDF, EE_FRAME, ARM, GRIPPER_JOINT, srdf_path = srdf_path(), package_dirs = mesh_pkg_dirs())
_model = ik.model


def shoulder_origin():
    q0 = ik.neutral()
    fk_model, fk_data = _model, _model.createData()
    pin.forwardKinematics(fk_model, fk_data, q0)
    pin.updateFramePlacements(fk_model, fk_data)
    jid = fk_model.getJointId("Shoulder_Pitch")
    return fk_data.oMi[jid].translation.copy()


SHOULDER = shoulder_origin()


def reach_shell(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    lo_lim = _model.lowerPositionLimit
    hi_lim = _model.upperPositionLimit
    qidx = [ik.qindex(j) for j in ARM[:3]]
    q = ik.neutral()
    lo_d, hi_d = 1e9, 0.0
    samples = rng.uniform([lo_lim[i] for i in qidx], [hi_lim[i] for i in qidx], size=(n, 3))
    fk_data = _model.createData()
    for s in samples:
        for k, i in enumerate(qidx):
            q[i] = s[k]
        pin.forwardKinematics(_model, fk_data, q)
        pin.updateFramePlacements(_model, fk_data)
        d = np.linalg.norm(fk_data.oMf[_model.getFrameId(EE_FRAME)].translation - SHOULDER)
        lo_d = min(lo_d, d)
        hi_d = max(hi_d, d)
    return lo_d, hi_d


MIN_REACH, MAX_REACH = reach_shell()
R_MIN = MIN_REACH + REACH_LO_FRAC * (MAX_REACH - MIN_REACH)
R_MAX = MIN_REACH + REACH_HI_FRAC * (MAX_REACH - MIN_REACH)

shared = {"target": None, "home": None, "anchor": None, "ref": None,
          "engaged": False, "ready": False, "trig_last": 0.0, "Rc": np.eye(3)}
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
            poses = vr.getDeviceToAbsoluteTrackingPose(UNIVERSE, 0, openvr.k_unMaxTrackedDeviceCount)
            if dev is None or vr.getTrackedDeviceClass(dev) != openvr.TrackedDeviceClass_Controller:
                dev = next((i for i in range(openvr.k_unMaxTrackedDeviceCount) if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller), None)
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
                        az0 = np.arctan2(arel[1], arel[0])
                        az = np.arctan2(rel0[1], rel0[0])
                        daz = (az - az0 + np.pi) % (2 * np.pi) - np.pi
                        naz = az0 + AZ_GAIN * daz
                        rh = np.hypot(rel0[0], rel0[1])
                        newp = SHOULDER + np.array([rh * np.cos(naz), rh * np.sin(naz), rel0[2]])
                        rel = newp - SHOULDER
                        r = np.linalg.norm(rel)
                        if r < 1e-6:
                            newp = shared["target"]
                        elif r > R_MAX:
                            newp = SHOULDER + rel * (R_MAX / r)
                        elif r < R_MIN:
                            newp = SHOULDER + rel * (R_MIN / r)
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
        super().__init__("vive_so100_pink_bridge")
        self.phase = "wait"
        self.t_home = None
        self.q_start = None
        self.pub = self.create_publisher(Float64MultiArray, ARM_CMD_TOPIC, 10)
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self.on_joint_states, 10)
        self.create_timer(DT, self.tick)
        self.grip = ActionClient(self, ParallelGripperCommand, GRIP_ACTION)
        self._grip_last = None
        self._held_since = None
        self._was_engaged = False
        self.Rc_ref = None
        self.R_anchor = None
        self._blk =  0
        self.pos = 0
        self.eff = 0
        self.dbg = 0
        print("Bridge up. Waiting for robot...")

    def send_arm(self, positions):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in positions]
        self.pub.publish(msg)

    def send_grip(self, pos):
        if not self.grip.server_is_ready():
            return False
        goal = ParallelGripperCommand.Goal()
        goal.command.name = [GRIPPER_JOINT]
        goal.command.position = [float(pos)]
        self.grip.send_goal_async(goal)
        return True

    def on_joint_states(self, msg):
        nm = dict(zip(msg.name, msg.position))
        self.pos = nm
        self.eff = dict(zip(msg.name, msg.effort)) if msg.effort else{}
        if not all(j in nm for j in ARM):
            return
        if self.phase == "wait":
            self.q_start = np.array([nm[j] for j in ARM], float)
            self.t_home = self.get_clock().now()
            self.phase = "homing"
            print(f"Homing to {HOME_Q} over {HOME_TIME}s (controller ignored)...")

    def tick(self):
        if self.phase == "wait":
            return

        if self.phase == "homing":
            el = (self.get_clock().now() - self.t_home).nanoseconds * 1e-9
            a = min(el / HOME_TIME, 1.0)
            q_cmd = (1.0 - a) * self.q_start + a * np.array(HOME_Q, float)
            self.send_arm(q_cmd)
            if a < 1.0:
                return
            q_home = ik.neutral()
            for name, val in zip(ARM, HOME_Q):
                q_home[ik.qindex(name)] = val
            ik.reset_to(q_home)
            with lock:
                shared["home"] = ik.fk_translation()
                shared["target"] = shared["home"].copy()
                shared["anchor"] = shared["home"].copy()
                shared["engaged"] = False
                shared["ready"] = True
            self.phase = "teleop"
            print("Teleop ready. Click trackpad to engage.")
            return

        with lock:
            tgt = shared["target"].copy()
            Rc = shared["Rc"].copy()
            engaged = shared["engaged"]

        if engaged and not self._was_engaged:
            self.Rc_ref = Rc.copy()
            self.R_anchor = ik.fk_rotation()
        self._was_engaged = engaged

        if engaged and self.Rc_ref is not None:
            dR = M @ (Rc @ self.Rc_ref.T) @ M.T
            R_des = dR @ self.R_anchor
        else:
            R_des = ik.fk_rotation()

        q_arm = ik.step(tgt, R_des, DT)
        self.dbg += 1
        if self.dbg % 50 == 0:
            print("blocked=%s" % ik.blocked)
            for i,j in enumerate(ARM):
                cmd = float(q_arm[i])
                meas = self.pos.get(j)
                eff = self.eff.get(j)
                ms = "n/a" if meas is None else "%+.4f" % meas
                gs = "n/a" if meas is None else "%+.4f" % (cmd - meas)
                es = "n/a" if eff is None else "%+.3f" % eff
                print("%-16s cmd=%+.4f meas=%s gap=%s eff=%s" % (j, cmd, ms, gs, es))
        if ik.blocked:
            self._blk += 1
            if self._blk % 50 == 1:
                print("collision == block")
        else:
            self._blk = 0
        self.send_arm(q_arm)

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