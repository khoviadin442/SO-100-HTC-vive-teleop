import os
import yaml
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

path = os.environ.get("teleop_config", os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_teleop.yaml"))
with open(path) as f:
    CFG = yaml.safe_load(f)

URDF = CFG["urdf"]
EE_FRAME = CFG["ee_frame"]
ARM = list(CFG["arm"])
GRIPPER_JOINT = CFG["gripper_joint"]

RATE = float(CFG["rate"])
DT = 1.0 / RATE

POSITION_COST = float(CFG["ik"]["position_cost"])
ORIENTATION_COST = float(CFG["ik"]["orientation_cost"])
LM_DAMPING = float(CFG["ik"]["lm_damping"])
TASK_GAIN = float(CFG["ik"]["task_gain"])
POSTURE_COST = float(CFG["ik"]["posture_cost"])
VEL_SCALE = float(CFG["ik"]["vel_scale"])
COLLISION_MARGIN = float(CFG["ik"]["collision_margin"])

SCALE = float(CFG["teleop"]["scale"])
AZ_GAIN = float(CFG["teleop"]["az_gain"])
REACH_LO_FRAC = float(CFG["teleop"]["reach_lo_frac"])
REACH_HI_FRAC = float(CFG["teleop"]["reach_hi_frac"])

AXIS_SIGN = np.array(CFG["teleop"]["axis_sign"])
AXIS_MAP = list(CFG["teleop"]["axis_map"])
M = np.zeros((3, 3))
for i in range(3):
    M[i, AXIS_MAP[i]] = AXIS_SIGN[i]

HOME_TIME = float(CFG["home"]["time"])
HOME_Q = list(CFG["home"]["q"])

GRIP_OPEN = float(CFG["gripper"]["grip_open"])
GRIP_CLOSE = float(CFG["gripper"]["grip_close"])
TRIG_TIMEOUT = float(CFG["gripper"]["trig_timeout"])
TRIG_HOLD = float(CFG["gripper"]["trig_hold"])
GRIP_ACTION = CFG["gripper"]["action"]

ARM_CMD_TOPIC = CFG["topics"]["arm_cmd"]
JOINT_STATES_TOPIC = CFG["topics"]["joint_states"]

RLIMITS = {k: tuple(v) for k,v in CFG["rlimits"].items()}

def mesh_pkg_dirs():
    """Package dirs used to resolve mesh paths referenced by the URDF."""
    return [os.path.dirname(get_package_share_directory('so_arm_100_description'))]

def srdf_path():
    """Path to the MoveIt SRDF, used to disable allowed collision pairs."""
    return os.path.join(get_package_share_directory('so_arm_100_moveit_config'), 'config', 'so_arm_100.srdf')

class PinkIK:
    """Pinocchio model + Pink differential IK for the SO-100 arm with self-collision checking."""
    def __init__(self, urdf_path, ee_frame, arm_joints, gripper_joint=None, position_cost=POSITION_COST, orientation_cost=ORIENTATION_COST,lm_damping=LM_DAMPING, gain=TASK_GAIN, posture_cost=POSTURE_COST, vel_scale=VEL_SCALE, solver=None, srdf_path=None, package_dirs=None, collision_margin=COLLISION_MARGIN):
        """Build the gripper-locked model and collision geometry, set up frame/posture tasks, 
        joint limits and the QP solver."""
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
        """Replace non-finite position/velocity limits, apply extra per-joint clamps from RLIMITS, 
        and scale velocity limits."""
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
        """Return True if configuration q is in self-collision (False when no geometry is loaded)."""
        if self.geom is None:
            return False
        return bool(pin.computeCollisions(self.model, self.col_data, self.geom, self.geom_data, np.asarray(q, float), True))

    def qindex(self, joint_name):
        """Index of a named joint inside the configuration vector q."""
        return self._qidx[joint_name]

    def neutral(self):
        """Model neutral configuration."""
        return pin.neutral(self.model)

    @property
    def q(self):
        """Current configuration q (copy)."""
        return self.configuration.q.copy()

    def arm_positions(self):
        """Current positions of the arm joints, in ARM order."""
        q = self.configuration.q
        return np.array([q[self._qidx[j]] for j in self.arm_joints], float)

    def fk_rotation(self):
        """EE frame rotation in world for the current configuration."""
        return self.configuration.get_transform_frame_to_world(self.ee).rotation.copy()

    def fk_translation(self):
        """EE frame position in world for the current configuration."""
        return self.configuration.get_transform_frame_to_world(self.ee).translation.copy()

    def reset_to(self, qf):
        """Reset the configuration to qf and re-anchor the posture target there."""
        self.configuration = Configuration(self.model, self.data, np.asarray(qf, float))
        self.posture.set_target(self.configuration.q)

    def step(self, target_pos, target_R, dt=DT):
        """One diff-IK step toward (target_pos, target_R): solve, clamp to limits, 
        reject on collision, return new arm positions."""
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

class Bridge(Node):
    """ROS2 node: HTC Vive controller -> Pink diff-IK -> SO-100 arm and gripper."""
    def __init__(self):
        """Build IK, compute shoulder origin and reach-shell radii, 
        set up publishers/subscribers/timers and shared teleop state."""
        super().__init__("vive_so100_pink_bridge")
        self.ik = PinkIK(URDF, EE_FRAME, ARM, GRIPPER_JOINT, srdf_path = srdf_path(), package_dirs = mesh_pkg_dirs())
        self.model = self.ik.model
        self.shoulder = self.shoulder_origin()
        mn,mx = self.reach_shell()
        self.r_min = mn + REACH_LO_FRAC * (mx - mn)
        self.r_max = mn + REACH_HI_FRAC * (mx - mn)
        self.phase = "wait"
        self.t_home = None
        self.q_start = None
        self.pub = self.create_publisher(Float64MultiArray, ARM_CMD_TOPIC, 10)
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self.on_joint_states, 10)
        self.create_timer(DT, self.tick)
        self.create_timer(1.0/250.0, self.vr_tick)
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
        self.shared = {"target": None, "home": None, "anchor": None, "ref": None, "engaged": False, "ready": False, "trig_last": 0.0, "Rc": np.eye(3)}
        self._vr = None
        self._dev = None
        self._cur = None
        self._pad_was = False
        self._menu_was = False
        self.mark = False
        self.get_logger().info("Bridge up. Waiting for robot...")

    def shoulder_origin(self):
        """World position of the Shoulder_Pitch joint at neutral, used as the teleop workspace center."""
        q0 = self.ik.neutral()
        fk_data = self.model.createData()
        pin.forwardKinematics(self.model, fk_data, q0)
        pin.updateFramePlacements(self.model, fk_data)
        jid = self.model.getJointId("Shoulder_Pitch")
        return fk_data.oMi[jid].translation.copy()
    
    def reach_shell(self, n=60000, seed=0):
        """Monte-Carlo sample the first three joints to estimate min/max EE distance 
        from the shoulder (reach envelope)."""
        rng = np.random.default_rng(seed)
        lo_lim = self.model.lowerPositionLimit
        hi_lim = self.model.upperPositionLimit
        qidx = [self.ik.qindex(j) for j in ARM[:3]]
        q = self.ik.neutral()
        lo_d, hi_d = 1e9, 0.0
        samples = rng.uniform([lo_lim[i] for i in qidx], [hi_lim[i] for i in qidx], size=(n, 3))
        fk_data = self.model.createData()
        for s in samples:
            for k, i in enumerate(qidx):
                q[i] = s[k]
            pin.forwardKinematics(self.model, fk_data, q)
            pin.updateFramePlacements(self.model, fk_data)
            d = np.linalg.norm(fk_data.oMf[self.model.getFrameId(EE_FRAME)].translation - self.shoulder)
            lo_d = min(lo_d, d)
            hi_d = max(hi_d, d)
        return lo_d, hi_d

    def vr_tick(self):
        """Poll the VR controller: move the target inside the reach shell while engaged, 
        toggle clutch on trackpad, mark on menu, latch trigger for the gripper."""

        if self._vr is None:
            try:
                self._vr = openvr.init(openvr.VRApplication_Other)
            except Exception as exc:
                self.get_logger().info(f"Waiting for vr: {exc}", throttle_duration_sec = 2.0)
                return
        vr = self._vr
        UNIVERSE = openvr.TrackingUniverseRawAndUncalibrated
        PAD = 1 << openvr.k_EButton_SteamVR_Touchpad
        MENU = 1 << openvr.k_EButton_ApplicationMenu
        poses = vr.getDeviceToAbsoluteTrackingPose(UNIVERSE, 0, openvr.k_unMaxTrackedDeviceCount)
        if self._dev is None or vr.getTrackedDeviceClass(self._dev) != openvr.TrackedDeviceClass_Controller:
            self._dev = next((i for i in range(openvr.k_unMaxTrackedDeviceCount) if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller), None)
        if self._dev is None:
            return
        p = poses[self._dev]
        if p.bPoseIsValid:
            mm = p.mDeviceToAbsoluteTracking
            self._cur = np.array([mm[0][3], mm[1][3], mm[2][3]])
            Rc = np.array([[mm[0][0], mm[0][1], mm[0][2]],
                           [mm[1][0], mm[1][1], mm[1][2]],
                           [mm[2][0], mm[2][1], mm[2][2]]])
            self.shared["Rc"] = Rc
            if self.shared["ready"] and self.shared["engaged"]:
                d = (self._cur - self.shared["ref"])[AXIS_MAP]
                newp = self.shared["anchor"] + SCALE * AXIS_SIGN * d
                arel = self.shared["anchor"] - self.shoulder
                rel0 = newp - self.shoulder
                az0 = np.arctan2(arel[1], arel[0])
                az = np.arctan2(rel0[1], rel0[0])
                daz = (az - az0 + np.pi) % (2 * np.pi) - np.pi
                naz = az0 + AZ_GAIN * daz
                rh = np.hypot(rel0[0], rel0[1])
                newp = self.shoulder + np.array([rh * np.cos(naz), rh * np.sin(naz), rel0[2]])
                rel = newp - self.shoulder
                r = np.linalg.norm(rel)
                if r < 1e-6:
                    newp = self.shared["target"]
                elif r > self.r_max:
                    newp = self.shoulder + rel * (self.r_max / r)
                elif r < self.r_min:
                    newp = self.shoulder + rel * (self.r_min / r)
                self.shared["target"] = newp
        res, state = vr.getControllerState(self._dev)
        if res:
            menu = bool(state.ulButtonPressed & MENU)
            pad = bool(state.ulButtonPressed & PAD)
            trig = state.rAxis[1].x
            if pad and not self._pad_was:
                if self.shared["ready"] and self._cur is not None:
                    if not self.shared["engaged"]:
                        self.shared["ref"] = self._cur.copy()
                        self.shared["anchor"] = self.shared["target"].copy()
                        self.shared["engaged"] = True
                        self.get_logger().info("ENGAGED")
                    else:
                        self.shared["engaged"] = False
                        self.get_logger().info("FROZEN")
            self._pad_was = pad
            if menu and not self._menu_was:
                self.mark = not self.mark
                self.get_logger().info(f"RECORD {'ON' if self.mark else 'OFF'}")
            self._menu_was = menu
            if trig > 0.5:
                self.shared["trig_last"] = time.monotonic()
        


    def send_arm(self, positions):
        """Publish arm joint positions to the arm controller."""
        msg = Float64MultiArray()
        msg.data = [float(x) for x in positions]
        self.pub.publish(msg)

    def send_grip(self, pos):
        """Send a gripper position goal via the action client; no-op if the server is not ready."""
        if not self.grip.server_is_ready():
            return False
        goal = ParallelGripperCommand.Goal()
        goal.command.name = [GRIPPER_JOINT]
        goal.command.position = [float(pos)]
        self.grip.send_goal_async(goal)
        return True

    def on_joint_states(self, msg):
        """Cache measured joint positions/efforts and start homing once all arm joints are known."""
        nm = dict(zip(msg.name, msg.position))
        self.pos = nm
        self.eff = dict(zip(msg.name, msg.effort)) if msg.effort else{}
        if not all(j in nm for j in ARM):
            return
        if self.phase == "wait":
            self.q_start = np.array([nm[j] for j in ARM], float)
            self.t_home = self.get_clock().now()
            self.phase = "homing"
            self.get_logger().info(f"Homing to {HOME_Q} over {HOME_TIME}s (controller ignored)...")

    def tick(self):
        """Main 100 Hz loop: home the arm, then run diff-IK toward the VR target and drive arm and gripper."""
        if self.phase == "wait":
            return

        if self.phase == "homing":
            el = (self.get_clock().now() - self.t_home).nanoseconds * 1e-9
            a = min(el / HOME_TIME, 1.0)
            q_cmd = (1.0 - a) * self.q_start + a * np.array(HOME_Q, float)
            self.send_arm(q_cmd)
            if a < 1.0:
                return
            q_home = self.ik.neutral()
            for name, val in zip(ARM, HOME_Q):
                q_home[self.ik.qindex(name)] = val
            self.ik.reset_to(q_home)
            self.shared["home"] = self.ik.fk_translation()
            self.shared["target"] = self.shared["home"].copy()
            self.shared["anchor"] = self.shared["home"].copy()
            self.shared["engaged"] = False
            self.shared["ready"] = True
            self.phase = "teleop"
            self.get_logger().info("Teleop ready. Click trackpad to engage.")
            return

        tgt = self.shared["target"].copy()
        Rc = self.shared["Rc"].copy()
        engaged = self.shared["engaged"]

        if engaged and not self._was_engaged:
            self.Rc_ref = Rc.copy()
            self.R_anchor = self.ik.fk_rotation()
        if (not engaged) and self._was_engaged and all(j in self.pos for j in ARM):
            q_hold = self.ik.neutral()
            for name in ARM:
                q_hold[self.ik.qindex(name)] = self.pos[name]
            self.ik.reset_to(q_hold)
            self.shared["target"] = self.ik.fk_translation()
            tgt = self.shared["target"].copy()
        self._was_engaged = engaged

        if engaged and self.Rc_ref is not None:
            dR = M @ (Rc @ self.Rc_ref.T) @ M.T
            R_des = dR @ self.R_anchor
        else:
            R_des = self.ik.fk_rotation()

        q_arm = self.ik.step(tgt, R_des, DT)
        self.dbg += 1
        if self.dbg % 50 == 0:
            print("blocked=%s" % self.ik.blocked)
            for i,j in enumerate(ARM):
                cmd = float(q_arm[i])
                meas = self.pos.get(j)
                eff = self.eff.get(j)
                ms = "n/a" if meas is None else "%+.4f" % meas
                gs = "n/a" if meas is None else "%+.4f" % (cmd - meas)
                es = "n/a" if eff is None else "%+.3f" % eff
                print("%-16s cmd=%+.4f meas=%s gap=%s eff=%s" % (j, cmd, ms, gs, es))
        if self.ik.blocked:
            self._blk += 1
            if self._blk % 50 == 1:
                self.get_logger().info("collision == block")
        else:
            self._blk = 0
        self.send_arm(q_arm)

        now = time.monotonic()
        trig_last = self.shared["trig_last"]
        held = (now - trig_last) < TRIG_TIMEOUT
        if not held:
            self._held_since = None
        elif self._held_since is None:
            self._held_since = now
        want = held and (now - self._held_since) >= TRIG_HOLD
        if want != self._grip_last and self.send_grip(GRIP_CLOSE if want else GRIP_OPEN):
            self._grip_last = want


def main():
    """Init rclpy, spin the Bridge node, shut down VR and ROS cleanly."""
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._vr is not None:
            openvr.shutdown()
        node.destroy_node()
        rclpy.ok() and rclpy.shutdown()


if __name__ == "__main__":
    main()