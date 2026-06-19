import os, sys, yaml, rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
JOINT = sys.argv[1] if len(sys.argv) > 1 else 'Shoulder_Pitch'
class Rec(Node):
    def __init__(self):
        super().__init__('calib_one')
        self.rec = self.create_client(Trigger, 'record_position')
        self.tor = self.create_client(Trigger, 'toggle_torque')
        for name, cli in (('record_position', self.rec), ('toggle_torque', self.tor)):
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('waiting for %s...' % name)
    def call(self, cli):
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if not fut.done():
            raise RuntimeError('service timed out')
        return fut.result()
    def ensure_torque(self, off=True):
        target = 'disabled' if off else 'enabled'
        msg = ''
        for _ in range(4):
            msg = self.call(self.tor).message or ''
            if target in msg.lower():
                return msg
        raise RuntimeError('could not set torque %s (last: %s)' % (target, msg))
    def grab(self):
        return yaml.safe_load(self.call(self.rec).message)
def main():
    cfg = os.path.join(get_package_share_directory('so_arm_100_hardware'), 'config')
    path = os.path.join(cfg, 'calibration.yaml')
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault('joints', {})
    rclpy.init()
    n = Rec()
    print("Calibrating ONLY: %s" % JOINT)
    print("Forcing torque OFF...")
    print(" ->", n.ensure_torque(off=True))
    input("Arm LIMP? Push %s by hand to confirm, then Enter..." % JOINT)
    poses = {}
    for key, desc in (('min','MIN stop'), ('center','true ZERO pose'), ('max','MAX stop')):
        input("Move ONLY %s to %s, ease off force, Enter..." % (JOINT, desc))
        d = n.grab()
        if JOINT not in d:
            print("no data for %s, got: %s" % (JOINT, d))
            n.ensure_torque(off=False); rclpy.shutdown(); return
        poses[key] = d[JOINT]
        print("  %s = %d ticks (load %s)" % (key, d[JOINT]['ticks'], d[JOINT]['load']))
    mid = (poses['min']['ticks'] + poses['max']['ticks']) // 2
    off = poses['center']['ticks'] - mid
    print("  midpoint=%d center off %d ticks (%d deg) %s"
          % (mid, off, off * 360 // 4096, 'OK' if abs(off) < 120 else 'CHECK'))
    data['joints'][JOINT] = poses        
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print("\nUpdated %s in %s (other joints untouched)" % (JOINT, path))
    print("Restoring torque ON...")
    try:
        print(" ->", n.ensure_torque(off=False))
    except Exception as e:
        print("warn:", e)
    n.destroy_node(); rclpy.shutdown()
if __name__ == '__main__':
    main()