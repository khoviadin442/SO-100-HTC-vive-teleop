import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

class CmdNode(Node):
    amp = [0.0, 1, 1.3, 1.3, 1] 
    def __init__(self):
        super().__init__('cmd_node') 
        self.pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory',10)
        self.t0 = None
        self.home = False
        #self.start = self.get_clock().now()
        self.timer = self.create_timer(0.05, self.tick)

    def send_pose(self, positions, sec = 0, nanosec = 0):
        msg = JointTrajectory()
        msg.joint_names = ['Shoulder_Rotation','Shoulder_Pitch','Elbow','Wrist_Pitch','Wrist_Roll']
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(sec=sec, nanosec = nanosec)
        msg.points = [pt]
        self.pub.publish(msg)

    def tick(self):
        if self.pub.get_subscription_count() == 0:
            return
        now = self.get_clock().now()
        if self.t0 is None:
            self.t0 = now
        past = (now - self.t0).nanoseconds * 1e-9
        
        if past < 4:
            if not self.home:
                self.send_pose([0.0, 0.0, 0.0, 0.0, 0.0], sec = 4)
                self.home = True
            return
        t = past - 4
        positions = [a * math.sin(2*math.pi*0.5*t) for a in self.amp]
        self.send_pose(positions, nanosec=100000000)

def main(args=None):
    rclpy.init(args=args)
    node = CmdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()