import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = LaunchDescription([
        DeclareLaunchArgument(
            'serial_port',
            default_value = '/dev/ttyACM0',
            description = 'Servo controller board serial port'
        ),
    ])

    def launch_setup(context, *args, **kwargs):

        serial_port = LaunchConfiguration('serial_port').perform(context)
        so_arm_100_bringup_path = get_package_share_directory('so_arm_100_bringup')

        hardware_launch = os.path.join(
            so_arm_100_bringup_path,
            'launch',
            'hardware.launch.py'
        )

        hardware = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(hardware_launch),
            launch_arguments = {
                'serial_port' : serial_port,
            }.items()
        )

        cmd_node = Node(
            package = 'so_arm_100_teleop',
            executable = 'cmd_node',
            output = 'screen'
        )

        return [hardware, cmd_node]
    return LaunchDescription([
        arguments,
        OpaqueFunction(function=launch_setup)
    ])

