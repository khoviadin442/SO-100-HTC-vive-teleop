# SO-100-HTC-vive-teleop

## Install 
```bash
git clone git@github.com:inria-paris-robotics-lab/SO-100-HTC-vive-teleop.git
cd so-100-htc-vive-teleop

pixi run init-src

pixi run build
```

You can now either run `pixi shell` to get a shell in the pixi environement (with ROS2 & the SO100 packages) or you can run one of the following:
 - `pixi run sim`
 - `pixi run rviz`
 - `pixi run hardware`
 - `pixi run rosdep-install`

The packages are build using symlink-install which means that you don't have to rebuild every time you modify a python file (you'll have to rebuild the c++ files though)