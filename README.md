# BlueBoat Marine Navigation Stack

ROS Noetic navigation stack for [Clearpath BlueBoat](https://clearpathrobotics.com/blueboat/) with marine-tuned SLAM, localization, and dynamic obstacle avoidance.

![BlueBoat avoiding buoys](docs/demo.jpg)  <!-- Optional: add screenshot later -->

## ✅ Features
- **Gmapping SLAM** with pointcloud-to-laserscan conversion (marine lidar tuned)
- **AMCL localization** against static harbor maps
- **Dynamic obstacle avoidance** for buoys/vessels (marine-tuned 12m×12m local costmap)
- **VRX-compatible** sensor configuration (buoyancy plugins, wave-aware planning)
- GPS/IMU fusion via `robot_localization` (dual EKF architecture)

## 🚀 Quick Start

### Build
```bash
cd ~/blueboat_ws
catkin_make
source devel/setup.bash
