#!/usr/bin/env python3
"""
Weather Data Collection Suite (Mission-Triggered) - FINAL VERSION
Workflow: Launch -> Publish Weather Config -> Wait for Mission Complete -> Save Bag -> Kill -> Next
Total Bags: 15 (3 Weather Types × 5 Severities)
Naming: [n,l,m,h,e][weather_type]_static8_lio_[wave_type].bag
"""
import rospy
from rospy.msg import AnyMsg
import subprocess
import os
import signal
import time
import json
from datetime import datetime
from pathlib import Path
from std_msgs.msg import Bool, Float32, String

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PACKAGE_NAME = "blueboat"
LAUNCH_FILE = "all-alg.launch"
RAIN_SCRIPT_PATH = "/home/ppninja/blueboat_ws/src/blueboat/scripts/pc_rain.py"
WAYPOINT_SCRIPT_PATH = "/home/ppninja/blueboat_ws/src/blueboat/scripts/waypoint_follower.py"
OUTPUT_DIR = "/home/ppninja/bags/alg/"

# ==============================================================================
# LOOP CONFIGURATION
# ==============================================================================
WEATHER_TYPES = ['rain'] #, 'fog', 'spray'
SEVERITY_LEVELS = [0, 2, 7, 35, 55]
SEVERITY_LABELS = {
    0: 'n', 2: 'l', 7: 'm', 35: 'h', 55: 'e'
}
WAVE_TYPE = 'nw'

# ==============================================================================
# TOPICS (MATCHED TO YOUR SYSTEM)
# ==============================================================================
# Critical topics to wait for before starting
WAIT_TOPICS = [
    "/blueboat/sensors/lidars/lidar_blueboat/points",
    "/blueboat/sensors/imu/imu/data",
    "/clock",
    "/move_base/status"
]

BAG_TOPICS = [
    # ========== LIO / SLAM (Raw + Noisy) ==========
    "/Laser_map",
    "/Laser_map_noisy",
    "/Odometry",
    "/Odometry_noisy",
    "/cloud_effected",
    "/cloud_effected_noisy",
    "/cloud_registered",
    "/cloud_registered_noisy",
    "/cloud_registered_body",
    "/cloud_registered_body_noisy",
    "/path",
    "/path_noisy",
    
    # ========== Localization ==========
    "/amcl_pose",
    "/particlecloud",
    "/map",
    "/map_metadata",
    "/map_updates",
    "/tf",
    "/tf_static",
    
    # ========== Robot State ==========
    "/blueboat/joint_states",
    "/blueboat/scan",
    
    # ========== Sensors - LiDAR ==========
    "/blueboat/sensors/lidars/lidar_blueboat/points",
    "/noisyLidar",
    
    # ========== Sensors - IMU ==========
    "/blueboat/sensors/imu/imu/data",
    "/blueboat/sensors/imu/imu/data/bias",
    
    # ========== Sensors - GPS ==========
    "/blueboat/sensors/gps/gps/fix",
    "/blueboat/sensors/gps/gps/fix_velocity",
    
    # ========== Sensors - Sonar ==========
    "/blueboat/sensors/lidars/sonar/scan",
    
    # ========== Sensors - Pingers ==========
    "/blueboat/sensors/pingers/pinger/marker/signal",
    "/blueboat/sensors/pingers/pinger/range_bearing",
    
    # ========== Robot Localization ==========
    "/blueboat/robot_localization/gps/filtered",
    "/blueboat/robot_localization/odometry/filtered",
    "/blueboat/robot_localization/odometry/gps",
    "/blueboat/robot_localization/set_pose",
    
    # ========== Thrusters / Control ==========
    "/blueboat/thrusters/left_thrust_angle",
    "/blueboat/thrusters/left_thrust_cmd",
    "/blueboat/thrusters/right_thrust_angle",
    "/blueboat/thrusters/right_thrust_cmd",
    "/cmd_vel",
    "/lateral_cmd",
    
    # ========== Navigation / Move Base ==========
    "/move_base/status",
    "/move_base/current_goal",
    "/move_base/goal",
    "/move_base/feedback",
    "/move_base/result",
    "/move_base/NavfnROS/plan",
    "/move_base/TebLocalPlannerROS/global_plan",
    "/move_base/TebLocalPlannerROS/local_plan",
    "/move_base/TebLocalPlannerROS/obstacles",
    "/move_base/global_costmap/costmap",
    "/move_base/local_costmap/costmap",
    
    # ========== Ground Truth ==========
    "/p3d/groundtruth",
    "/gazebo/link_states",
    "/gazebo/model_states",
    
    # ========== Mission ==========
    "/mission_complete",
    "/waypoints",
    "/clicked_point",
    "/initialpose",
    
    # ========== System ==========
    "/clock",
    "/diagnostics",
    "/rosout",
    "/rosout_agg",
    
    # ========== Visualization ==========
    "/nav_marker",
    "/nav_marker_array",
    "/visualization_marker",
    "/visualization_marker_array",
    
    # ========== Environment ==========
    "/vrx",
    "/vrx/debug/wind/direction",
    "/vrx/debug/wind/speed"
]


class WeatherConfigPublisher:
    """Publishes weather configuration to pc_rain.py via ROS topics"""
    def __init__(self):
        self.severity_pub = rospy.Publisher('/Noise_LEVEL', Float32, queue_size=10, latch=True)
        self.mode_pub = rospy.Publisher('/weather_mode', String, queue_size=10, latch=True)
        time.sleep(1)  # Wait for subscribers to connect
    
    def set_weather(self, weather_type, severity):
        """Publish weather configuration"""
        rospy.loginfo(f"Publishing weather config: type={weather_type}, severity={severity}")
        
        # Publish severity
        severity_msg = Float32()
        severity_msg.data = float(severity)
        self.severity_pub.publish(severity_msg)
        
        # Publish weather mode (map to pc_rain.py's expected format)
        mode_msg = String()
        if weather_type == 'rain':
            mode_msg.data = 'rain'
        elif weather_type == 'fog':
            mode_msg.data = 'fog_coastal'  # Use coastal fog as default
        elif weather_type == 'spray':
            mode_msg.data = 'spray_strong'  # Use strong spray as default
        else:
            mode_msg.data = weather_type
        
        self.mode_pub.publish(mode_msg)
        
        # Wait for pc_rain.py to process the config
        time.sleep(2)
        rospy.loginfo(f"✓ Weather config published")


class MissionOrchestrator:
    def __init__(self):
        rospy.init_node('weather_orchestrator', anonymous=True, disable_signals=True)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        self.mission_done = False
        self.sub = rospy.Subscriber('/mission_complete', Bool, self.mission_callback)
        # NOTE: 'teleop' is NOT included - it's launched via all-alg.launch and killed with the main launch process
        self.procs = {'launch': None, 'rain': None, 'waypoint': None, 'bag': None, 'nfastlio': None}
        self.weather_pub = None
        
        # Verify scripts exist
        if not os.path.isfile(RAIN_SCRIPT_PATH):
            rospy.logerr(f"❌ Rain script not found: {RAIN_SCRIPT_PATH}")
        if not os.path.isfile(WAYPOINT_SCRIPT_PATH):
            rospy.logerr(f"❌ Waypoint script not found: {WAYPOINT_SCRIPT_PATH}")
        
        rospy.loginfo("=== Orchestrator Ready ===")
    
    def wait_for_stream(self, topic, timeout=60.0, min_msgs=3):
        """
        Wait until `topic` is actively publishing (receives >= min_msgs messages).
        Uses AnyMsg so we don't need the concrete message type.
        """
        rospy.loginfo(f"Waiting for stream on {topic} (need {min_msgs} msgs, timeout={timeout}s)...")
        count = {'n': 0}
        
        def cb(_msg):
            count['n'] += 1
        
        sub = rospy.Subscriber(topic, AnyMsg, cb, queue_size=10)
        start = time.time()
        
        try:
            while time.time() - start < timeout and not rospy.is_shutdown():
                if count['n'] >= min_msgs:
                    rospy.loginfo(f"✓ Stream detected on {topic} (received {count['n']} msgs)")
                    return True
                time.sleep(0.1)
            
            rospy.logerr(f"✗ Timeout waiting for stream on {topic}. Received {count['n']} msgs.")
            return False
        finally:
            sub.unregister()
    
    def launch_nfastlio_after_stream(self, topic="/blueboat/sensors/lidars/lidar_blueboat/points", timeout=60):
        """
        Wait for topic to STREAM data, then delay 5 seconds, then launch Fast-LIO.
        """
        if not self.wait_for_stream(topic, timeout=timeout, min_msgs=3):
            return False
        
        rospy.loginfo("Delaying 5 seconds before launching Fast-LIO...")
        time.sleep(5)
        
        rospy.loginfo("Launching Fast-LIO (noisy)...")
        self.procs['nfastlio'] = subprocess.Popen(
            ["roslaunch", "fast_lio", "fastlio_noisy.launch"],
            preexec_fn=os.setsid
        )
        time.sleep(3)
        return True
    
    def mission_callback(self, msg):
        if msg.data:
            rospy.loginfo("🚩 Mission Complete Signal Received!")
            self.mission_done = True
    
    def kill_pg(self, proc, name):
        if not proc:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=5)
            rospy.loginfo(f"✓ Killed: {name}")
        except Exception as e:
            try:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=2)
                rospy.loginfo(f"✓ Force killed: {name}")
            except:
                rospy.logwarn(f"⚠ Could not kill: {name}")
    
    def kill_all(self):
        rospy.loginfo("Tearing down stack...")
        self.kill_pg(self.procs['bag'], "bag")
        self.kill_pg(self.procs['waypoint'], "waypoint")
        self.kill_pg(self.procs['rain'], "rain")
        self.kill_pg(self.procs['nfastlio'], "nfastlio")
        self.kill_pg(self.procs['launch'], "launch")
        # NOTE: teleop is killed with the main launch process (all-alg.launch)
        subprocess.run(["pkill", "-9", "-f", "gzserver"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "gzclient"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
    
    def wait_topics(self, topics, timeout=60):
        rospy.loginfo(f"Waiting for {len(topics)} topics...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                cur = subprocess.check_output(['rostopic', 'list'], timeout=5).decode().splitlines()
                missing = [t for t in topics if t not in cur]
                if not missing:
                    rospy.loginfo("✓ All topics ready!")
                    return True
                if int(time.time() - start) % 10 == 0:
                    rospy.loginfo(f"  Still waiting... Missing: {len(missing)} topics")
            except Exception as e:
                rospy.logwarn(f"Error checking topics: {e}")
            time.sleep(1)
        rospy.logerr(f"✗ Topic timeout! Missing: {missing}")
        return False
    
    def build_rain_cmd(self, w_type, severity):
        """Build command to launch pc_rain.py (no CLI args needed - uses ROS topics)"""
        # pc_rain.py doesn't use CLI args, it listens to ROS topics
        # We just launch it normally and configure via topics
        cmd = ["python3", RAIN_SCRIPT_PATH]
        return cmd
    
    def run_combo(self, w_type, severity):
        sev_label = SEVERITY_LABELS.get(severity, f's{severity}')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_name = f"{OUTPUT_DIR}/{sev_label}{w_type}_staticxl_lio_{WAVE_TYPE}_{timestamp}"
        
        rospy.loginfo(f"\n>>> Running: {w_type} | Severity: {severity} ({sev_label})")
        self.mission_done = False
        
        try:
            # 1. Launch Sim
            rospy.loginfo("Launching Simulation...")
            self.procs['launch'] = subprocess.Popen(
                ["roslaunch", PACKAGE_NAME, LAUNCH_FILE],
                preexec_fn=os.setsid)
            time.sleep(5)
            
            if not self.wait_topics(WAIT_TOPICS, timeout=45):
                raise Exception("Critical topics not found - check launch file")
            
            if not self.launch_nfastlio_after_stream(topic="/blueboat/sensors/lidars/lidar_blueboat/points", timeout=60):
                raise Exception("Fast-LIO stream trigger failed (no streaming data)")
            
            # 2. Start Noise Injector
            rain_cmd = self.build_rain_cmd(w_type, severity)
            rospy.loginfo(f"Starting noise injector...")
            rospy.loginfo(f"  Cmd: {' '.join(rain_cmd)}")
            self.procs['rain'] = subprocess.Popen(rain_cmd, preexec_fn=os.setsid)
            time.sleep(3)
            
            # Check if rain script died immediately
            if self.procs['rain'].poll() is not None:
                rospy.logwarn("⚠ Rain script exited immediately!")
            
            # 3. Publish Weather Configuration via ROS Topics
            rospy.loginfo("Publishing weather configuration to pc_rain.py...")
            self.weather_pub = WeatherConfigPublisher()
            self.weather_pub.set_weather(w_type, severity)
            
            # 4. Start Waypoint Follower
            rospy.loginfo("Starting Waypoint Follower...")
            rospy.loginfo(f"  Cmd: python3 {WAYPOINT_SCRIPT_PATH}")
            self.procs['waypoint'] = subprocess.Popen(
                ["python3", WAYPOINT_SCRIPT_PATH],
                preexec_fn=os.setsid)
            time.sleep(3)
            
            # Check if waypoint script died immediately
            if self.procs['waypoint'].poll() is not None:
                raise Exception("Waypoint follower exited immediately - check script!")
            
            # 5. Start Bagging (Use absolute path)
            rospy.loginfo(f"Recording Bag: {bag_name}")
            bag_cmd = ["rosbag", "record", "-O", bag_name] + BAG_TOPICS
            self.procs['bag'] = subprocess.Popen(
                bag_cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            time.sleep(2)
            
            # 6. WAIT FOR MISSION COMPLETE
            rospy.loginfo("Waiting for mission completion signal...")
            rate = rospy.Rate(1)
            while not self.mission_done and not rospy.is_shutdown():
                if self.procs['waypoint'] and self.procs['waypoint'].poll() is not None:
                    rospy.logwarn("⚠ Waypoint follower died prematurely!")
                    break
                rate.sleep()
            
            # 7. Stop Bag
            rospy.loginfo("Mission done. Stopping bag...")
            self.kill_pg(self.procs['bag'], "bag")
            self.procs['bag'] = None
            
            # 8. Verify Bag Saved
            bag_file = bag_name + ".bag"
            if os.path.isfile(bag_file):
                size_mb = os.path.getsize(bag_file) / (1024 * 1024)
                rospy.loginfo(f"✓ Bag saved: {bag_file} ({size_mb:.1f} MB)")
            else:
                rospy.logerr(f"✗ Bag file not found! Expected: {bag_file}")
            
            # 9. Save Metadata
            meta = {
                "bag_path": bag_file,
                "weather_type": w_type,
                "severity": severity,
                "severity_label": sev_label,
                "wave_type": WAVE_TYPE,
                "timestamp": datetime.now().isoformat(),
                "topics_recorded": BAG_TOPICS
            }
            with open(bag_file + ".json", 'w') as f:
                json.dump(meta, f, indent=2)
            
            return True
            
        except Exception as e:
            rospy.logerr(f"✗ Failed: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return False
        finally:
            self.kill_all()
    
    def run_suite(self):
        rospy.loginfo("\n" + "="*70)
        rospy.loginfo("=== STARTING 15 BAG SUITE ===")
        rospy.loginfo(f"Weather Types: {WEATHER_TYPES}")
        rospy.loginfo(f"Severities: {SEVERITY_LEVELS}")
        rospy.loginfo(f"Output Dir: {OUTPUT_DIR}")
        rospy.loginfo("="*70)
        
        self.kill_all()
        count = 0
        results = []
        
        for w_type in WEATHER_TYPES:
            rospy.loginfo(f"\n{'='*70}")
            rospy.loginfo(f">>> WEATHER TYPE: {w_type.upper()}")
            rospy.loginfo(f"{'='*70}")
            
            for sev in SEVERITY_LEVELS:
                count += 1
                rospy.loginfo(f"\n[{count}/15] Starting: {w_type} @ severity {sev}")
                
                if rospy.is_shutdown():
                    break
                
                ok = self.run_combo(w_type, sev)
                results.append((w_type, sev, ok))
                time.sleep(2)
        
        # Summary
        rospy.loginfo("\n" + "="*70)
        rospy.loginfo("=== SUITE SUMMARY ===")
        rospy.loginfo("="*70)
        for w, s, ok in results:
            label = SEVERITY_LABELS.get(s, f's{s}')
            status = "✓ PASS" if ok else "✗ FAIL"
            rospy.loginfo(f"{w:<12} {s:<10} {label:<8} {status}")
        rospy.loginfo("\n=== COMPLETE ===")


if __name__ == '__main__':
    orch = None
    try:
        orch = MissionOrchestrator()
        orch.run_suite()
    except KeyboardInterrupt:
        rospy.logwarn("Interrupted")
    finally:
        if orch:
            orch.kill_all()