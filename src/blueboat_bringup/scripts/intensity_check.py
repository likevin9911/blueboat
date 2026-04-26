#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def callback(msg):
    # Extract points (x, y, z, intensity, ring)
    points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity", "ring"), skip_nans=True))
    
    if not points:
        print("No points received")
        return

    print(f"Received {len(points)} points.")
    print("First 10 Intensities:")
    for i in range(min(10, len(points))):
        p = points[i]
        print(f"Point {i}: XYZ={p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f} | Intensity={p[3]:.4f} | Ring={p[4]}")
        
    # Check for variation
    intensities = [p[3] for p in points]
    if max(intensities) - min(intensities) > 0.01:
        print("\n✅ SUCCESS: Intensities are VARYING!")
    else:
        print("\n❌ FAILURE: All intensities are identical.")
    exit(0) # Exit after first message

rospy.init_node('intensity_checker')
sub = rospy.Subscriber('/blueboat/sensors/lidars/lidar_blueboat/points', PointCloud2, callback)
rospy.spin()