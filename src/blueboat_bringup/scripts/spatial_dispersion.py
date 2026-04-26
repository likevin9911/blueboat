#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA-Based Weather Noise Filter for LiDAR Point Clouds
======================================================
PURE NUMPY IMPLEMENTATION - No ros_numpy, No Open3D
Method: Orthogonal Distance to Principal Plane (k-NN PCA)

Dependencies:
    - ROS (rosbag, sensor_msgs, std_msgs)
    - numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy import linalg
import struct
import rosbag
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# ============================================================================
# MANUAL POINTCLOUD2 <-> NUMPY CONVERSION (Replaces ros_numpy)
# ============================================================================

def pointcloud2_to_xyz_array(msg):
    """
    Convert sensor_msgs/PointCloud2 to numpy array (N, 3) using pure Python.
    Handles float32 x, y, z fields with arbitrary offsets.
    """
    if msg.width == 0 or msg.height == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    # Build field lookup
    fields = {f.name: f for f in msg.fields}
    required = ['x', 'y', 'z']
    if not all(f in fields for f in required):
        raise ValueError(f"PointCloud2 missing required fields {required}")
    
    # Extract field properties
    def unpack_value(data, offset, field):
        """Unpack a single value from binary data based on PointField datatype"""
        DATATYPE_MAP = {
            PointField.INT8: 'b', PointField.UINT8: 'B',
            PointField.INT16: 'h', PointField.UINT16: 'H',
            PointField.INT32: 'i', PointField.UINT32: 'I',
            PointField.FLOAT32: 'f', PointField.FLOAT64: 'd'
        }
        fmt = DATATYPE_MAP.get(field.datatype)
        if fmt is None:
            raise ValueError(f"Unsupported datatype: {field.datatype}")
        return struct.unpack_from(fmt, data, offset)[0]
    
    n_points = msg.width * msg.height
    points = np.zeros((n_points, 3), dtype=np.float32)
    
    for i in range(n_points):
        byte_offset = i * msg.point_step
        points[i, 0] = unpack_value(msg.data, byte_offset + fields['x'].offset, fields['x'])
        points[i, 1] = unpack_value(msg.data, byte_offset + fields['y'].offset, fields['y'])
        points[i, 2] = unpack_value(msg.data, byte_offset + fields['z'].offset, fields['z'])
    
    return points


def xyz_array_to_pointcloud2(points, stamp, frame_id):
    """
    Convert numpy array (N, 3) to sensor_msgs/PointCloud2 using pure Python.
    """
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    
    msg.height = 1
    msg.width = len(points)
    
    # Define fields: x, y, z as FLOAT32 at offsets 0, 4, 8
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    
    msg.is_bigendian = False
    msg.point_step = 12  # 3 floats * 4 bytes
    msg.row_step = msg.point_step * len(points)
    msg.is_dense = True  # No invalid points
    
    # Pack numpy array to bytes
    msg.data = points.astype(np.float32).tobytes()
    
    return msg


# ============================================================================
# CORE PCA FILTERING LOGIC (Pure NumPy/SciPy)
# ============================================================================

class PCAPointCloudFilter:
    def __init__(self, k_neighbors=20, distance_threshold_factor=3.0):
        """
        Args:
            k_neighbors: Number of nearest neighbors for local PCA
            distance_threshold_factor: Sigma multiplier for outlier threshold
        """
        self.k = k_neighbors
        self.threshold_factor = distance_threshold_factor
        
    def compute_local_pca_features(self, points):
        """
        Compute orthogonal distance and geometric features for each point.
        Returns dict with features suitable for unsupervised learning.
        """
        n = points.shape[0]
        if n < self.k:
            raise ValueError(f"Need at least {self.k} points, got {n}")
        
        orth_dist = np.zeros(n)
        eigenvalues = np.zeros((n, 3))
        
        # Build KD-Tree for neighbor search
        tree = KDTree(points)
        _, indices = tree.query(points, k=self.k)
        
        # Loop over points (vectorization is complex for per-point PCA)
        for i in range(n):
            neighbors = points[indices[i]]
            centroid = neighbors.mean(axis=0)
            centered = neighbors - centroid
            
            # 3x3 covariance matrix
            cov = centered.T @ centered / (self.k - 1)
            
            # Eigen decomposition (ascending order: λ3 ≤ λ2 ≤ λ1)
            evals, evecs = linalg.eigh(cov)
            eigenvalues[i] = evals[::-1]  # Store descending: λ1 ≥ λ2 ≥ λ3
            
            # Normal vector = eigenvector of smallest eigenvalue
            v_normal = evecs[:, 0]
            
            # Orthogonal distance of query point to local plane
            orth_dist[i] = np.abs(np.dot(points[i] - centroid, v_normal))
        
        # Compute geometric features for ML
        λ1, λ2, λ3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
        sum_λ = λ1 + λ2 + λ3 + 1e-8  # Avoid division by zero
        
        features = {
            'orth_dist': orth_dist,
            'planarity': (λ2 - λ3) / sum_λ,      # Surface-like: high
            'linearity': (λ1 - λ2) / sum_λ,      # Edge-like: high
            'omnivariance': np.cbrt(λ1 * λ2 * λ3),  # Scattered: high
            'eigenentropy': -np.sum(
                (eigenvalues / (sum_λ[:, None] + 1e-8)) * 
                np.log(eigenvalues / (sum_λ[:, None] + 1e-8) + 1e-8), axis=1
            ),
            'eigenvalues': eigenvalues
        }
        return features

    def filter_weather_noise(self, points, features):
        """
        Adaptive thresholding to remove weather-induced floaters.
        Returns boolean mask: True = keep (surface), False = remove (noise)
        """
        orth = features['orth_dist']
        plan = features['planarity']
        
        # Range-adaptive threshold: noise increases with distance
        ranges = np.linalg.norm(points, axis=1)
        range_norm = (ranges - ranges.min()) / (ranges.max() - ranges.min() + 1e-8)
        
        # Base threshold from robust statistics
        base_thresh = np.median(orth) + self.threshold_factor * np.std(orth)
        adaptive_thresh = base_thresh * (1.0 + 0.5 * range_norm)
        
        # Combine geometric cues
        mask_surface = orth < adaptive_thresh
        mask_planar = plan > 0.05  # Reject purely scattered points
        
        return mask_surface & mask_planar


# ============================================================================
# VISUALIZATION (Matplotlib Only)
# ============================================================================

def visualize_results(orig_pts, clean_pts, features, save_path=None):
    """Generate diagnostic plots for research analysis"""
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: 3D Original vs Cleaned
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(orig_pts[:, 0], orig_pts[:, 1], orig_pts[:, 2], 
                c='lightgray', s=0.5, alpha=0.4, label='Original')
    ax1.scatter(clean_pts[:, 0], clean_pts[:, 1], clean_pts[:, 2], 
                c='green', s=0.5, alpha=0.9, label='Cleaned')
    ax1.set_title('3D Point Cloud: Original vs Filtered')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.legend(fontsize=8)
    
    # Plot 2: Orthogonal Distance Heatmap (Top-Down)
    ax2 = fig.add_subplot(132)
    sc = ax2.scatter(orig_pts[:, 0], orig_pts[:, 1], 
                     c=np.log1p(features['orth_dist']), 
                     cmap='magma', s=0.5, alpha=0.6)
    plt.colorbar(sc, label='log(1 + Orthogonal Distance)', ax=ax2)
    ax2.set_title('Spatial Dispersion: Weather Floaters = Bright')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    
    # Plot 3: Feature Space for Unsupervised Learning
    ax3 = fig.add_subplot(133)
    ax3.scatter(features['orth_dist'], features['planarity'], 
                c='steelblue', s=1, alpha=0.3, label='All Points')
    
    # Highlight removed points
    mask = ~((features['orth_dist'] < np.median(features['orth_dist']) + 3*np.std(features['orth_dist'])) & 
             (features['planarity'] > 0.05))
    if mask.any():
        ax3.scatter(features['orth_dist'][mask], features['planarity'][mask], 
                    c='red', s=2, alpha=0.6, label='Removed (Weather Noise)')
    
    ax3.axvline(x=np.median(features['orth_dist']) + 3*np.std(features['orth_dist']), 
                color='orange', linestyle='--', label='Ortho Threshold')
    ax3.axhline(y=0.05, color='purple', linestyle='--', label='Planarity Threshold')
    ax3.set_xlabel('Orthogonal Distance to Principal Plane')
    ax3.set_ylabel('Planarity = (λ₂ - λ₃) / Σλᵢ')
    ax3.set_title('Feature Space: Noise vs Surface Separation')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================================
# MAIN BAG PROCESSING PIPELINE
# ============================================================================

def process_bag_file(bag_path, topic='/velodyne_points', output_bag=None, 
                     visualize_first=True, max_frames=None):
    """
    Process ROS bag file with PCA weather filter.
    
    Args:
        bag_path: Input .bag file
        topic: PointCloud2 topic name
        output_bag: Optional output .bag path for cleaned clouds
        visualize_first: Generate plots for first frame
        max_frames: Limit processing for testing (None = all frames)
    """
    print(f"🔍 Opening bag: {bag_path}")
    bag = rosbag.Bag(bag_path, 'r')
    
    filter_engine = PCAPointCloudFilter(k_neighbors=20, distance_threshold_factor=2.5)
    
    if output_bag:
        out_bag = rosbag.Bag(output_bag, 'w')
    
    frame_count = 0
    kept_total, removed_total = 0, 0
    
    try:
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            if max_frames and frame_count >= max_frames:
                break
                
            # Convert to numpy
            try:
                points = pointcloud2_to_xyz_array(msg)
            except Exception as e:
                print(f"⚠ Frame {frame_count}: Conversion error - {e}")
                frame_count += 1
                continue
                
            if len(points) == 0 or np.isnan(points).any():
                points = points[~np.isnan(points).any(axis=1)]
                if len(points) == 0:
                    continue
            
            # === CORE FILTERING ===
            features = filter_engine.compute_local_pca_features(points)
            mask = filter_engine.filter_weather_noise(points, features)
            cleaned = points[mask]
            # ======================
            
            # Stats
            n_orig, n_clean = len(points), len(cleaned)
            kept_total += n_clean
            removed_total += (n_orig - n_clean)
            
            if visualize_first and frame_count == 0:
                print(f"\n📊 First Frame Analysis:")
                print(f"   Original points: {n_orig:,}")
                print(f"   Cleaned points : {n_clean:,} ({100*n_clean/n_orig:.1f}% kept)")
                print(f"   Removed (noise): {n_orig - n_clean:,}")
                print(f"   Mean orth_dist (noise): {features['orth_dist'][~mask].mean():.4f} m")
                print(f"   Mean orth_dist (surface): {features['orth_dist'][mask].mean():.4f} m")
                visualize_results(points, cleaned, features, save_path="pca_filter_diagnostic.png")
            
            # Write to output bag if requested
            if output_bag:
                cleaned_msg = xyz_array_to_pointcloud2(
                    cleaned, stamp=msg.header.stamp, frame_id=msg.header.frame_id)
                out_bag.write(topic_name, cleaned_msg, t)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"✓ Processed {frame_count} frames...", end='\r')
    
    finally:
        bag.close()
        if output_bag:
            out_bag.close()
    
    # Final summary
    total = kept_total + removed_total
    print(f"\n✅ Processing complete:")
    print(f"   Frames processed: {frame_count}")
    print(f"   Total points: {total:,}")
    print(f"   Kept (surface) : {kept_total:,} ({100*kept_total/total:.1f}%)")
    print(f"   Removed (noise): {removed_total:,} ({100*removed_total/total:.1f}%)")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import os
    import sys
    
    # === USER CONFIGURATION ===
    INPUT_BAG = "/home/ppninja/bags/alg/lrain_staticxl_lio_nw.bag"           # ← Change this
    OUTPUT_BAG = "weather_dataset_cleaned.bag"  # ← Optional output
    TOPIC_NAME = "/cloud_registered_body_noisy"             # ← Your LiDAR topic
    MAX_FRAMES = None                           # Set to e.g., 5 for testing
    # ==========================
    
    if not os.path.exists(INPUT_BAG):
        print(f"❌ Input bag not found: {INPUT_BAG}")
        print("\n💡 Running DEMO MODE with synthetic data...")
        
        # Generate synthetic point cloud: ground plane + weather floaters
        np.random.seed(42)
        # Ground surface (planar)
        ground = np.random.randn(2000, 3) * [10, 10, 0.1] + [0, 0, 0]
        # Vertical wall (linear structure)
        wall = np.random.randn(500, 3) * [0.1, 10, 5] + [15, 0, 2.5]
        # Weather floaters (scattered, high orth_dist)
        rain = np.random.randn(300, 3) * [2, 2, 2] + [5, 5, 3]
        snow = np.random.randn(200, 3) * [1.5, 1.5, 1.5] + [-3, 8, 1.8]
        
        demo_points = np.vstack([ground, wall, rain, snow])
        
        # Run filter
        engine = PCAPointCloudFilter(k_neighbors=15)
        feats = engine.compute_local_pca_features(demo_points)
        mask = engine.filter_weather_noise(demo_points, feats)
        
        # Visualize
        visualize_results(demo_points, demo_points[mask], feats, 
                         save_path="demo_pca_filter.png")
        print("✓ Demo visualization saved: demo_pca_filter.png")
        sys.exit(0)
    
    # Process real bag file
    process_bag_file(
        bag_path=INPUT_BAG,
        topic=TOPIC_NAME,
        output_bag=OUTPUT_BAG,
        visualize_first=True,
        max_frames=MAX_FRAMES
    )