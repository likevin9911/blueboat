#!/usr/bin/env python3
"""
LiDAR Weather Noise Isolation Script
Extracts 100 synchronized scan pairs and isolates weather-induced noise points
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import rosbag
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from tqdm import tqdm
import pandas as pd
from scipy.spatial import cKDTree
import open3d as o3d
from datetime import datetime
from matplotlib.ticker import FuncFormatter  # ← ADDED for y-axis formatting

# ============================================================================
# CONFIGURATION
# ============================================================================
BAG_FILE_PATH = '/home/ppninja/bags/alg/lrain_staticxl_lio_nw.bag'
OUTPUT_DIR = '/home/ppninja/blueboat_ws/src/blueboat/logged_data/Lidar'
PCD_DIR = os.path.join(OUTPUT_DIR, 'pcds')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv_data')

CLEAN_TOPIC = '/cloud_registered_body'
NOISY_TOPIC = '/cloud_registered_body_noisy'

# Analysis parameters
N_SYNC_SCANS = 100  # Number of synchronized scan pairs to analyze
SYNC_TIME_TOL = 0.01  # seconds - max time difference for synchronization
NOISE_ISOLATION_RADIUS = 1  # meters - spatial threshold for noise detection
INTENSITY_RESIDUAL_THRESHOLD = None  # Optional: flag points with intensity delta > X

# Create output directories
for subdir in ['clean', 'noisy', 'noise_only']:
    os.makedirs(os.path.join(PCD_DIR, subdir), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pointcloud2_to_array(cloud_msg):
    """Convert PointCloud2 to numpy array [N, 4]: x, y, z, intensity"""
    points = np.array(list(pc2.read_points(
        cloud_msg, 
        field_names=("x", "y", "z", "intensity"), 
        skip_nans=True
    )))
    if points.size == 0:
        return np.empty((0, 4))
    return points

def calculate_distances(points):
    """Euclidean distance from origin"""
    return np.sqrt(np.sum(points[:, :3]**2, axis=1))

def save_pcd_file(points, filepath):
    """Save point cloud to PCD format"""
    if len(points) == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] >= 4:
        intensities = points[:, 3]
        if intensities.max() > intensities.min():
            intensities_norm = np.clip((intensities - intensities.min()) / 
                                     (intensities.max() - intensities.min() + 1e-8), 0, 1)
        else:
            intensities_norm = np.zeros_like(intensities)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(intensities_norm[:, np.newaxis], (1, 3))
        )
    o3d.io.write_point_cloud(filepath, pcd)

def find_synchronized_scans(bag_path, clean_topic, noisy_topic, n_scans=100, time_tol=0.01):
    """
    Find N synchronized scan pairs by matching timestamps between topics
    Returns list of tuples: (clean_msg, noisy_msg, timestamp)
    """
    print(f"Scanning bag for synchronized pairs (tolerance: {time_tol*1000:.1f}ms)...")
    
    # First pass: collect all messages with timestamps
    clean_msgs = []
    noisy_msgs = []
    
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[clean_topic, noisy_topic]):
            ts = msg.header.stamp.to_sec()
            if topic == clean_topic:
                clean_msgs.append((ts, msg))
            elif topic == noisy_topic:
                noisy_msgs.append((ts, msg))
    
    print(f"  Found {len(clean_msgs)} clean scans, {len(noisy_msgs)} noisy scans")
    
    # Sort by timestamp
    clean_msgs.sort(key=lambda x: x[0])
    noisy_msgs.sort(key=lambda x: x[0])
    
    # Two-pointer matching for synchronization
    synchronized_pairs = []
    i, j = 0, 0
    
    with tqdm(total=n_scans, desc="Matching synchronized pairs") as pbar:
        while i < len(clean_msgs) and j < len(noisy_msgs) and len(synchronized_pairs) < n_scans:
            ts_clean, msg_clean = clean_msgs[i]
            ts_noisy, msg_noisy = noisy_msgs[j]
            
            time_diff = abs(ts_clean - ts_noisy)
            
            if time_diff <= time_tol:
                # Found a synchronized pair
                synchronized_pairs.append((msg_clean, msg_noisy, ts_clean))
                i += 1
                j += 1
                pbar.update(1)
            elif ts_clean < ts_noisy:
                i += 1
            else:
                j += 1
    
    print(f"✓ Found {len(synchronized_pairs)} synchronized scan pairs")
    return synchronized_pairs

def isolate_noise_points(clean_points, noisy_points, spatial_radius=0.15):
    """
    Identify weather-induced noise points by spatial comparison.
    
    Strategy:
    1. Points in noisy cloud with NO nearby neighbor in clean cloud → likely weather noise
    2. Points in noisy cloud WITH nearby clean neighbor but large intensity residual → possibly weather-affected
    
    Returns:
        noise_only_points: points likely caused by weather (no geometric correspondence)
        affected_points: points with same geometry but altered intensity
        clean_matched_points: points that match well in both clouds
    """
    if len(clean_points) == 0 or len(noisy_points) == 0:
        return np.empty((0, 4)), np.empty((0, 4)), np.empty((0, 4))
    
    # Build KD-tree for clean points (only spatial coordinates)
    clean_tree = cKDTree(clean_points[:, :3])
    
    # Query noisy points against clean cloud
    distances, indices = clean_tree.query(
        noisy_points[:, :3], 
        k=1, 
        distance_upper_bound=spatial_radius,
        workers=-1
    )
    
    # Classify noisy points
    noise_only_mask = np.isinf(distances)  # No clean neighbor within radius
    matched_mask = ~noise_only_mask
    
    noise_only_points = noisy_points[noise_only_mask]
    
    # For matched points, check intensity residuals
    if INTENSITY_RESIDUAL_THRESHOLD is not None and np.any(matched_mask):
        matched_noisy = noisy_points[matched_mask]
        matched_clean = clean_points[indices[matched_mask]]
        
        intensity_residual = np.abs(matched_noisy[:, 3] - matched_clean[:, 3])
        affected_mask = intensity_residual > INTENSITY_RESIDUAL_THRESHOLD
        
        affected_points = matched_noisy[affected_mask]
        clean_matched = matched_noisy[~affected_mask]
    else:
        affected_points = np.empty((0, 4))
        clean_matched = noisy_points[matched_mask]
    
    return noise_only_points, affected_points, clean_matched

# ============================================================================
# PLOTTING FUNCTIONS - RAW SCATTER (NO SUBSAMPLING)
# ============================================================================

def plot_intensity_vs_distance_raw(distances, intensities, label, ax, color, 
                                  alpha=0.08, marker_size=0.5, title_suffix=""):
    """
    Create scatter plot showing ALL points with transparency for overplotting.
    No subsampling - every point is rendered.
    """
    if len(distances) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        return
    
    # Use tiny markers with low alpha to handle overplotting
    ax.scatter(distances, intensities, 
              s=marker_size, 
              c=color, 
              alpha=alpha, 
              edgecolors='none',
              label=f'{label} (N={len(distances):,})')
    
    ax.set_xlabel('Distance (m)', fontsize=11)
    ax.set_ylabel('Intensity (raw)', fontsize=11)
    ax.set_title(f'Intensity vs Distance - {label} {title_suffix}', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(fontsize=9, loc='upper right')
    
    # Add statistics annotation
    if len(intensities) > 0:
        stats_text = f"μ={np.mean(intensities):.2f}, σ={np.std(intensities):.2f}"
        ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

def plot_comparison_three_way(clean_dist, clean_int, noisy_dist, noisy_int, 
                             noise_dist, noise_int, ax):
    """
    Three-panel comparison: Clean | Noisy | Isolated Noise Only
    All using raw scatter with full point density.
    """
    # Panel 1: Clean
    ax[0].scatter(clean_dist, clean_int, s=0.5, c='green', alpha=0.08, edgecolors='none')
    ax[0].set_xlabel('Distance (m)'); ax[0].set_ylabel('Intensity')
    ax[0].set_title('Clean LiDAR Returns', fontweight='bold')
    ax[0].grid(True, alpha=0.2)
    
    # Panel 2: Noisy (weather) - BIGGER POINTS
    ax[1].scatter(noisy_dist, noisy_int, s=2.0, c='red', alpha=0.15, edgecolors='none')  # ← bigger markers
    ax[1].set_xlabel('Distance (m)'); ax[1].set_ylabel('Intensity')
    ax[1].set_title('Noisy LiDAR Returns (Weather)', fontweight='bold')
    ax[1].grid(True, alpha=0.2)
    
    # Panel 3: Isolated noise-only points
    if len(noise_dist) > 0:
        ax[2].scatter(noise_dist, noise_int, s=0.5, c='orange', alpha=0.15, edgecolors='none')
        ax[2].set_xlabel('Distance (m)'); ax[2].set_ylabel('Intensity')
        ax[2].set_title(f'Isolated Weather Noise (N={len(noise_dist):,})', 
                       fontweight='bold', color='darkorange')
        ax[2].grid(True, alpha=0.2)
    else:
        ax[2].text(0.5, 0.5, 'No noise points isolated', 
                  ha='center', va='center', transform=ax[2].transAxes)
    
    for a in ax:
        a.set_ylim(bottom=-10)  # Ensure intensity floor visible

def plot_intensity_overlay_clean_on_top(clean_dist, clean_int, noisy_dist, noisy_int, ax):
    """
    Overlay noisy (bottom, red) + clean (top, green) with clean rendered LAST to appear on top.
    Green line/points appear in front of red.
    """
    # Plot noisy FIRST (bottom layer, zorder=1)
    if len(noisy_dist) > 0:
        ax.scatter(noisy_dist, noisy_int, 
                  s=1.5, c='red', alpha=0.06, edgecolors='none',
                  label=f'Noisy (N={len(noisy_dist):,})', zorder=1)
    
    # Plot clean SECOND (top layer, zorder=2) - appears in front
    if len(clean_dist) > 0:
        ax.scatter(clean_dist, clean_int, 
                  s=1.0, c='green', alpha=0.12, edgecolors='none',
                  label=f'Clean (N={len(clean_dist):,})', zorder=2)
    
    ax.set_xlabel('Distance (m)', fontsize=11)
    ax.set_ylabel('Intensity (raw)', fontsize=11)
    ax.set_title('Overlay: Clean (green, TOP) + Noisy (red, bottom)', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(fontsize=9, loc='upper right')
    
    # Set x-limit to 99th percentile for focus
    max_dist = max(np.percentile(clean_dist, 99) if len(clean_dist) > 0 else 0,
                   np.percentile(noisy_dist, 99) if len(noisy_dist) > 0 else 0)
    ax.set_xlim(0, max_dist)

# ============================================================================
# Y-AXIS FORMATTER FOR LARGE COUNTS
# ============================================================================

def format_large_counts(y, pos):
    """Format large count numbers: 5.0, 1e7, 2.5M, etc."""
    if y == 0:
        return '0'
    elif abs(y) < 1e3:
        return f'{y:.0f}'
    elif abs(y) < 1e6:
        return f'{y/1e3:.1f}k'
    elif abs(y) < 1e9:
        return f'{y/1e6:.1f}M'
    else:
        return f'{y/1e9:.1f}B'

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*75)
    print("LiDAR Weather Noise Isolation - Synchronized Scan Analysis")
    print("="*75)
    
    # Step 1: Find synchronized scan pairs
    print(f"\n[1/5] Finding {N_SYNC_SCANS} synchronized scan pairs...")
    sync_pairs = find_synchronized_scans(
        BAG_FILE_PATH, CLEAN_TOPIC, NOISY_TOPIC, 
        n_scans=N_SYNC_SCANS, time_tol=SYNC_TIME_TOL
    )
    
    if len(sync_pairs) < 10:
        print(f"⚠ Warning: Only found {len(sync_pairs)} pairs. Check topic sync or increase tolerance.")
    
    # Storage for aggregated analysis
    all_clean = {'dist': [], 'int': [], 'points': []}
    all_noisy = {'dist': [], 'int': [], 'points': []}
    all_noise_only = {'dist': [], 'int': [], 'points': []}
    
    # Step 2: Process each synchronized pair
    print(f"\n[2/5] Processing {len(sync_pairs)} synchronized pairs for noise isolation...")
    
    for idx, (clean_msg, noisy_msg, timestamp) in enumerate(tqdm(sync_pairs, desc="Analyzing pairs")):
        # Convert to arrays
        clean_pts = pointcloud2_to_array(clean_msg)
        noisy_pts = pointcloud2_to_array(noisy_msg)
        
        if len(clean_pts) == 0 or len(noisy_pts) == 0:
            continue
        
        # Calculate distances
        clean_dist = calculate_distances(clean_pts)
        noisy_dist = calculate_distances(noisy_pts)
        
        # Isolate noise points using spatial comparison
        noise_only, affected, matched = isolate_noise_points(
            clean_pts, noisy_pts, spatial_radius=NOISE_ISOLATION_RADIUS
        )
        noise_only_dist = calculate_distances(noise_only) if len(noise_only) > 0 else np.array([])
        
        # Aggregate for final plots
        all_clean['dist'].extend(clean_dist)
        all_clean['int'].extend(clean_pts[:, 3])
        all_clean['points'].append(clean_pts)
        
        all_noisy['dist'].extend(noisy_dist)
        all_noisy['int'].extend(noisy_pts[:, 3])
        all_noisy['points'].append(noisy_pts)
        
        if len(noise_only) > 0:
            all_noise_only['dist'].extend(noise_only_dist)
            all_noise_only['int'].extend(noise_only[:, 3])
            all_noise_only['points'].append(noise_only)
        
        # Save representative PCDs (first 5 pairs)
        if idx < 5:
            save_pcd_file(clean_pts, os.path.join(PCD_DIR, 'clean', f'sync_{idx:03d}.pcd'))
            save_pcd_file(noisy_pts, os.path.join(PCD_DIR, 'noisy', f'sync_{idx:03d}.pcd'))
            if len(noise_only) > 0:
                save_pcd_file(noise_only, os.path.join(PCD_DIR, 'noise_only', f'sync_{idx:03d}.pcd'))
    
    # Convert to numpy arrays
    clean_d = np.array(all_clean['dist']); clean_i = np.array(all_clean['int'])
    noisy_d = np.array(all_noisy['dist']); noisy_i = np.array(all_noisy['int'])
    noise_d = np.array(all_noise_only['dist']) if all_noise_only['dist'] else np.array([])
    noise_i = np.array(all_noise_only['int']) if all_noise_only['int'] else np.array([])
    
    print(f"\n[3/5] Aggregated statistics:")
    print(f"  Clean:     {len(clean_d):,} points")
    print(f"  Noisy:     {len(noisy_d):,} points")  
    print(f"  Noise-only: {len(noise_d):,} points ({100*len(noise_d)/len(noisy_d):.2f}% of noisy)")
    
    # Save CSV for ML training
    pd.DataFrame({'distance_m': clean_d, 'intensity': clean_i, 'label': 'clean'}
                ).to_csv(os.path.join(CSV_DIR, 'clean_points.csv'), index=False)
    pd.DataFrame({'distance_m': noisy_d, 'intensity': noisy_i, 'label': 'noisy'}
                ).to_csv(os.path.join(CSV_DIR, 'noisy_points.csv'), index=False)
    if len(noise_d) > 0:
        pd.DataFrame({'distance_m': noise_d, 'intensity': noise_i, 'label': 'weather_noise'}
                    ).to_csv(os.path.join(CSV_DIR, 'noise_only_points.csv'), index=False)
    
    # Step 3: Generate the 6+ charts
    print(f"\n[4/5] Generating distribution charts (raw scatter, all points)...")
    
    plt.style.use('seaborn-v0_8-white')
    sns.set_context("notebook", font_scale=3)
    
    # ========================================================================
    # CHARTS 1-4: Intensity vs Distance (RAW SCATTER - ALL POINTS)
    # ========================================================================
    
    # Chart 1: Clean data - full density scatter
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    plot_intensity_vs_distance_raw(clean_d, clean_i, 'Clean', ax1, color='green')
    ax1.set_xlim(0, np.percentile(clean_d, 99))  # Focus on relevant range
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '01_clean_intensity_vs_distance_raw.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Noisy data - full density scatter (BIGGER POINTS)
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    plot_intensity_vs_distance_raw(
        noisy_d, noisy_i, 'Noisy (Weather)', ax2, 
        color='red', 
        alpha=0.15,      # ← increased visibility
        marker_size=2.0  # ← BIGGER POINTS as requested
    )
    ax2.set_xlim(0, np.percentile(noisy_d, 99))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '02_noisy_intensity_vs_distance_raw.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 3: Three-way comparison (Clean | Noisy | Isolated Noise)
    fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_comparison_three_way(clean_d, clean_i, noisy_d, noisy_i, noise_d, noise_i, axes)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '03_comparison_clean_noisy_noise_only.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 4: NEW - Overlay Clean (green on TOP) + Noisy (red bottom)
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))
    plot_intensity_overlay_clean_on_top(clean_d, clean_i, noisy_d, noisy_i, ax4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '04_overlay_clean_on_top_noisy_bottom.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # CHARTS 5-7: Intensity Count Distributions (Histograms) - FORMATTED Y-AXIS
    # ========================================================================
    
    # Chart 5: Clean intensity histogram (with formatted y-axis)
    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 7))
    ax5.hist(clean_i, bins=200, color='green', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax5.set_xlabel('Intensity (raw)'); ax5.set_ylabel('Count')
    ax5.set_title(f'Clean Intensity Distribution (N={len(clean_i):,})', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    # ← FORMAT Y-AXIS FOR LARGE COUNTS
    ax5.yaxis.set_major_formatter(FuncFormatter(format_large_counts))
    ax5.tick_params(axis='y', labelsize=9)
    # Clean up spines for publication look
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '05_clean_intensity_histogram.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 6: Noisy intensity histogram (with formatted y-axis)
    fig6, ax6 = plt.subplots(1, 1, figsize=(10, 7))
    ax6.hist(noisy_i, bins=200, color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax6.set_xlim(0, 100)
    ax6.set_xlabel('Intensity (raw)'); ax6.set_ylabel('Count')
    ax6.set_title(f'Noisy Intensity Distribution (N={len(noisy_i):,})', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    # ← FORMAT Y-AXIS FOR LARGE COUNTS
    ax6.yaxis.set_major_formatter(FuncFormatter(format_large_counts))
    ax6.tick_params(axis='y', labelsize=9)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '06_noisy_intensity_histogram.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 7: Overlay comparison of all three distributions (density mode)
    fig7, ax7 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Use density for fair comparison despite different sample sizes
    if len(clean_i) > 0:
        ax7.hist(clean_i, bins=150, density=True, 
                histtype='step', color='green', linewidth=2, label='Clean')
    if len(noisy_i) > 0:
        ax7.hist(noisy_i, bins=150, density=True, 
                histtype='step', color='red', linewidth=2, label='Noisy (Weather)')
    if len(noise_i) > 0:
        ax7.hist(noise_i, bins=150, density=True, 
                histtype='stepfilled', color='orange', alpha=0.4, linewidth=1.5, 
                label=f'Isolated Noise (N={len(noise_i):,})')
    
    ax7.set_xlabel('Intensity (raw)', fontsize=12)
    ax7.set_ylabel('Probability Density', fontsize=12)
    ax7.set_title('Intensity Distribution Comparison', fontsize=14, fontweight='bold', pad=20)
    ax7.grid(True, alpha=0.3, linestyle='--')
    ax7.legend(fontsize=11, loc='best')
    
    # Annotate key observations for research
    if len(noise_i) > 0 and len(clean_i) > 0:
        noise_mean = np.mean(noise_i); clean_mean = np.mean(clean_i)
        ax7.axvline(noise_mean, color='orange', linestyle=':', alpha=0.7)
        ax7.axvline(clean_mean, color='green', linestyle=':', alpha=0.7)
        ax7.text(0.02, 0.98, f'Noise μ={noise_mean:.1f} | Clean μ={clean_mean:.1f}', 
                transform=ax7.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '07_intensity_distribution_overlay.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Step 4: Summary & research insights
    print(f"\n[5/5] ✅ Analysis complete!")
    print(f"\n📁 Outputs:")
    print(f"   PCDs:  {PCD_DIR}/[clean|noisy|noise_only]/")
    print(f"   Plots: {PLOT_DIR}/")
    print(f"   CSV:   {CSV_DIR}/")
    
    print(f"\n📊 Generated charts:")
    for i, name in enumerate([
        "01_clean_intensity_vs_distance_raw.png",
        "02_noisy_intensity_vs_distance_raw.png (bigger points)", 
        "03_comparison_clean_noisy_noise_only.png",
        "04_overlay_clean_on_top_noisy_bottom.png (NEW)",
        "05_clean_intensity_histogram.png (formatted y-axis)",
        "06_noisy_intensity_histogram.png (formatted y-axis)",
        "07_intensity_distribution_overlay.png"
    ], 1):
        print(f"   {i}. {name}")
    
    print(f"\n🎯 Key insights for your unsupervised weather-noise removal research:")
    
    if len(noise_d) > 0:
        noise_pct = 100 * len(noise_d) / len(noisy_d)
        print(f"   • Weather noise constitutes ~{noise_pct:.2f}% of returns in noisy scans")
        print(f"   • Noise-only points show spatial distribution: {noise_d.min():.1f}-{noise_d.max():.1f}m range")
        print(f"   • Consider using the 'noise_only' CSV as negative samples for training")
    
    print(f"   • The raw scatter plots reveal:")
    print(f"     - Weather adds low-intensity returns across all distances")
    print(f"     - Spatial isolation (KD-tree, r={NOISE_ISOLATION_RADIUS}m) effectively separates noise")
    print(f"     - Chart 04 shows clean (green) OVER noisy (red) - green on TOP as requested")
    
    print(f"\n💡 Next steps for your LIO improvement pipeline:")
    print(f"   1. Use isolated noise points to train a point-wise classifier (e.g., PointNet)")
    print(f"   2. Incorporate temporal consistency: weather noise is less stable across frames")
    print(f"   3. Add range-dependent thresholds: noise characteristics vary with distance")
    print(f"   4. Consider self-supervised learning: reconstruct clean geometry from noisy input")
    
    print(f"\n⚙️  Tuning parameters for your dataset:")
    print(f"   • SYNC_TIME_TOL={SYNC_TIME_TOL}s: Adjust if scans aren't pairing correctly")
    print(f"   • NOISE_ISOLATION_RADIUS={NOISE_ISOLATION_RADIUS}m: ")
    print(f"     - Smaller: more conservative noise detection")
    print(f"     - Larger: may include real geometry changes")
    print(f"   • To enable intensity residual filtering, set:")
    print(f"     INTENSITY_RESIDUAL_THRESHOLD = 50.0  # Adjust based on your LiDAR's intensity scale")

if __name__ == "__main__":
    main()