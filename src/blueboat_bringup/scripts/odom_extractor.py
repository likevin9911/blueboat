#!/usr/bin/env python3
import os
import math
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import rosbag
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helpers: Math & Transformations
# -----------------------------
def quaternion_to_yaw(x, y, z, w) -> float:
    """Convert quaternion to yaw (psi) in radians. Z-up rotation."""
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

# -----------------------------
# Helpers: Extraction + Reading
# -----------------------------
def extract_pose_semantic(msg) -> Optional[Dict]:
    """Extract x, y, yaw, and frame_id from common message types."""
    data = {"x": None, "y": None, "yaw": None, "frame_id": None}
    
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        try: data['frame_id'] = msg.header.frame_id
        except: pass
    elif hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
        pos = msg.pose.position
        ori = msg.pose.orientation
        try: data['frame_id'] = msg.header.frame_id
        except: pass
    elif hasattr(msg, 'position'):
        pos = msg.position
        ori = getattr(msg, 'orientation', None)
    else:
        return None

    try:
        data['x'] = float(pos.x)
        data['y'] = float(pos.y)
    except: return None

    if ori is not None:
        try: data['yaw'] = quaternion_to_yaw(ori.x, ori.y, ori.z, ori.w)
        except: data['yaw'] = 0.0
    
    return data

def read_pose_time(bag: rosbag.Bag, topic: str) -> Dict:
    """Read (t, x, y, yaw, frame_id) from a topic."""
    ts, xs, ys, yaws, frames = [], [], [], [], []
    for _topic, msg, t in bag.read_messages(topics=[topic]):
        data = extract_pose_semantic(msg)
        if data is None or data['x'] is None: continue
        if not (math.isfinite(data['x']) and math.isfinite(data['y'])): continue
        
        ts.append(t.to_sec())
        xs.append(data['x'])
        ys.append(data['y'])
        yaws.append(data['yaw'] if data['yaw'] is not None else 0.0)
        frames.append(data['frame_id'] if data['frame_id'] else "unknown")

    if len(ts) == 0:
        return {"t": np.array([]), "x": np.array([]), "y": np.array([]), "yaw": np.array([]), "frame": ""}

    ts = np.asarray(ts, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    yaws = np.asarray(yaws, dtype=float)
    
    order = np.argsort(ts)
    unique_frames = np.unique(np.asarray(frames)[order])
    dominant_frame = unique_frames[0] if len(unique_frames) > 0 else "unknown"
    
    if len(unique_frames) > 1:
        print(f"[WARN] Topic {topic} has mixed frame_ids: {unique_frames}. Using '{dominant_frame}'.")

    return {
        "t": ts[order], "x": xs[order], "y": ys[order], "yaw": yaws[order],
        "frame": dominant_frame
    }

def zero_start(xs: np.ndarray, ys: np.ndarray, yaws: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xs.size == 0: return xs, ys, yaws
    return xs - xs[0], ys - ys[0], yaws - yaws[0]

def interp_to(t_src: np.ndarray, src_vals: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    if t_src.size == 0: return np.array([])
    return np.interp(t_tgt, t_src, src_vals)

# -----------------------------
# SE(2) Alignment (Position Only)
# -----------------------------
def se2_align_xy(est_xy: np.ndarray, gt_xy: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Align estimate to GT using least-squares SE(2) on position only."""
    est_mean = est_xy.mean(axis=0)
    gt_mean = gt_xy.mean(axis=0)
    E = est_xy - est_mean
    G = gt_xy - gt_mean

    H = E.T @ G
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = gt_mean - (R @ est_mean)
    aligned = (est_xy @ R.T) + t
    theta = math.atan2(R[1, 0], R[0, 0])
    return aligned, theta, t

# -----------------------------
# Metrics: Position & Orientation (COMPLETE)
# -----------------------------
def compute_ate_rpe_complete(est_xy: np.ndarray, gt_xy: np.ndarray, 
                             est_yaw: np.ndarray, gt_yaw: np.ndarray, 
                             seg_len_m: float = 1.0):
    """
    Compute ALL metrics: position (ATE, RMSE, drift, drift/m, %, RPE) + orientation.
    Returns dict with all original + new metrics.
    """
    # === POSITION METRICS (Original Script) ===
    diff = est_xy - gt_xy
    e_pos = np.linalg.norm(diff, axis=1)
    
    ate = float(np.mean(e_pos))
    rmse = float(np.sqrt(np.mean(e_pos ** 2)))
    drift = float(e_pos[-1]) if len(e_pos) > 0 else 0.0
    
    # Drift per meter (normalized by path length)
    gt_len = path_length(gt_xy)
    drift_per_m = (drift / gt_len) if gt_len > 1e-9 else float("nan")
    drift_pct_per_m = 100.0 * drift_per_m if math.isfinite(drift_per_m) else float("nan")
    
    # === RPE METRICS (Original Script) ===
    n = gt_xy.shape[0]
    rpe_rmse_m = float("nan")
    rpe_mean_m = float("nan")
    rpe_rmse_per_m = float("nan")
    rpe_mean_per_m = float("nan")
    rpe_count = 0
    
    if n > 2:
        d = np.zeros(n, dtype=float)
        d[1:] = np.cumsum(np.linalg.norm(gt_xy[1:] - gt_xy[:-1], axis=1))
        errs = []
        for i in range(n - 1):
            target = d[i] + seg_len_m
            j = int(np.searchsorted(d, target))
            if j <= i or j >= n: break
            d_gt = gt_xy[j] - gt_xy[i]
            d_est = est_xy[j] - est_xy[i]
            errs.append(float(np.linalg.norm(d_est - d_gt)))
        
        if len(errs) > 0:
            errs = np.asarray(errs, dtype=float)
            rpe_rmse_m = float(np.sqrt(np.mean(errs ** 2)))
            rpe_mean_m = float(np.mean(errs))
            rpe_rmse_per_m = rpe_rmse_m / seg_len_m
            rpe_mean_per_m = rpe_mean_m / seg_len_m
            rpe_count = len(errs)
    
    # === ORIENTATION METRICS (New) ===
    e_yaw_raw = est_yaw - gt_yaw
    e_yaw = np.array([normalize_angle(a) for a in e_yaw_raw])
    yaw_ate = float(np.mean(np.abs(e_yaw)))
    yaw_rmse = float(np.sqrt(np.mean(e_yaw ** 2)))
    yaw_drift = float(e_yaw[-1]) if len(e_yaw) > 0 else 0.0
    
    return {
        # Position (original)
        "ate": ate, "rmse": rmse, "drift": drift,
        "drift_per_m": drift_per_m, "drift_pct_per_m": drift_pct_per_m,
        "rpe_rmse_m": rpe_rmse_m, "rpe_mean_m": rpe_mean_m,
        "rpe_rmse_per_m": rpe_rmse_per_m, "rpe_mean_per_m": rpe_mean_per_m,
        "rpe_count": rpe_count,
        # Orientation (new)
        "yaw_ate": yaw_ate, "yaw_rmse": yaw_rmse, "yaw_drift": yaw_drift,
        # Vectors for plotting
        "pos_err_vec": e_pos, "yaw_err_vec": e_yaw
    }

def path_length(xy: np.ndarray) -> float:
    if xy.shape[0] < 2: return 0.0
    return float(np.sum(np.linalg.norm(xy[1:] - xy[:-1], axis=1)))

# -----------------------------
# Frame Map Parsing
# -----------------------------
def parse_frame_map(frame_map_str: str) -> Dict[str, str]:
    """Parse --frame_map argument: 'odom:/Odometry,map:/p3d/groundtruth'"""
    frame_map = {}
    if not frame_map_str: return frame_map
    for item in frame_map_str.split(','):
        if ':' not in item: continue
        frame_type, topic = item.split(':', 1)
        frame_map[topic.strip()] = frame_type.strip()
    return frame_map

# -----------------------------
# Main
# -----------------------------
def main(args):
    os.makedirs(args.out, exist_ok=True)

    print(f"Reading bag: {args.bag}")
    print(f"Topics: {args.topics}")
    print(f"Groundtruth topic: {args.gt_topic}")
    print(f"Frame map: {args.frame_map or 'auto-detect from headers'}")
    print(f"SE(2) alignment: {'ON' if args.se2_align else 'OFF'}")
    print(f"RPE segment length: {args.rpe_segment_m} m")
    
    frame_map = parse_frame_map(args.frame_map) if args.frame_map else {}
    
    # Read all trajectories
    traj: Dict[str, Dict] = {}
    with rosbag.Bag(args.bag, "r") as bag:
        for topic in args.topics:
            data = read_pose_time(bag, topic)
            traj[topic] = data
            if topic in frame_map:
                traj[topic]["frame"] = frame_map[topic]
                print(f"[Frame override] {topic} -> {frame_map[topic]}")

    if args.gt_topic not in traj or traj[args.gt_topic]["t"].size == 0:
        raise RuntimeError(f"No groundtruth data found for {args.gt_topic}")

    # --- FRAME SEMANTICS CHECK ---
    gt_frame = traj[args.gt_topic]["frame"]
    gt_frame_type = frame_map.get(args.gt_topic, gt_frame)
    print(f"\n[Frame Semantics Check]")
    print(f"  GT: {args.gt_topic} -> frame_id='{gt_frame}' (type: {gt_frame_type})")
    
    frame_groups = defaultdict(list)
    for topic in args.topics:
        if topic == args.gt_topic: continue
        est_frame = traj[topic]["frame"]
        est_frame_type = frame_map.get(topic, est_frame)
        frame_groups[est_frame_type].append(topic)
        print(f"  {topic} -> frame_id='{est_frame}' (type: {est_frame_type})")
        
    print(f"\n  Grouped by frame type:")
    for ftype, topics in frame_groups.items():
        print(f"    {ftype}: {topics}")
    
    if len(frame_groups) > 1:
        print(f"\n  [WARNING] Comparing topics from different frame types: {list(frame_groups.keys())}")
        print(f"            SE(2) alignment will remove global offsets, but interpret drift metrics cautiously.")
    print("-" * 50)

    # Zero Start
    for topic in args.topics:
        x, y, yaw = zero_start(traj[topic]["x"], traj[topic]["y"], traj[topic]["yaw"])
        traj[topic]["x0"] = x
        traj[topic]["y0"] = y
        traj[topic]["yaw0"] = yaw

    # Setup GT
    gt_t = traj[args.gt_topic]["t"]
    gt_xy = np.vstack([traj[args.gt_topic]["x0"], traj[args.gt_topic]["y0"]]).T
    gt_yaw = traj[args.gt_topic]["yaw0"]
    gt_len = path_length(gt_xy)
    print(f"GT path length: {gt_len:.3f} m")

    # --- Plotting ---
    if args.plot_aligned:
        plt.figure(figsize=(10, 8))
        plt.plot(gt_xy[:, 0], gt_xy[:, 1], linestyle=":", linewidth=2.5, alpha=0.95, 
                 label=f"GT: {args.gt_topic}\n({gt_frame_type})", color='black')
        
        frame_styles = {
            "map": {"color": "green", "ls": "-."},
            "odom": {"color": "blue", "ls": "--"},
            "camera_init": {"color": "blue", "ls": "--"},
            "base_link": {"color": "red", "ls": "-"},
            "unknown": {"color": "gray", "ls": ":"}
        }
        
        for topic in args.topics:
            if topic == args.gt_topic: continue
            if traj[topic]["t"].size == 0: continue
            
            est_frame_type = frame_map.get(topic, traj[topic]["frame"])
            style = frame_styles.get(est_frame_type, frame_styles["unknown"])
            
            est_xi = interp_to(traj[topic]["t"], traj[topic]["x0"], gt_t)
            est_yi = interp_to(traj[topic]["t"], traj[topic]["y0"], gt_t)
            est_xy = np.vstack([est_xi, est_yi]).T
            
            if args.se2_align:
                est_xy, _, _ = se2_align_xy(est_xy, gt_xy)
            
            plt.plot(est_xy[:, 0], est_xy[:, 1], 
                     linestyle=style["ls"], color=style["color"],
                     linewidth=2.0, alpha=0.85, 
                     label=f"{topic}\n({est_frame_type})")
            
        plt.title("XY Trajectories (Interpolated to GT time; SE(2)-aligned if enabled)")
        plt.xlabel("x (m)"); plt.ylabel("y (m)")
        plt.axis("equal"); plt.grid(True, alpha=0.5, linestyle=":")
        plt.legend(loc="best", fontsize=8)
        out_plot = os.path.join(args.out, "traj_aligned_frames.png")
        plt.tight_layout()
        plt.savefig(out_plot, dpi=200)
        print(f"Saved plot: {out_plot}")

    # --- Metrics & CSV (COMPLETE) ---
    all_rows = []
    metrics_lines = [
        f"Bag: {args.bag}", 
        f"GT topic: {args.gt_topic} (frame: {gt_frame_type})", 
        f"Frame map: {args.frame_map or 'auto'}",
        f"SE2 Align: {args.se2_align}", 
        f"RPE segment: {args.rpe_segment_m} m",
        f"GT path length: {gt_len:.6f} m",
        ""
    ]

    for topic in args.topics:
        if topic == args.gt_topic: continue
        if traj[topic]["t"].size == 0: 
            print(f"[WARN] No data for {topic}")
            metrics_lines.append(f"{topic}: NO DATA\n")
            continue

        # Interpolate
        est_xi = interp_to(traj[topic]["t"], traj[topic]["x0"], gt_t)
        est_yi = interp_to(traj[topic]["t"], traj[topic]["y0"], gt_t)
        est_yawi = interp_to(traj[topic]["t"], traj[topic]["yaw0"], gt_t)
        
        est_xy = np.vstack([est_xi, est_yi]).T
        
        # Align
        theta, trans = 0.0, np.array([0.0, 0.0])
        if args.se2_align:
            est_xy, theta, trans = se2_align_xy(est_xy, gt_xy)
            est_yawi = est_yawi + theta 
            est_yawi = np.array([normalize_angle(a) for a in est_yawi])

        metrics = compute_ate_rpe_complete(est_xy, gt_xy, est_yawi, gt_yaw, seg_len_m=args.rpe_segment_m)

        # Print (RESTORED ORIGINAL FORMAT + ORIENTATION)
        est_frame_type = frame_map.get(topic, traj[topic]["frame"])
        print(f"\n--- vs Groundtruth ---")
        print(f"Topic: {topic} (frame: {est_frame_type})")
        if args.se2_align:
            print(f"  SE2 theta(rad)={theta:.6f}, t=({trans[0]:.6f},{trans[1]:.6f})")
        print(f"  ATE (mean)            : {metrics['ate']:.4f} m")
        print(f"  RMSE                  : {metrics['rmse']:.4f} m")  # ← RESTORED
        print(f"  Final drift           : {metrics['drift']:.4f} m")
        print(f"  Drift per meter       : {metrics['drift_per_m']:.6f} m/m")  # ← RESTORED
        print(f"  % drift per meter     : {metrics['drift_pct_per_m']:.4f} %/m")  # ← RESTORED
        print(f"  RPE trans RMSE        : {metrics['rpe_rmse_m']:.4f} m   (N={metrics['rpe_count']}, seg={args.rpe_segment_m}m)")  # ← RESTORED
        print(f"  RPE trans mean        : {metrics['rpe_mean_m']:.4f} m")  # ← RESTORED
        print(f"  RPE RMSE (m per m)    : {metrics['rpe_rmse_per_m']:.6f} m/m")  # ← RESTORED
        print(f"  RPE mean (m per m)    : {metrics['rpe_mean_per_m']:.6f} m/m")  # ← RESTORED
        # Orientation (new)
        print(f"  Yaw ATE (mean)        : {np.degrees(metrics['yaw_ate']):.2f} deg")
        print(f"  Yaw RMSE              : {np.degrees(metrics['yaw_rmse']):.2f} deg")
        print(f"  Yaw final drift       : {np.degrees(metrics['yaw_drift']):.2f} deg")

        # Optional error-over-time plot
        if args.save_error_plots:
            safe_name = topic.strip("/").replace("/", "__")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(gt_t - gt_t[0], metrics['pos_err_vec'], linewidth=2.0, color='blue')
            ax1.set_ylabel('Position Error (m)')
            ax1.grid(True, linestyle=":", alpha=0.7)
            ax2.plot(gt_t - gt_t[0], np.degrees(metrics['yaw_err_vec']), linewidth=2.0, color='red')
            ax2.set_ylabel('Yaw Error (deg)')
            ax2.set_xlabel('Time since start (s)')
            ax2.grid(True, linestyle=":", alpha=0.7)
            plt.suptitle(f'Error vs. Time: {topic}')
            out_err_plot = os.path.join(args.out, f"{safe_name}_error_vs_time.png")
            plt.tight_layout()
            plt.savefig(out_err_plot, dpi=200)
            print(f"  Saved error plot      : {out_err_plot}")

        # CSV Rows (ALL METRICS)
        t0 = gt_t[0]
        for i in range(gt_t.size):
            all_rows.append([
                gt_t[i] - t0, topic, est_frame_type,
                gt_xy[i,0], gt_xy[i,1], gt_yaw[i],
                est_xy[i,0], est_xy[i,1], est_yawi[i],
                metrics['pos_err_vec'][i], metrics['yaw_err_vec'][i],
                # Position metrics (original)
                metrics['ate'], metrics['rmse'], metrics['drift'],
                metrics['drift_per_m'], metrics['drift_pct_per_m'],
                metrics['rpe_rmse_m'], metrics['rpe_mean_m'],
                metrics['rpe_rmse_per_m'], metrics['rpe_mean_per_m'],
                metrics['rpe_count'],
                # Orientation metrics (new)
                metrics['yaw_ate'], metrics['yaw_rmse'], metrics['yaw_drift'],
                # SE2 params
                theta, trans[0], trans[1]
            ])
        
        # Metrics text file (RESTORED ORIGINAL FORMAT)
        metrics_lines.append(f"Topic: {topic} (frame: {est_frame_type})")
        if args.se2_align:
            metrics_lines.append(f"  SE2 theta(rad)={theta:.6f}, t=({trans[0]:.6f},{trans[1]:.6f})")
        metrics_lines.append(f"  ATE_mean_m: {metrics['ate']:.8f}")
        metrics_lines.append(f"  RMSE_m: {metrics['rmse']:.8f}")  # ← RESTORED
        metrics_lines.append(f"  Final_drift_m: {metrics['drift']:.8f}")
        metrics_lines.append(f"  Drift_per_meter_m_per_m: {metrics['drift_per_m']:.12f}")  # ← RESTORED
        metrics_lines.append(f"  Drift_pct_per_meter_pct_per_m: {metrics['drift_pct_per_m']:.12f}")  # ← RESTORED
        metrics_lines.append(f"  RPE_trans_RMSE_m: {metrics['rpe_rmse_m']:.8f}")  # ← RESTORED
        metrics_lines.append(f"  RPE_trans_mean_m: {metrics['rpe_mean_m']:.8f}")  # ← RESTORED
        metrics_lines.append(f"  RPE_RMSE_m_per_m: {metrics['rpe_rmse_per_m']:.12f}")  # ← RESTORED
        metrics_lines.append(f"  RPE_mean_m_per_m: {metrics['rpe_mean_per_m']:.12f}")  # ← RESTORED
        metrics_lines.append(f"  RPE_count: {metrics['rpe_count']}")
        # Orientation
        metrics_lines.append(f"  Yaw_ATE_deg: {np.degrees(metrics['yaw_ate']):.8f}")
        metrics_lines.append(f"  Yaw_RMSE_deg: {np.degrees(metrics['yaw_rmse']):.8f}")
        metrics_lines.append(f"  Yaw_drift_deg: {np.degrees(metrics['yaw_drift']):.8f}")
        metrics_lines.append("")

    # Save CSV (ALL COLUMNS)
    csv_path = os.path.join(args.out, "results_complete.csv")
    with open(csv_path, "w") as f:
        f.write(
            "t_rel,topic,frame_type,"
            "gt_x,gt_y,gt_yaw,est_x,est_y,est_yaw,"
            "err_pos,err_yaw,"
            # Position metrics
            "ATE_mean,RMSE,final_drift,drift_per_m,drift_pct_per_m,"
            "RPE_rmse_m,RPE_mean_m,RPE_rmse_m_per_m,RPE_mean_m_per_m,RPE_count,"
            # Orientation metrics
            "Yaw_ATE_deg,Yaw_RMSE_deg,Yaw_drift_deg,"
            # SE2 params
            "se2_theta,se2_tx,se2_ty\n"
        )
        for r in all_rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"\nSaved consolidated CSV: {csv_path}")

    # Save metrics summary
    metrics_path = os.path.join(args.out, "metrics_complete.txt")
    with open(metrics_path, "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
    print(f"Saved metrics summary: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIO evaluation: complete metrics + frame semantics + orientation")
    parser.add_argument("--bag", default="/home/ppninja/bags/alg/lrain_staticxl_lio_nw.bag")
    parser.add_argument("--topics", nargs="+", default=[
        "/p3d/groundtruth",
        "/Odometry", 
        "/Odometry_noisy",
        "/blueboat/robot_localization/odometry/filtered",
    ])
    parser.add_argument("--out", default="logged_data")
    parser.add_argument("--gt_topic", default="/p3d/groundtruth")
    parser.add_argument("--frame_map", type=str, default="", 
                        help="Comma-separated frame:type pairs, e.g., 'map:/p3d/groundtruth,odom:/Odometry'")
    parser.add_argument("--se2_align", action="store_true")
    parser.add_argument("--plot_aligned", action="store_true")
    parser.add_argument("--rpe_segment_m", type=float, default=1.0)
    parser.add_argument("--save_error_plots", action="store_true", help="Save position+yaw error vs time plots")
    args = parser.parse_args()
    main(args)