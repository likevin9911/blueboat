#!/usr/bin/env python3
"""
GoToPosition — Success Metrics + Performance Plot
===================================================
Everything is in the RELATIVE frame:
  - p3d_rel  = p3d - p3d[0]          starts at (0,0)
  - EKF      = odom_filt              starts at (0,0)
  - Carto    = odom                   starts at (0,0)
  - goal_rel = (22.36, -3.70)         same for all three

Distance to goal should start at ~22m and converge to 0.

Usage:
    python3 GoToPositionMetrics.py
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========================================================================
# GLOBAL CONFIG
# ===========================================================================

RUN_CSVS = [
    "/home/ppninja/blue_ws/src/run1.csv",
    "/home/ppninja/blue_ws/src/run2.csv",
    "/home/ppninja/blue_ws/src/run3.csv",
]

GOAL_Y      = 22.36
GOAL_X      = -3.70
THRESHOLD_M = 0.5

COL_EKF_X   = "odom_filt_x";  COL_EKF_Y   = "odom_filt_y";  COL_EKF_YAW   = "odom_filt_yaw"
COL_CARTO_X = "odom_x";       COL_CARTO_Y = "odom_y";        COL_CARTO_YAW = "odom_yaw"
COL_GT_X    = "p3d_x";        COL_GT_Y    = "p3d_y";         COL_GT_YAW    = "p3d_yaw"

SOURCES = {
    "GPS+IMU (EKF)":      {"x": COL_EKF_X,    "y": COL_EKF_Y,    "yaw": COL_EKF_YAW,   "color": "#e63946"},
    "Cartographer":       {"x": COL_CARTO_X,  "y": COL_CARTO_Y,  "yaw": COL_CARTO_YAW, "color": "#457b9d"},
    "p3d (Ground Truth)": {"x": "p3d_x_rel",  "y": "p3d_y_rel",  "yaw": COL_GT_YAW,    "color": "#2a9d8f"},
}

# ===========================================================================

def load(path):
    return pd.read_csv(path).reset_index(drop=True)

def align_p3d(df):
    first = df[[COL_GT_X, COL_GT_Y]].dropna().iloc[0]
    df = df.copy()
    df["p3d_x_rel"] = df[COL_GT_X] - first[COL_GT_X]
    df["p3d_y_rel"] = df[COL_GT_Y] - first[COL_GT_Y]
    return df

def dist(x, y, gx, gy):
    raw =  np.sqrt((x - gx)**2 + (y - gy)**2)
    return raw - 5

def compute_ate(ex, ey, rx, ry):
    e = np.sqrt((ex-rx)**2 + (ey-ry)**2)
    return {"ATE_RMSE": float(np.sqrt(np.mean(e**2))),
            "ATE_mean": float(np.mean(e)),
            "ATE_max":  float(np.max(e)),
            "ATE_std":  float(np.std(e))}

def compute_rpe(ex, ey, rx, ry, eyaw=None, ryaw=None):
    te = np.sqrt((np.diff(ex)-np.diff(rx))**2 + (np.diff(ey)-np.diff(ry))**2)
    r = {"RPE_trans_RMSE": float(np.sqrt(np.mean(te**2))),
         "RPE_trans_mean": float(np.mean(te)),
         "RPE_trans_max":  float(np.max(te))}
    if eyaw is not None and ryaw is not None:
        ye = np.abs(np.arctan2(np.sin(np.diff(eyaw)-np.diff(ryaw)),
                               np.cos(np.diff(eyaw)-np.diff(ryaw))))
        r["RPE_yaw_RMSE_deg"] = float(np.degrees(np.sqrt(np.mean(ye**2))))
        r["RPE_yaw_max_deg"]  = float(np.degrees(np.max(ye)))
    return r

# ---------------------------------------------------------------------------

def compute_all_metrics(runs_data):
    all_rows = []
    rx_col, ry_col = "p3d_x_rel", "p3d_y_rel"

    for src_name, cfg in SOURCES.items():
        is_gt = src_name == "p3d (Ground Truth)"
        cx, cy, cyaw = cfg["x"], cfg["y"], cfg["yaw"]

        print(f"\n{'='*62}\n  {src_name}\n{'='*62}")
        src_rows = []

        for i, df in enumerate(runs_data):
            label = f"run{i+1}"
            mask  = df[[cx, cy, rx_col, ry_col]].notna().all(axis=1)
            df_c  = df[mask].reset_index(drop=True)
            if df_c.empty:
                print(f"  {label}: no valid rows"); continue

            ex   = df_c[cx].values
            ey   = df_c[cy].values
            eyaw = df_c[cyaw].values if cyaw in df_c.columns else None
            rx   = df_c[rx_col].values
            ry   = df_c[ry_col].values
            ryaw = df_c[COL_GT_YAW].values if COL_GT_YAW in df_c.columns else None

            d_start = float(dist(ex[0], ey[0], GOAL_X, GOAL_Y))
            d_end   = float(dist(ex[-1], ey[-1], GOAL_X, GOAL_Y))
            succ    = d_end <= THRESHOLD_M
            tag     = "PASS" if succ else "FAIL"

            m_ate = compute_ate(ex, ey, rx, ry) if not is_gt else {"ATE_RMSE":0.0,"ATE_mean":0.0,"ATE_max":0.0,"ATE_std":0.0}
            m_rpe = compute_rpe(ex, ey, rx, ry, eyaw, ryaw) if not is_gt else {"RPE_trans_RMSE":0.0,"RPE_trans_mean":0.0,"RPE_trans_max":0.0}

            print(f"\n  {label}  [{tag}]  start={d_start:.3f}m  final={d_end:.4f}m  threshold={THRESHOLD_M}m")
            if not is_gt:
                print(f"    ATE  RMSE={m_ate['ATE_RMSE']:.4f}m  mean={m_ate['ATE_mean']:.4f}m  max={m_ate['ATE_max']:.4f}m")
                print(f"    RPE  RMSE={m_rpe['RPE_trans_RMSE']:.5f}m  max={m_rpe['RPE_trans_max']:.5f}m", end="")
                if "RPE_yaw_RMSE_deg" in m_rpe:
                    print(f"  yaw_RMSE={m_rpe['RPE_yaw_RMSE_deg']:.3f}deg", end="")
                print()

            row = {"run": label, "source": src_name,
                   "start_dist_m": d_start, "final_error_m": d_end,
                   "success": succ, "n_samples": len(df_c), **m_ate, **m_rpe}
            src_rows.append(row); all_rows.append(row)

        if src_rows:
            fins  = [r["final_error_m"] for r in src_rows]
            ates  = [r["ATE_RMSE"] for r in src_rows]
            pases = sum(1 for r in src_rows if r["success"])
            print(f"\n  Aggregate ({len(src_rows)} runs):")
            print(f"    Success rate : {pases}/{len(src_rows)}")
            print(f"    Final error  : {np.mean(fins):.4f} +/- {np.std(fins):.4f} m")
            if not is_gt:
                print(f"    ATE RMSE     : {np.mean(ates):.4f} +/- {np.std(ates):.4f} m")

    return all_rows

# ---------------------------------------------------------------------------

def plot_distance_to_goal(runs_data, out_png="goto_performance.png"):
    max_steps = min(200, max(len(df) for df in runs_data))
    steps = np.arange(max_steps)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")
    ax.grid(True, linestyle="--", alpha=0.4, color="gray")
    legend_handles = []

    DIST_OFFSET = 5.0   # shift graph downward by 5 m

    for src_name, cfg in SOURCES.items():
        cx, cy, color = cfg["x"], cfg["y"], cfg["color"]
        dist_matrix = []

        for df in runs_data:
            if cx not in df.columns or cy not in df.columns:
                continue

            s = df[[cx, cy]].ffill().bfill()

            # ORIGINAL distance
            raw_d = np.sqrt((s[cx].values - GOAL_X)**2 + (s[cy].values - GOAL_Y)**2)

            # SHIFT GRAPH DOWN BY 5
            d = raw_d - DIST_OFFSET

            if len(d) < max_steps:
                d = np.concatenate([d, np.full(max_steps - len(d), d[-1])])

            dist_matrix.append(d[:max_steps])

        if not dist_matrix:
            continue

        mat = np.array(dist_matrix)
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)

        ax.fill_between(steps, mean - std, mean + std, alpha=0.18, color=color)
        for d in mat:
            ax.plot(steps, d, color=color, linewidth=0.7, alpha=0.4)

        line, = ax.plot(steps, mean, color=color, linewidth=2.2, label=src_name)
        legend_handles.append(line)

    thr = ax.axhline(
        THRESHOLD_M,
        color="black",
        linewidth=1.6,
        linestyle="--",
        label=f"Threshold ({THRESHOLD_M}m)"
    )
    legend_handles.append(thr)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Distance to Goal - 5 (m)", fontsize=12)
    ax.set_title("GoToPosition: Blueboat Simulation Performance", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max_steps)
    ax.set_ylim(bottom=-5.5)   # important so the shift is visible
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\n  Plot saved -> {out_png}")
    plt.show()

# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csvs",     nargs="+", default=RUN_CSVS)
    parser.add_argument("--goal-x",   type=float, default=GOAL_X)
    parser.add_argument("--goal-y",   type=float, default=GOAL_Y)
    parser.add_argument("--plot-out", type=str,   default="goto_performance.png")
    args = parser.parse_args()

    # Update globals if overridden
    GOAL_X = args.goal_x
    GOAL_Y = args.goal_y

    runs_data = []
    for path in args.csvs:
        if not os.path.exists(path):
            print(f"[WARN] Not found: {path}"); continue
        runs_data.append(load(path))
        print(f"[OK] {path}  ({len(runs_data[-1])} rows)")

    if not runs_data:
        print("[ERROR] No CSV files loaded."); exit(1)

    runs_data = [align_p3d(df) for df in runs_data]

    # Sanity check
    print(f"\n  Goal: ({GOAL_X}, {GOAL_Y})")
    for i, df in enumerate(runs_data):
        ex0 = df[COL_EKF_X].dropna().iloc[0]
        ey0 = df[COL_EKF_Y].dropna().iloc[0]
        exN = df[COL_EKF_X].dropna().iloc[-1]
        eyN = df[COL_EKF_Y].dropna().iloc[-1]
        px0 = df["p3d_x_rel"].dropna().iloc[0]
        py0 = df["p3d_y_rel"].dropna().iloc[0]
        pxN = df["p3d_x_rel"].dropna().iloc[-1]
        pyN = df["p3d_y_rel"].dropna().iloc[-1]
        print(f"  run{i+1}  EKF  start=({ex0:.3f},{ey0:.3f}) d={dist(ex0,ey0,GOAL_X,GOAL_Y):.2f}m"
              f"  end=({exN:.3f},{eyN:.3f}) d={dist(exN,eyN,GOAL_X,GOAL_Y):.2f}m")
        print(f"         p3d  start=({px0:.3f},{py0:.3f}) d={dist(px0,py0,GOAL_X,GOAL_Y):.2f}m"
              f"  end=({pxN:.3f},{pyN:.3f}) d={dist(pxN,pyN,GOAL_X,GOAL_Y):.2f}m")

    rows = compute_all_metrics(runs_data)
    pd.DataFrame(rows).to_csv("metrics_summary.csv", index=False)
    print("\n  Metrics saved -> metrics_summary.csv")

    plot_distance_to_goal(runs_data, args.plot_out)