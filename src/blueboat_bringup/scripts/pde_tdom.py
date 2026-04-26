#!/usr/bin/env python3
"""
Wave Height & Period from IMU /imu/data (ROS bag) with denoising comparisons

Implements (and contrasts) three preprocessing pipelines before frequency-domain
double integration to displacement:
  1) Butterworth low-pass filter
  2) Wavelet soft-threshold denoising
  3) Anisotropic diffusion (Perona–Malik) on 1D signal (recommended by paper)

Then performs:
  - Geographic vertical acceleration reconstruction from IMU (acc + quat)
  - Frequency-domain double integration with low-freq trimming
  - Zero-upcrossing wave-by-wave analysis to get wave heights & periods
  - Summary metrics + plots for comparison

USAGE (ROS1 bag example):
  python imu_waves_pde_vs_filters.py --bag path/to/file.bag --topic /imu/data --rate 50

ALTERNATIVES:
  - CSV with columns: time, ax, ay, az, qw, qx, qy, qz
    python imu_waves_pde_vs_filters.py --csv path/to/imu.csv --rate 50

NOTES:
  - This script tries to read ROS1 bags via `rosbag` if available.
  - For ROS2 bags, export the IMU stream to CSV first (e.g., ros2bag->CSV) or
    extend `load_imu_from_rosbag` with rosbag2_py (not guaranteed available).
"""

import argparse
import sys
import os
import math
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Optional deps
try:
    import rosbag  # ROS1
except Exception:
    rosbag = None

try:
    import pywt
except Exception:
    pywt = None

from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft, rfftfreq

GRAVITY = 9.80665  # m/s^2

# ------------------------------- Utilities ---------------------------------

def quat_to_euler(qw, qx, qy, qz):
    """Quaternion (w, x, y, z) -> roll (gamma), pitch (theta), yaw (phi) in radians.
    Uses aerospace sequence (Z-Y-X) consistent with paper's Rz(phi) Ry(theta) Rx(gamma).
    """
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw*qy - qz*qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)  # use 90 deg if out of range
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw  # gamma, theta, phi


def vertical_acc_from_imu(ax, ay, az, qw, qx, qy, qz):
    """Compute geographic-vertical acceleration Ã_Z per paper Eq. (6):
       ÃZ = − sinθ * ax + sinγ cosθ * ay + cosγ cosθ * az − g
       where γ=roll, θ=pitch, ϕ=yaw from quaternion.
    ax,ay,az are in sensor frame (m/s^2). Returns Atilde_Z (m/s^2).
    """
    gamma, theta, phi = quat_to_euler(qw, qx, qy, qz)
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    sin_g  = math.sin(gamma)
    cos_g  = math.cos(gamma)
    return (-sin_th * ax) + (sin_g * cos_th * ay) + (cos_g * cos_th * az) - GRAVITY


def butter_lowpass(sig, fs, fc, order=4):
    """Zero-phase Butterworth low-pass (for comparison)."""
    nyq = 0.5 * fs
    wc = min(fc/nyq, 0.999)
    b, a = butter(order, wc, btype='low')
    return filtfilt(b, a, sig)


def wavelet_denoise(sig, wavelet='db6', level=None, mode='soft'):
    """Wavelet soft-threshold denoise (Donoho). Requires pywt."""
    if pywt is None:
        raise RuntimeError("pywt not installed. Install PyWavelets to use wavelet denoise.")
    if level is None:
        level = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # Universal threshold sigma * sqrt(2 log N)
    detail_coeffs = coeffs[1:]
    sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(sig)))
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode=mode) for c in detail_coeffs]
    return pywt.waverec(new_coeffs, wavelet)[:len(sig)]


def anisotropic_diffusion_1d(sig, n_iter=50, k=None, lam=0.2, use_g2=True):
    """
    1D Perona–Malik anisotropic diffusion for time series.

    PDE: ∂u/∂t = d/dx ( g(|∂u/∂x|) * ∂u/∂x )
    g(s) = 1 / (1 + (s/K)^2)   [use_g2=True]   (Perona-Malik Eq. 8, option 2)
         = exp(-(s/K)^2)       [use_g2=False]  (option 1)

    Args:
      sig: 1D np.array
      n_iter: iterations
      k: gradient threshold K; if None, set to 1.4826 * MAD of first differences (Canny-esque)
      lam: time step (stability ~ <= 0.25 for this 1D scheme)
      use_g2: choose g2 vs exp

    Returns:
      denoised np.array of same length
    """
    u = sig.astype(float).copy()
    # Estimate K from gradient statistics (robust)
    if k is None:
        grad = np.diff(u, prepend=u[0])
        mad = np.median(np.abs(grad - np.median(grad)))
        k = 1.4826 * mad if mad > 1e-12 else np.std(grad) + 1e-6

    for _ in range(n_iter):
        # Forward/backward differences
        ux_f = np.roll(u, -1) - u
        ux_b = u - np.roll(u, 1)

        # Conductance
        if use_g2:
            c_f = 1.0 / (1.0 + (ux_f/k)**2)
            c_b = 1.0 / (1.0 + (ux_b/k)**2)
        else:
            c_f = np.exp(- (ux_f/k)**2)
            c_b = np.exp(- (ux_b/k)**2)

        # Divergence term (Neumann boundary)
        div = c_f * ux_f - c_b * ux_b
        u += lam * div
    return u


def freq_double_integrate(acc_z, fs, f_hp=0.03):
    """
    Frequency-domain double integration:
      - FFT(acc_z)
      - Zero out very low frequencies (< f_hp) to remove trend/bias
      - Divide by -(2πf)^2 to get displacement spectrum
      - IFFT to time domain

    Returns displacement (meters) with zero-mean.
    """
    n = len(acc_z)
    freqs = rfftfreq(n, d=1.0/fs)
    A = rfft(acc_z)

    # High-pass like trimming near DC
    hp_mask = freqs >= max(f_hp, 1.0/(len(acc_z)/fs))  # avoid true DC
    A_filt = np.zeros_like(A, dtype=complex)
    A_filt[hp_mask] = A[hp_mask]

    # Double integration in frequency domain
    omega = 2*np.pi*freqs
    denom = -(omega**2)
    denom[0] = np.inf  # avoid divide by zero
    S = A_filt / denom

    disp = irfft(S, n=n)
    return disp - np.mean(disp)


@dataclass
class WaveStats:
    H_mean: float
    H_s: float
    T_mean: float
    T_z: float
    count: int


def zero_upcrossing_waves(eta, fs):
    """Wave-by-wave analysis from displacement eta(t) via zero-UPcrossings.
       Returns arrays of individual wave heights (crest-to-trough) and periods.
    """
    # Zero upcrossings indices
    x = eta
    sgn = np.signbit(x)  # True if negative
    crossings = np.where((sgn[:-1] == True) & (sgn[1:] == False))[0]

    Hs = []
    Ts = []
    for i in range(len(crossings)-1):
        i0 = crossings[i]
        i1 = crossings[i+1]
        segment = x[i0:i1+1]
        if segment.size < 3:
            continue
        c = np.max(segment)
        t = np.min(segment)
        H = c - t
        T = (i1 - i0) / fs
        if H > 0 and T > 0:
            Hs.append(H)
            Ts.append(T)
    return np.array(Hs), np.array(Ts)


def summarize_waves(Hs, Ts):
    if len(Hs) == 0 or len(Ts) == 0:
        return WaveStats(np.nan, np.nan, np.nan, np.nan, 0)
    idx = np.argsort(Hs)[::-1]
    top_third = idx[:max(1, len(Hs)//3)]
    H_mean = float(np.mean(Hs))
    H_s = float(np.mean(Hs[top_third]))  # significant wave height = avg highest 1/3
    T_mean = float(np.mean(Ts))
    T_z = float(np.mean(Ts))             # here equal to mean zero-upcrossing period
    return WaveStats(H_mean, H_s, T_mean, T_z, len(Hs))


# ---------------------------- Data ingestion -------------------------------

def load_imu_from_rosbag(path, topic='/imu/data', limit=None):
    """Load from ROS1 bag if rosbag is available. Returns dict with arrays:
       t, ax, ay, az, qw, qx, qy, qz
    """
    if rosbag is None:
        raise RuntimeError("rosbag (ROS1) not available in your Python env. "
                           "Install ROS1 or use --csv input.")
    t_list = []
    ax, ay, az = [], [], []
    qw, qx, qy, qz = [], [], [], []
    count = 0
    with rosbag.Bag(path, 'r') as bag:
        for _, msg, stamp in bag.read_messages(topics=[topic]):
            # sensor_msgs/Imu
            t_list.append(stamp.to_sec())
            ax.append(msg.linear_acceleration.x)
            ay.append(msg.linear_acceleration.y)
            az.append(msg.linear_acceleration.z)
            qw.append(msg.orientation.w)
            qx.append(msg.orientation.x)
            qy.append(msg.orientation.y)
            qz.append(msg.orientation.z)
            count += 1
            if limit and count >= limit:
                break
    t = np.array(t_list)
    return dict(
        t=t,
        ax=np.array(ax), ay=np.array(ay), az=np.array(az),
        qw=np.array(qw), qx=np.array(qx), qy=np.array(qy), qz=np.array(qz)
    )


def load_imu_from_csv(path):
    """CSV columns: time, ax, ay, az, qw, qx, qy, qz"""
    import pandas as pd
    df = pd.read_csv(path)
    for col in ['time','ax','ay','az','qw','qx','qy','qz']:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")
    return dict(
        t=df['time'].to_numpy(),
        ax=df['ax'].to_numpy(), ay=df['ay'].to_numpy(), az=df['az'].to_numpy(),
        qw=df['qw'].to_numpy(), qx=df['qx'].to_numpy(),
        qy=df['qy'].to_numpy(), qz=df['qz'].to_numpy()
    )


# ---------------------------- Main pipeline --------------------------------

def compute_vertical_acc_series(imu):
    """Vectorize vertical acceleration computation from IMU arrays."""
    ax, ay, az = imu['ax'], imu['ay'], imu['az']
    qw, qx, qy, qz = imu['qw'], imu['qx'], imu['qy'], imu['qz']
    Atilde = np.zeros_like(ax, dtype=float)
    for i in range(len(ax)):
        Atilde[i] = vertical_acc_from_imu(ax[i], ay[i], az[i], qw[i], qx[i], qy[i], qz[i])
    return Atilde


def run_pipeline(acc_z, fs, method, lp_fc=0.5, wavelet_name='db6', wavelet_level=None,
                 pde_iter=50, pde_k=None, pde_lam=0.2, hp_f=0.03):
    """Apply denoise method -> frequency double integrate -> wave stats."""
    if method == 'lowpass':
        az = butter_lowpass(acc_z, fs, fc=lp_fc, order=4)
    elif method == 'wavelet':
        az = wavelet_denoise(acc_z, wavelet=wavelet_name, level=wavelet_level, mode='soft')
    elif method == 'pde':
        az = anisotropic_diffusion_1d(acc_z, n_iter=pde_iter, k=pde_k, lam=pde_lam, use_g2=True)
    else:
        raise ValueError("Unknown method")

    eta = freq_double_integrate(az, fs, f_hp=hp_f)
    Hs, Ts = zero_upcrossing_waves(eta, fs)
    stats = summarize_waves(Hs, Ts)
    return az, eta, Hs, Ts, stats


def main():
    ap = argparse.ArgumentParser(description="IMU-to-Waves: PDE vs Low-pass/Wavelet")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--bag', type=str, help='ROS1 bag file path')
    src.add_argument('--csv', type=str, help='CSV path with time,ax,ay,az,qw,qx,qy,qz')
    ap.add_argument('--topic', type=str, default='/imu/data', help='IMU topic (ROS1)')
    ap.add_argument('--rate', type=float, required=True, help='IMU sample rate [Hz]')
    ap.add_argument('--limit', type=int, default=None, help='Limit number of samples (debug)')

    # Method params
    ap.add_argument('--lp_fc', type=float, default=0.5, help='Low-pass cutoff Hz')
    ap.add_argument('--wavelet', type=str, default='db6')
    ap.add_argument('--wavelet_level', type=int, default=None)
    ap.add_argument('--pde_iter', type=int, default=60)
    ap.add_argument('--pde_k', type=float, default=None)
    ap.add_argument('--pde_lam', type=float, default=0.2)
    ap.add_argument('--hp_f', type=float, default=0.03, help='High-pass trim before integration (Hz)')
    ap.add_argument('--outdir', type=str, default='waves_out')

    args = ap.parse_args()
    fs = args.rate
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    if args.bag:
        imu = load_imu_from_rosbag(args.bag, topic=args.topic, limit=args.limit)
    else:
        imu = load_imu_from_csv(args.csv)

    t = imu['t']
    if args.limit:
        for k in list(imu.keys()):
            imu[k] = imu[k][:args.limit]
        t = imu['t']

    # Compute vertical acceleration (geographic Z) per paper
    acc_z_raw = compute_vertical_acc_series(imu)

    methods = ['lowpass', 'wavelet', 'pde']
    labels = {'lowpass': f'Low-pass ({args.lp_fc} Hz)',
              'wavelet': f'Wavelet ({args.wavelet})',
              'pde': 'Anisotropic diffusion (PDE)'}

    all_stats = {}
    all_eta = {}
    all_az = {}

    for m in methods:
        az, eta, Hs, Ts, stats = run_pipeline(
            acc_z_raw, fs, m,
            lp_fc=args.lp_fc,
            wavelet_name=args.wavelet, wavelet_level=args.wavelet_level,
            pde_iter=args.pde_iter, pde_k=args.pde_k, pde_lam=args.pde_lam,
            hp_f=args.hp_f
        )
        all_stats[m] = stats
        all_eta[m] = eta
        all_az[m] = az

        # Save per-method CSV of waves
        out_csv = os.path.join(args.outdir, f'waves_{m}.csv')
        np.savetxt(out_csv,
                   np.column_stack([np.arange(len(Hs)), Hs, Ts]),
                   delimiter=',', header='i,H,T', comments='')
        print(f"[{m}] Saved wave list: {out_csv} (n={len(Hs)})")

    # Print summary
    print("\n=== Wave Statistics (zero-upcrossing) ===")
    for m in methods:
        s = all_stats[m]
        print(f"{labels[m]:35s}  N={s.count:4d}  H_mean={s.H_mean:6.3f} m  Hs={s.H_s:6.3f} m  "
              f"T_mean={s.T_mean:5.2f} s  Tz={s.T_z:5.2f} s")

    # Plots
    # 1) Acc vertical (raw vs denoised overlays)
    plt.figure()
    n = len(t)
    t_rel = t - t[0]
    plt.plot(t_rel, acc_z_raw, alpha=0.6, label='Raw Ãz')
    for m in methods:
        plt.plot(t_rel, all_az[m], alpha=0.9, label=labels[m])
    plt.xlabel('Time [s]'); plt.ylabel('Vertical Accel [m/s²]')
    plt.title('Vertical Acceleration: Raw vs Denoised')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'acc_vertical_comparison.png'), dpi=200)

    # 2) Displacement comparison
    plt.figure()
    for m in methods:
        plt.plot(t_rel, all_eta[m], alpha=0.9, label=labels[m])
    plt.xlabel('Time [s]'); plt.ylabel('Displacement η [m]')
    plt.title('Displacement from Frequency-Domain Double Integration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'disp_comparison.png'), dpi=200)

    # 3) Boxplots of H and T
    plt.figure()
    data_H = [np.loadtxt(os.path.join(args.outdir, f'waves_{m}.csv'), delimiter=',', skiprows=1)[:,1]
              if os.path.exists(os.path.join(args.outdir, f'waves_{m}.csv')) else np.array([np.nan])
              for m in methods]
    plt.boxplot(data_H, labels=[labels[m] for m in methods])
    plt.ylabel('Wave Height [m]')
    plt.title('Wave Height Distribution (crest-to-trough)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'box_heights.png'), dpi=200)

    plt.figure()
    data_T = [np.loadtxt(os.path.join(args.outdir, f'waves_{m}.csv'), delimiter=',', skiprows=1)[:,2]
              if os.path.exists(os.path.join(args.outdir, f'waves_{m}.csv')) else np.array([np.nan])
              for m in methods]
    plt.boxplot(data_T, labels=[labels[m] for m in methods])
    plt.ylabel('Wave Period [s]')
    plt.title('Wave Period Distribution (zero-upcrossing)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'box_periods.png'), dpi=200)

    print(f"\nSaved figures in: {args.outdir}")
    print("Done.")

if __name__ == '__main__':
    main()
'''
# Write the script to disk
path = '/mnt/data/imu_waves_pde_vs_filters.py'
with open(path, 'w') as f:
    f.write(script_content)
path
