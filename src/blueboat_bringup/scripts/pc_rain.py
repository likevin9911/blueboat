#!/usr/bin/env python3
import rospy
import numpy as np
import threading

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, String

from pylisa import atmos_models as am

# -----------------------------
# Thread-Safe Global State
# -----------------------------
subscriber_topic = '/blueboat/sensors/lidars/lidar_blueboat/points'
publisher_topic  = '/noisyLidar'
pub = None

state_lock           = threading.Lock()
my_severity          = 5.0   # Rain/Snow: mm/hr,  Fog: visibility in metres
current_weather_mode = 'rain'

# -----------------------------
# Simulator LiDAR Constants
# Derived from blueboat_3d_lidar.xacro (32-beam config in lidar.xacro):
#   z (mount height)       = 1.8 m
#   min_vertical_angle     = -0.186 rad  = -10.66 deg
#   max_vertical_angle     =  0.540 rad  =  30.93 deg
#   samples (horizontal)   = 2187
#   max_range              = 130 m
#   min_range (Gazebo)     = 0.1 m
#   update_rate            = 10 Hz
# -----------------------------
LIDAR_MOUNT_HEIGHT = 1.8    # metres above water surface
N_RINGS            = 32
# Beam elevation angles derived from xacro min/max_vertical_angle
VLP32_ELEV_DEG     = np.linspace(-10.66, 30.93, N_RINGS, dtype=np.float32)
R_MAX              = 130.0  # metres
R_MIN              = 0.1    # metres
N_AZ_BINS          = 2048   # nearest power-of-two to xacro samples=2187
SCAN_DURATION      = 0.1    # seconds (10 Hz)

OCCLUSION_MARGIN      = 0.25
FILL_ONLY_WHEN_NO_HIT = False
FRAC_ON_EXISTING      = 0.35
MAX_ADD               = 25000
GROUND_TOLERANCE      = 0.05   # metres
HULL_EXCLUSION_RADIUS = 1.5    # metres
ELEV_REFRESH_SECS     = 0.5    # seconds

# Plugin intensity convention (libgazebo_ros_velodyne_gpu_laser):
#   material hit:  intensity = raw_reflectance * 255.0        (range: 0-255)
#   no material:   intensity = 255.0 * exp(-0.01 * r)         (range-dependent)
# Weather injection must match this envelope -- see _compute_intensity.

# -----------------------------
# PointField datatype map
# -----------------------------
_PF_DTYPE = {
    1: np.int8,   2: np.uint8,  3: np.int16,  4: np.uint16,
    5: np.int32,  6: np.uint32, 7: np.float32, 8: np.float64,
}

# -----------------------------
# Weather Configuration
# -----------------------------
# UNIT NOTES
#   Rain / snow (sampler_type='direct'):
#       mp_sample_rain / mg_sample_snow expect and return diameters in mm.
#       dstart / dmax must be in mm.
#
#   Fog / spray (sampler_type='rejection'):
#       nd_* functions expect diameter in METRES (they do d*1e3 -> um internally).
#       dstart / dmax must be in METRES.
#
#   The ratio (diam / diam_mean) is dimensionless in both cases so the shared
#   intensity formula works correctly regardless of which path is active.

WEATHER_CONFIGS = {
    'rain': {
        'sampler':               am.mp_sample_rain,
        'sampler_type':          'direct',
        'intensity_base':        12.0,
        'intensity_gain':        40.0,
        'hit_prob_per_rr':       0.01,
        'attenuation_start':     25.0,
        'attenuation_max_range': 80.0,
        'max_drop_prob':         0.65,
        'dstart':  0.2,   # mm
        'dmax':    6.0,   # mm
        'label':   'Rain (Marshall-Palmer)',
        'intensity_noise': 0.15,
    },
    'snow': {
        'sampler':               am.mg_sample_snow,
        'sampler_type':          'direct',
        'intensity_base':        18.0,
        'intensity_gain':        30.0,
        'hit_prob_per_rr':       0.015,
        'attenuation_start':     15.0,
        'attenuation_max_range': 60.0,
        'max_drop_prob':         0.75,
        'dstart':  0.5,   # mm
        'dmax':   10.0,   # mm
        'label':   'Snow (Marshall-Gunn)',
        'intensity_noise': 0.20,
    },
    # ---- FOG / SPRAY  (dstart & dmax in METRES) ----
    'fog_coastal': {
        'density_func':          am.nd_haze_coast,
        'sampler_type':          'rejection',
        'intensity_base':        4.0,
        'intensity_gain':        5.0,
        'hit_prob_per_rr':       0.15,
        'attenuation_start':     3.0,
        'attenuation_max_range': 20.0,
        'max_drop_prob':         0.95,
        'dstart':  1e-6,    # 1 um
        'dmax':   100e-6,   # 100 um
        'label':   'Fog (Coastal)',
        'intensity_noise': 0.30,
    },
    'fog_continental': {
        'density_func':          am.nd_haze_continental,
        'sampler_type':          'rejection',
        'intensity_base':        4.0,
        'intensity_gain':        5.0,
        'hit_prob_per_rr':       0.15,
        'attenuation_start':     3.0,
        'attenuation_max_range': 20.0,
        'max_drop_prob':         0.95,
        'dstart':  1e-6,    # 1 um
        'dmax':   200e-6,   # 200 um  (continental fog peaks ~70 um)
        'label':   'Fog (Continental)',
        'intensity_noise': 0.30,
    },
    'fog_strong_advection': {
        'density_func':          am.nd_strong_advection_fog,
        'sampler_type':          'rejection',
        'intensity_base':        5.0,
        'intensity_gain':        6.0,
        'hit_prob_per_rr':       0.2,
        'attenuation_start':     2.0,
        'attenuation_max_range': 15.0,
        'max_drop_prob':         0.98,
        'dstart':  1e-6,
        'dmax':   200e-6,
        'label':   'Fog (Strong Advection)',
        'intensity_noise': 0.35,
    },
    'fog_moderate_advection': {
        'density_func':          am.nd_moderate_advection_fog,
        'sampler_type':          'rejection',
        'intensity_base':        4.5,
        'intensity_gain':        5.5,
        'hit_prob_per_rr':       0.18,
        'attenuation_start':     2.5,
        'attenuation_max_range': 18.0,
        'max_drop_prob':         0.96,
        'dstart':  1e-6,
        'dmax':   200e-6,
        'label':   'Fog (Moderate Advection)',
        'intensity_noise': 0.30,
    },
    'spray_strong': {
        'density_func':          am.nd_strong_spray,
        'sampler_type':          'rejection',
        'intensity_base':        6.0,
        'intensity_gain':        8.0,
        'hit_prob_per_rr':       0.25,
        'attenuation_start':     2.0,
        'attenuation_max_range': 12.0,
        'max_drop_prob':         0.98,
        'dstart':  20e-6,   # 20 um  (sea spray 20-500 um)
        'dmax':   500e-6,   # 500 um
        'label':   'Spray (Strong)',
        'intensity_noise': 0.25,
    },
    'spray_moderate': {
        'density_func':          am.nd_moderate_spray,
        'sampler_type':          'rejection',
        'intensity_base':        5.5,
        'intensity_gain':        7.0,
        'hit_prob_per_rr':       0.2,
        'attenuation_start':     2.5,
        'attenuation_max_range': 15.0,
        'max_drop_prob':         0.96,
        'dstart':  20e-6,
        'dmax':   500e-6,
        'label':   'Spray (Moderate)',
        'intensity_noise': 0.25,
    },
    'fog_chu_hogg': {
        'density_func':          am.nd_chu_hogg,
        'sampler_type':          'rejection',
        'intensity_base':        4.0,
        'intensity_gain':        5.0,
        'hit_prob_per_rr':       0.15,
        'attenuation_start':     3.0,
        'attenuation_max_range': 20.0,
        'max_drop_prob':         0.95,
        'dstart':  1e-6,
        'dmax':   100e-6,
        'label':   'Fog (Chu/Hogg)',
        'intensity_noise': 0.30,
    },
}

# -----------------------------
# Fog density cache
# Evaluated at correct metre-scale diameters so the rejection sampler
# finds a real non-zero peak and actually samples the physics distribution.
# -----------------------------
FOG_DENSITY_MAX = {}

def get_fog_max_density(cfg):
    key = cfg['label']
    if key not in FOG_DENSITY_MAX:
        d_test = np.linspace(cfg['dstart'], cfg['dmax'], 2000)
        try:
            peak = float(np.max(cfg['density_func'](d_test)))
        except Exception:
            peak = 1.0
        FOG_DENSITY_MAX[key] = peak if peak > 0 else 1.0
    return FOG_DENSITY_MAX[key]


# -----------------------------
# Dynamic PointCloud2 layout
# -----------------------------

def _build_numpy_dtype(msg):
    """Build a numpy structured dtype that exactly mirrors the PointCloud2 layout.

    Handles all Gazebo Velodyne plugin versions by reading msg.fields at runtime:
      16 bytes  x,y,z,intensity            (no ring, no time)
      18 bytes  x,y,z,intensity,ring
      22 bytes  x,y,z,intensity,ring,time
      24 bytes  x,y,z,intensity,ring,time + 2-byte alignment pad

    Gap bytes between fields and any trailing pad are captured as _pad_N so
    that dtype.itemsize always equals msg.point_step, making frombuffer safe.
    """
    fields_sorted = sorted(msg.fields, key=lambda f: f.offset)
    dtype_list    = []
    pos           = 0
    for f in fields_sorted:
        if f.offset > pos:
            dtype_list.append(('_pad_%d' % pos, np.uint8, f.offset - pos))
        np_type = _PF_DTYPE.get(f.datatype, np.uint8)
        dtype_list.append((f.name, np_type))
        pos = f.offset + np.dtype(np_type).itemsize
    if msg.point_step > pos:
        dtype_list.append(('_pad_end', np.uint8, msg.point_step - pos))
    return np.dtype(dtype_list)


def _decode_points(msg):
    """Decode a PointCloud2 into an (N, 6) float32 array.

    Internal column layout:
        0  x          metres
        1  y          metres
        2  z          metres
        3  intensity  0-255
        4  ring       0..N_RINGS-1  (float32 for array homogeneity)
        5  time       seconds from scan start  [0, SCAN_DURATION)

    ring is read from the message when present, else synthesised from the
    nearest beam elevation angle.

    time is read from the message when present, else synthesised from azimuth:
        time = (azimuth / 2*pi) * SCAN_DURATION
    This matches the convention used by the real VLP-32 driver and the Gazebo
    plugin when it does emit the time field.

    Returns
    -------
    pts      : (N, 6) float32
    has_ring : bool
    has_time : bool
    dtype    : numpy dtype (passed back to _repack_pc2)
    """
    num_points  = int(msg.width) * int(msg.height)
    dtype       = _build_numpy_dtype(msg)
    pts_struct  = np.frombuffer(msg.data, dtype=dtype, count=num_points)

    field_names = {f.name for f in msg.fields}
    has_ring    = 'ring' in field_names
    has_time    = 'time' in field_names

    pts      = np.empty((num_points, 6), dtype=np.float32)
    pts[:, 0] = pts_struct['x'].astype(np.float32)
    pts[:, 1] = pts_struct['y'].astype(np.float32)
    pts[:, 2] = pts_struct['z'].astype(np.float32)
    pts[:, 3] = pts_struct['intensity'].astype(np.float32)

    # The Gazebo GPU laser plugin already scales intensity to [0, 255]:
    #   material hit:  intensity = raw_intensity * 255.0
    #   no material:   intensity = 255.0 * exp(-0.01 * r)
    # No rescaling needed here -- passthrough values are already on 0-255.

    if has_ring:
        pts[:, 4] = pts_struct['ring'].astype(np.float32)
    else:
        pts[:, 4] = _synthesise_rings(pts[:, :3]).astype(np.float32)

    if has_time:
        pts[:, 5] = pts_struct['time'].astype(np.float32)
    else:
        pts[:, 5] = _time_from_azimuth(pts[:, :3])

    return pts, has_ring, has_time, dtype


def _repack_pc2(msg, pts6, has_ring, has_time, dtype):
    """Write an (N, 6) float32 array back into the PointCloud2 message.

    The buffer layout is preserved exactly (same point_step, same field offsets).
    ring and time are written back only if they were present in the original
    message — downstream consumers always see a schema consistent with what the
    plugin declared.

    msg.fields is rebuilt from the original fields so that offsets and datatypes
    remain authoritative.  The original header stamp is not touched.
    """
    n          = pts6.shape[0]
    out_struct = np.zeros(n, dtype=dtype)

    out_struct['x']         = pts6[:, 0]
    out_struct['y']         = pts6[:, 1]
    out_struct['z']         = pts6[:, 2]
    out_struct['intensity'] = pts6[:, 3]

    if has_ring and 'ring' in dtype.names:
        out_struct['ring'] = pts6[:, 4].astype(np.uint16)

    if has_time and 'time' in dtype.names:
        out_struct['time'] = pts6[:, 5]

    msg.data     = out_struct.tobytes()
    msg.width    = n
    msg.height   = 1
    msg.row_step = msg.point_step * n
    msg.is_dense = False

    # Rebuild fields list: keep every field from the original message.
    # This ensures point_step and field offsets remain consistent.
    msg.fields = [
        PointField(name=f.name, offset=f.offset,
                   datatype=f.datatype, count=f.count)
        for f in sorted(msg.fields, key=lambda f: f.offset)
        if not f.name.startswith('_pad')
    ]
    # msg.header.stamp intentionally NOT updated -- preserve original sensor stamp


# -----------------------------
# Ring and time synthesis
# -----------------------------

def _synthesise_rings(pts_xyz):
    """Assign ring index 0..N_RINGS-1 from nearest beam elevation angle.

    Used only when the plugin does not publish a ring field.
    Broadcasting (N,1) against (1,N_RINGS) avoids an explicit loop.
    """
    rr     = np.linalg.norm(pts_xyz, axis=1).astype(np.float32)
    rr     = np.clip(rr, 1e-6, 1e9)
    el_deg = np.degrees(
        np.arcsin(np.clip(pts_xyz[:, 2] / rr, -1.0, 1.0))
    ).astype(np.float32)
    diff     = np.abs(el_deg[:, None] - VLP32_ELEV_DEG[None, :])
    ring_idx = np.argmin(diff, axis=1).astype(np.int32)
    return ring_idx


def _time_from_azimuth(pts_xyz):
    """Compute scan-relative timestamps from azimuth angles.

    A Velodyne sensor rotates at a fixed rate; each point's firing time is
    proportional to its azimuth angle within the current 360-degree scan:
        time = (azimuth / 2*pi) * SCAN_DURATION
    Range: [0, SCAN_DURATION).
    """
    az = np.arctan2(pts_xyz[:, 1], pts_xyz[:, 0]).astype(np.float32)
    az = (az + 2.0 * np.pi) % (2.0 * np.pi)
    return (az / (2.0 * np.pi) * SCAN_DURATION).astype(np.float32)


def _time_from_az_rad(az_rad):
    """Same as _time_from_azimuth but accepts pre-computed radian azimuths."""
    az = (az_rad + 2.0 * np.pi) % (2.0 * np.pi)
    return (az / (2.0 * np.pi) * SCAN_DURATION).astype(np.float32)


# -----------------------------
# Elevation estimate & range map
# -----------------------------
ELEV_DEG_EST      = None
_last_elev_update = 0.0


def _estimate_elev_deg(pts6):
    """Per-ring median elevation in degrees from the current scan."""
    ring_idx = pts6[:, 4].astype(np.int32)
    rr       = np.linalg.norm(pts6[:, :3], axis=1).astype(np.float32)
    rr       = np.clip(rr, 1e-6, 1e9)
    el_deg   = np.degrees(
        np.arcsin(np.clip(pts6[:, 2] / rr, -1.0, 1.0))
    ).astype(np.float32)

    elev = np.full(N_RINGS, np.nan, dtype=np.float32)
    for r in range(N_RINGS):
        mask = ring_idx == r
        if np.any(mask):
            elev[r] = float(np.nanmedian(el_deg[mask]))

    good = np.isfinite(elev)
    if np.sum(good) >= 2:
        x_all       = np.arange(N_RINGS)
        elev[~good] = np.interp(
            x_all[~good], np.flatnonzero(good), elev[good]
        ).astype(np.float32)
    else:
        elev[:] = VLP32_ELEV_DEG
    return elev


def _build_range_map(pts6, n_az_bins=N_AZ_BINS):
    """(N_RINGS, n_az_bins) minimum-range map via scatter-minimum."""
    ring_idx = pts6[:, 4].astype(np.int32)
    xyz      = pts6[:, :3]
    rr       = np.linalg.norm(xyz, axis=1).astype(np.float32)
    az       = np.arctan2(xyz[:, 1], xyz[:, 0]).astype(np.float32)
    az       = (az + 2.0 * np.pi) % (2.0 * np.pi)
    az_bin   = np.clip(
        np.floor(az / (2.0 * np.pi) * n_az_bins).astype(np.int32),
        0, n_az_bins - 1
    )

    range_min = np.full((N_RINGS, n_az_bins), np.inf, dtype=np.float32)
    valid     = (ring_idx >= 0) & (ring_idx < N_RINGS)
    if np.any(valid):
        np.minimum.at(
            range_min.ravel(),
            ring_idx[valid] * n_az_bins + az_bin[valid],
            rr[valid]
        )
    return range_min


# -----------------------------
# Rejection sampler for fog/spray
# -----------------------------

def sample_from_density(density_func, N, dstart, dmax, nd_max_cached=1.0):
    """Rejection-sample N diameters (metres) from a LISA fog density function.

    dstart/dmax are in metres -- the nd_* LISA functions expect metres and
    convert to um internally via d*1e3.  nd_max_cached is the peak of the
    density function over [dstart, dmax], used as the rejection envelope.
    """
    samples   = []
    max_iters = N * 100
    iters     = 0
    envelope  = float(nd_max_cached) if nd_max_cached > 0 else 1.0

    while len(samples) < N and iters < max_iters:
        d = np.random.uniform(dstart, dmax)
        u = np.random.uniform(0.0, envelope)
        try:
            nd = float(density_func(d))
        except Exception:
            nd = 0.0
        if u <= nd:
            samples.append(d)
        iters += 1

    if len(samples) < N:
        remaining = N - len(samples)
        rospy.logwarn_throttle(
            10.0,
            "Rejection sampler fallback: %d/%d samples drawn uniformly",
            remaining, N
        )
        samples.extend(np.random.uniform(dstart, dmax, remaining).tolist())

    return np.array(samples[:N], dtype=np.float32)


# -----------------------------
# Main weather injection
# -----------------------------

def add_weather(msg: PointCloud2):
    global my_severity, current_weather_mode, ELEV_DEG_EST, _last_elev_update

    with state_lock:
        cfg          = WEATHER_CONFIGS.get(current_weather_mode, WEATHER_CONFIGS['rain'])
        severity_val = my_severity

    is_fog = cfg['sampler_type'] == 'rejection'

    if is_fog:
        vis           = float(np.clip(severity_val, 5.0, 500.0))
        density_scale = float(np.clip(500.0 / vis, 1.0, 50.0))
        Rr            = density_scale
    else:
        Rr            = float(max(severity_val, 0.0))
        density_scale = 1.0

    if Rr <= 0.0 and not is_fog:
        return

    if msg.point_step < 16:
        rospy.logwarn_throttle(2.0, "point_step=%d < 16; skipping", msg.point_step)
        return

    num_points = int(msg.width) * int(msg.height)
    if num_points <= 0:
        return

    try:
        pts, has_ring, has_time, point_dtype = _decode_points(msg)
    except Exception as e:
        rospy.logwarn_throttle(2.0, "Decode error: %s", e)
        return

    # Refresh per-ring elevation estimate
    now = rospy.get_time()
    if ELEV_DEG_EST is None or (now - _last_elev_update) > ELEV_REFRESH_SECS:
        ELEV_DEG_EST      = _estimate_elev_deg(pts)
        _last_elev_update = now

    # -------------------------
    # Attenuation / dropout
    # -------------------------
    r    = np.linalg.norm(pts[:, :3], axis=1)
    keep = np.ones(num_points, dtype=bool)

    if cfg['attenuation_max_range'] > cfg['attenuation_start']:
        far = r > cfg['attenuation_start']
        if np.any(far):
            alpha         = np.clip(
                (r - cfg['attenuation_start']) /
                (cfg['attenuation_max_range'] - cfg['attenuation_start']),
                0.0, 1.0
            )
            severity_norm = Rr / (Rr + 10.0)
            drop_prob     = np.clip(
                alpha * cfg['max_drop_prob'] * severity_norm,
                0.0, cfg['max_drop_prob']
            )
            keep = ~(far & (np.random.rand(num_points) < drop_prob))

    pts_kept = pts[keep]
    if pts_kept.shape[0] == 0:
        return

    range_min = _build_range_map(pts_kept)

    # -------------------------
    # Sample counts
    # -------------------------
    n_rays = pts_kept.shape[0]
    p_hit  = float(np.clip(cfg['hit_prob_per_rr'] * Rr, 0.0, 1.0))
    if is_fog:
        p_hit = float(np.clip(0.15 * density_scale, 0.0, 1.0))

    N_add = int(np.clip(np.random.binomial(n_rays, p_hit), 0, MAX_ADD))

    if N_add <= 0:
        _repack_pc2(msg, pts_kept, has_ring, has_time, point_dtype)
        return

    # -------------------------
    # Sample particle diameters
    # -------------------------
    try:
        if cfg['sampler_type'] == 'direct':
            diam = cfg['sampler'](Rr, N_add, dstart=cfg['dstart']).astype(np.float32)
            diam = np.clip(diam, cfg['dstart'], cfg['dmax'])
        else:
            nd_max = get_fog_max_density(cfg)
            diam   = sample_from_density(
                cfg['density_func'], N_add,
                cfg['dstart'], cfg['dmax'],
                nd_max_cached=nd_max,
            )
    except Exception as e:
        rospy.logerr_throttle(2.0, "Weather sampling failed: %s", e)
        _repack_pc2(msg, pts_kept, has_ring, has_time, point_dtype)
        return

    N_on   = int(FRAC_ON_EXISTING * N_add)
    N_fill = N_add - N_on

    drops_list = []

    if N_on > 0:
        drops_on = _generate_on_existing_rays(
            pts_kept, N_on, diam[:N_on], Rr, cfg
        )
        if drops_on.shape[0] > 0:
            drops_list.append(drops_on)

    if N_fill > 0:
        drops_fill = _generate_360_occlusion(
            N_fill, diam[N_on:], Rr, ELEV_DEG_EST, range_min, cfg
        )
        if drops_fill.shape[0] > 0:
            drops_list.append(drops_fill)

    if not drops_list:
        _repack_pc2(msg, pts_kept, has_ring, has_time, point_dtype)
        return

    drops = np.vstack(drops_list)

    # Final hard Z-filter against water surface
    WATER_SURFACE_Z = -LIDAR_MOUNT_HEIGHT + GROUND_TOLERANCE
    underwater      = drops[:, 2] < WATER_SURFACE_Z
    if np.any(underwater):
        rospy.logwarn_throttle(
            2.0, "Rejected %d underwater weather points (Z < %.3f m)",
            int(np.sum(underwater)), WATER_SURFACE_Z
        )
        drops = drops[~underwater]

    pts_out = np.vstack([pts_kept, drops]) if drops.shape[0] > 0 else pts_kept
    _repack_pc2(msg, pts_out, has_ring, has_time, point_dtype)


# -----------------------------
# Weather point generators
# Both return (N, 6) float32: x, y, z, intensity, ring, time
# -----------------------------

def _compute_intensity(diam, rr, cfg, N):
    """Compute weather-point intensity consistent with the Gazebo GPU laser plugin.

    The plugin formula is:
        material hit:  intensity = raw_reflectance * 255
        no material:   intensity = 255 * exp(-0.01 * r)

    Weather particles are partial hits in open air, so they should follow the
    same exp(-0.01*r) envelope scaled by a backscatter fraction that depends on
    particle size and type:

        intensity = 255 * exp(-0.01 * r) * backscatter_fraction * noise

    Rain / snow (sampler_type='direct'):
        backscatter = clip((diam / diam_mean) * (intensity_gain / 255), 0.05, 1.0)
        Larger drops have greater cross-section and return more signal.
        intensity_gain / 255 sets the backscatter at mean drop size.

    Fog / spray (sampler_type='rejection'):
        backscatter = intensity_base / 255.0  (fixed small fraction)
        Fog droplets are orders of magnitude smaller than rain drops and
        produce diffuse, low-level backscatter modelled as a fixed fraction.

    Parameters
    ----------
    diam : (N,) float32 -- mm for rain/snow, metres for fog/spray
                           (ratio diam/diam_mean is dimensionless either way)
    rr   : (N,) float32 -- slant range in metres
    cfg  : weather config dict
    N    : int
    """
    # Plugin-matched range envelope
    envelope = (255.0 * np.exp(-0.01 * rr)).astype(np.float32)

    is_fog = cfg['sampler_type'] == 'rejection'
    if is_fog:
        # Fixed backscatter fraction for small aerosol particles
        backscatter = float(cfg['intensity_base']) / 255.0
        backscatter = np.full(N, backscatter, dtype=np.float32)
    else:
        # Size-dependent backscatter for rain / snow drops
        diam_mean   = float(np.mean(diam)) + 1e-6
        scale       = float(cfg['intensity_gain']) / 255.0
        backscatter = np.clip(
            (diam / diam_mean) * scale, 0.05, 1.0
        ).astype(np.float32)

    noise = np.random.uniform(
        1.0 - cfg['intensity_noise'], 1.0 + cfg['intensity_noise'], N
    ).astype(np.float32)

    return np.clip(envelope * backscatter * noise, 0.0, 255.0).astype(np.float32)


def _generate_on_existing_rays(pts_kept, N_add, diam, Rr, cfg):
    """Place weather returns along rays that already produced a valid hit.

    Ring is inherited from the parent ray so the ring-organised point cloud
    structure remains self-consistent.  Time is synthesised from the weather
    point's azimuth so temporal ordering within the scan is correct.
    """
    idx  = np.random.randint(0, pts_kept.shape[0], size=N_add)
    base = pts_kept[idx]
    xyz  = base[:, :3].astype(np.float32)
    r0   = np.linalg.norm(xyz, axis=1).astype(np.float32)
    r0   = np.clip(r0, 1e-3, 1e9)
    d    = (xyz / r0[:, None]).astype(np.float32)   # unit direction vectors

    k  = 0.05 + 0.15 * (Rr / (Rr + 10.0))
    u  = np.clip(np.random.rand(N_add).astype(np.float32), 1e-6, 1.0 - 1e-6)
    rr = np.clip(-np.log(1.0 - u) / k, R_MIN, R_MAX).astype(np.float32)
    rr = np.minimum(rr, r0 - 0.25).astype(np.float32)

    WATER_SURFACE_Z = -LIDAR_MOUNT_HEIGHT + GROUND_TOLERANCE
    z_rel  = d[:, 2] * rr
    valid  = rr > R_MIN
    valid &= z_rel >= WATER_SURFACE_Z
    valid &= ~((rr < HULL_EXCLUSION_RADIUS) & (z_rel < 0))

    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float32)

    rr   = rr[valid];   d    = d[valid]
    base = base[valid]; diam = diam[valid]
    N    = rr.shape[0]

    xyz_w = d * rr[:, None]
    az_w  = np.arctan2(xyz_w[:, 1], xyz_w[:, 0]).astype(np.float32)

    drops = np.empty((N, 6), dtype=np.float32)
    drops[:, :3] = xyz_w
    drops[:, 3]  = _compute_intensity(diam, rr, cfg, N)
    drops[:, 4]  = base[:, 4]              # ring inherited from parent ray
    drops[:, 5]  = _time_from_az_rad(az_w) # time from weather point's azimuth
    return drops


def _generate_360_occlusion(N_add, diam, Rr, elev_deg, range_min, cfg):
    """Generate weather returns across full 360 degrees with occlusion checks.

    Ring is drawn uniformly from 0..N_RINGS-1 and used to look up the
    xacro-configured elevation angle.  Time is synthesised from azimuth.
    """
    if N_add <= 0:
        return np.empty((0, 6), dtype=np.float32)

    ring_idx = np.random.randint(0, N_RINGS, size=N_add).astype(np.int32)
    az       = (2.0 * np.pi * np.random.rand(N_add)).astype(np.float32)
    az_bin   = np.clip(
        np.floor(az / (2.0 * np.pi) * N_AZ_BINS).astype(np.int32),
        0, N_AZ_BINS - 1
    )

    k  = 0.05 + 0.15 * (Rr / (Rr + 10.0))
    u  = np.clip(np.random.rand(N_add).astype(np.float32), 1e-6, 1.0 - 1e-6)
    rr = np.clip(-np.log(1.0 - u) / k, R_MIN, R_MAX).astype(np.float32)

    el    = np.deg2rad(elev_deg[ring_idx]).astype(np.float32)
    z_rel = rr * np.sin(el)

    WATER_SURFACE_Z = -LIDAR_MOUNT_HEIGHT + GROUND_TOLERANCE
    valid  = np.ones(N_add, dtype=bool)
    valid &= z_rel >= WATER_SURFACE_Z
    valid &= ~((rr < HULL_EXCLUSION_RADIUS) & (z_rel < 0))
    valid &= el >= (np.deg2rad(VLP32_ELEV_DEG[ring_idx]).astype(np.float32) - 0.01)

    # Scene occlusion: cap range at nearest existing return minus margin
    scene_r = range_min[ring_idx, az_bin]
    if FILL_ONLY_WHEN_NO_HIT:
        valid &= ~np.isfinite(scene_r)
    else:
        rr    = np.minimum(rr, scene_r - OCCLUSION_MARGIN).astype(np.float32)
        z_rel = rr * np.sin(el)
        valid &= z_rel >= WATER_SURFACE_Z

    valid &= rr > R_MIN

    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float32)

    rr       = rr[valid];       az       = az[valid]
    el       = el[valid];       diam     = diam[valid]
    ring_idx = ring_idx[valid]; N        = rr.shape[0]

    cos_el = np.cos(el)
    x = rr * cos_el * np.cos(az)
    y = rr * cos_el * np.sin(az)
    z = rr * np.sin(el)

    drops = np.empty((N, 6), dtype=np.float32)
    drops[:, 0] = x
    drops[:, 1] = y
    drops[:, 2] = z
    drops[:, 3] = _compute_intensity(diam, rr, cfg, N)
    drops[:, 4] = ring_idx.astype(np.float32)  # synthesised ring
    drops[:, 5] = _time_from_az_rad(az)         # time from azimuth
    return drops


# -----------------------------
# ROS callbacks
# -----------------------------

def callback(msg: PointCloud2):
    add_weather(msg)
    pub.publish(msg)


def callback_severity(data: Float32):
    global my_severity
    with state_lock:
        my_severity = float(data.data)


def callback_mode(data: String):
    global current_weather_mode
    new_mode = data.data.lower().replace(' ', '_')
    with state_lock:
        if new_mode in WEATHER_CONFIGS:
            current_weather_mode = new_mode
            rospy.loginfo("Weather mode -> %s", WEATHER_CONFIGS[new_mode]['label'])
        else:
            rospy.logwarn(
                "Unknown weather mode '%s'. Available: %s",
                new_mode, ', '.join(WEATHER_CONFIGS.keys())
            )


def listener():
    global pub
    rospy.init_node('interceptLidar', anonymous=True)
    pub = rospy.Publisher(publisher_topic, PointCloud2, queue_size=5)

    rospy.Subscriber(subscriber_topic, PointCloud2, callback,          queue_size=5)
    rospy.Subscriber('Noise_LEVEL',    Float32,     callback_severity, queue_size=10)
    rospy.Subscriber('/weather_mode',  String,      callback_mode,     queue_size=10)

    rospy.loginfo("Weather LiDAR node started")
    rospy.loginfo("Mount height=%.2f m  |  water surface at Z=%.3f m",
                  LIDAR_MOUNT_HEIGHT, -LIDAR_MOUNT_HEIGHT + GROUND_TOLERANCE)
    rospy.loginfo("Elevation: %.2f deg -> %.2f deg  (%d rings)",
                  float(VLP32_ELEV_DEG[0]), float(VLP32_ELEV_DEG[-1]), N_RINGS)
    rospy.loginfo("Range: %.1f - %.1f m  |  az bins: %d",
                  R_MIN, R_MAX, N_AZ_BINS)
    rospy.loginfo("Available modes: %s", ', '.join(WEATHER_CONFIGS.keys()))
    rospy.spin()


if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
