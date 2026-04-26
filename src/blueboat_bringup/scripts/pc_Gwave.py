#!/usr/bin/env python3
import math
import struct
import numpy as np
import rospy

from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry


# ============================================================
# Timed Velodyne point format (matches your working scripts)
# x,y,z,intensity float32 + ring uint16 + time float32
# ============================================================
FMT = "<4fHf"
FMT_SIZE = struct.calcsize(FMT)  # 22 bytes
G = 9.81


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def euler_R(roll, pitch, yaw):
    # body->world (Z-Y-X)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def quat_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


# ============================================================
# Wikipedia-style multi-component Gerstner surface (parametric)
# ============================================================
class GerstnerSea:
    """
    theta = kx*alpha + kz*beta - omega*t - phi

    xi  = alpha - sum (kx/k)*a*sin(theta)/tanh(kh)
    eta = beta  - sum (kz/k)*a*sin(theta)/tanh(kh)
    zeta=         sum a*cos(theta)

    omega^2 = g k tanh(kh)  (deep water => tanh->1)
    """

    def __init__(self, components, depth_h=np.inf, g=9.81):
        self.g = float(g)
        self.h = float(depth_h) if depth_h is not None else float("inf")
        self.comp = []

        for c in components:
            a = float(c["a"])               # amplitude (m)
            L = float(c["L"])               # wavelength (m)
            theta = float(c.get("theta", 0.0))  # direction (rad)
            phi = float(c.get("phi", 0.0))      # phase offset (rad)

            k = 2.0 * math.pi / max(L, 1e-9)
            kx = k * math.cos(theta)
            kz = k * math.sin(theta)

            if math.isfinite(self.h):
                tanh_kh = math.tanh(k * self.h)
                tanh_kh = max(tanh_kh, 1e-6)
                omega = math.sqrt(self.g * k * tanh_kh)
                inv_tanh = 1.0 / tanh_kh
            else:
                omega = math.sqrt(self.g * k)
                inv_tanh = 1.0

            self.comp.append((a, k, kx, kz, omega, phi, inv_tanh))

    def surface_point(self, alpha, beta, t):
        xi = float(alpha)
        eta = float(beta)
        zeta = 0.0
        for (a, k, kx, kz, omega, phi, inv_tanh) in self.comp:
            th = kx * alpha + kz * beta - omega * t - phi
            s = math.sin(th)
            c = math.cos(th)
            xi  -= (kx / k) * a * inv_tanh * s
            eta -= (kz / k) * a * inv_tanh * s
            zeta += a * c
        return np.array([xi, eta, zeta], dtype=np.float64)


def pca_plane_normal(points_xyz):
    c = points_xyz.mean(axis=0)
    X = points_xyz - c
    C = (X.T @ X) / max(1.0, float(points_xyz.shape[0]))
    w, v = np.linalg.eigh(C)  # ascending eigenvalues
    n = v[:, 0]
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / nn
    if n[2] < 0:
        n = -n
    return n, c


# ============================================================
# Second-order “buoyancy-ish” response
# x_ddot + 2*zeta*wn*x_dot + wn^2*x = wn^2*u
# ============================================================
class SecondOrder:
    def __init__(self, wn, zeta):
        self.wn = float(max(wn, 1e-3))
        self.zeta = float(max(zeta, 0.0))
        self.x = 0.0
        self.xd = 0.0

    def step(self, u, dt):
        dt = float(max(dt, 1e-4))
        wn = self.wn
        z = self.zeta
        xdd = (wn * wn) * (u - self.x) - (2.0 * z * wn) * self.xd
        self.xd += xdd * dt
        self.x += self.xd * dt
        return self.x, self.xd, xdd


# ============================================================
# Main node: occluded cloud + simulated IMU
# ============================================================
class WaveLidarImuInjector:
    def __init__(self):
        rospy.init_node("pc_gerstner_lidar_imu_injector", anonymous=True)

        # Topics
        self.cloud_in  = rospy.get_param("~cloud_in",  "/velodyne_points_timed")
        self.cloud_out = rospy.get_param("~cloud_out", "/noisyLidar")
        self.imu_out   = rospy.get_param("~imu_out",   "/imu/data_wave")
        self.odom_in   = rospy.get_param("~odom_in",   "/odometry/filtered")

        self.use_odom = bool(rospy.get_param("~use_odom", True))

        # Boat geometry / hydro-ish params
        self.Lb = float(rospy.get_param("~boat_length_m", 7.75))
        self.Bb = float(rospy.get_param("~boat_beam_m",   2.74))
        self.draft = float(rospy.get_param("~draft_m",    0.46))   # meters

        self.mass_kg = float(rospy.get_param("~mass_kg", 3322))
        self.GMt = float(rospy.get_param("~GMt_m", 0.4))
        self.GMl = float(rospy.get_param("~GMl_m", 1.5))

        self.zeta_heave = float(rospy.get_param("~zeta_heave", 0.35))
        self.zeta_roll  = float(rospy.get_param("~zeta_roll",  0.35))
        self.zeta_pitch = float(rospy.get_param("~zeta_pitch", 0.35))

        # LiDAR mounting (relative to boat/body origin used here)
        self.lidar_z = float(rospy.get_param("~lidar_mount_z_m", 1.2))
        self.plane_bias = float(rospy.get_param("~plane_bias_m", 0.0))

        # Hull sampling grid for plane fit
        self.grid_n_long = int(rospy.get_param("~grid_n_long", 5))
        self.grid_n_lat  = int(rospy.get_param("~grid_n_lat",  3))

        # Occlusion gates
        self.r_min = float(rospy.get_param("~r_min", 0.8))
        self.r_max = float(rospy.get_param("~r_max", 120.0))

        # IMU output settings
        self.imu_frame = str(rospy.get_param("~imu_frame", "imu_link"))
        self.imu_rate_hz = float(rospy.get_param("~imu_rate_hz", 200.0))
        self.gyro_noise_std = float(rospy.get_param("~gyro_noise_std", 0.002))
        self.acc_noise_std  = float(rospy.get_param("~acc_noise_std",  0.05))

        # Sea model (Wikipedia Gerstner)
        waves = rospy.get_param("~waves", [
            {"a": 0.25, "L": 8.0, "theta": 0.3, "phi": 0.0},
            {"a": 0.10, "L": 4.5, "theta": 1.1, "phi": 1.2},
        ])
        depth_h = rospy.get_param("~depth_h", float("inf"))
        self.sea = GerstnerSea(waves, depth_h=depth_h, g=G)

        # Response natural frequencies from crude hydro linearization
        rho = 1025.0
        A_wp = max(self.Lb * self.Bb, 1e-3)
        kz = rho * G * A_wp
        wn_heave = math.sqrt(max(kz / max(self.mass_kg, 1e-3), 1e-6))

        Ixx = max(self.mass_kg * (self.Bb * self.Bb) / 12.0, 1e-6)
        Iyy = max(self.mass_kg * (self.Lb * self.Lb) / 12.0, 1e-6)
        kphi = max(self.mass_kg * G * self.GMt, 1e-6)
        kthe = max(self.mass_kg * G * self.GMl, 1e-6)
        wn_roll  = math.sqrt(max(kphi / Ixx, 1e-6))
        wn_pitch = math.sqrt(max(kthe / Iyy, 1e-6))

        self.heave_sys = SecondOrder(wn_heave, self.zeta_heave)
        self.roll_sys  = SecondOrder(wn_roll,  self.zeta_roll)
        self.pitch_sys = SecondOrder(wn_pitch, self.zeta_pitch)

        # Odom state
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.yaw_rate = 0.0
        self.have_odom = False
        self._last_odom_t = None
        self._last_odom_yaw = None

        # Time state
        self.last_t = None

        # Publishers/subscribers
        self.pub_cloud = rospy.Publisher(self.cloud_out, PointCloud2, queue_size=3)
        self.pub_imu   = rospy.Publisher(self.imu_out, Imu, queue_size=50)

        rospy.Subscriber(self.cloud_in, PointCloud2, self.cb_cloud, queue_size=1)
        if self.use_odom:
            rospy.Subscriber(self.odom_in, Odometry, self.cb_odom, queue_size=10)

        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.imu_rate_hz)), self.cb_imu_timer)

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        t = msg.header.stamp.to_sec()
        if self._last_odom_t is not None:
            dt = max(1e-4, t - self._last_odom_t)
            dy = wrap_pi(yaw - self._last_odom_yaw)
            self.yaw_rate = dy / dt

        self.odom_x = float(p.x)
        self.odom_y = float(p.y)
        self.odom_yaw = float(yaw)
        self.have_odom = True
        self._last_odom_t = t
        self._last_odom_yaw = yaw

    def sample_hull_points(self, xc, yc, yaw, t):
        f = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        r = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)

        nl = max(2, self.grid_n_long)
        nb = max(2, self.grid_n_lat)
        ls = np.linspace(-0.5 * self.Lb, 0.5 * self.Lb, nl)
        bs = np.linspace(-0.5 * self.Bb, 0.5 * self.Bb, nb)

        pts = []
        for dl in ls:
            for db in bs:
                ab = np.array([xc, yc], dtype=np.float64) + dl * f + db * r
                pts.append(self.sea.surface_point(ab[0], ab[1], t))
        return np.stack(pts, axis=0)

    def decode_points(self, msg: PointCloud2):
        if msg.point_step < FMT_SIZE:
            return None, None, None, None

        n = int(msg.width) * int(msg.height)
        if n <= 0:
            return None, None, None, None

        step = msg.point_step
        pad_len = step - FMT_SIZE
        data = msg.data

        pts = np.empty((n, 6), dtype=np.float32)
        extra = [b""] * n if pad_len > 0 else None

        for i in range(n):
            off = i * step
            x, y, z, intensity, ring, tfield = struct.unpack(FMT, data[off:off + FMT_SIZE])
            pts[i, 0] = x
            pts[i, 1] = y
            pts[i, 2] = z
            pts[i, 3] = intensity
            pts[i, 4] = float(ring)
            pts[i, 5] = float(tfield)
            if extra is not None:
                extra[i] = data[off + FMT_SIZE:off + step]

        return pts, extra, step, pad_len

    def repack_pc2(self, msg_out: PointCloud2, pts6: np.ndarray, extra_list, step: int, pad_len: int):
        n = int(pts6.shape[0])
        out = bytearray(n * step)

        for i in range(n):
            off = i * step
            x = float(pts6[i, 0])
            y = float(pts6[i, 1])
            z = float(pts6[i, 2])
            intensity = float(pts6[i, 3])
            ring = int(np.uint16(pts6[i, 4]))
            tfield = float(pts6[i, 5])

            out[off:off + FMT_SIZE] = struct.pack(FMT, x, y, z, intensity, ring, tfield)

            if pad_len > 0:
                if extra_list is not None and i < len(extra_list) and extra_list[i] is not None:
                    ex = extra_list[i]
                    if len(ex) != pad_len:
                        ex = (ex + (b"\x00" * pad_len))[:pad_len]
                    out[off + FMT_SIZE:off + step] = ex
                else:
                    out[off + FMT_SIZE:off + step] = b"\x00" * pad_len

        msg_out.data = bytes(out)
        msg_out.height = 1
        msg_out.width = n
        msg_out.row_step = n * msg_out.point_step
        msg_out.is_dense = False

    def update_boat_state(self, t, dt):
        # Pose source: odom if available
        if self.use_odom and self.have_odom:
            xc, yc, yaw = self.odom_x, self.odom_y, self.odom_yaw
            yaw_rate = self.yaw_rate
        else:
            xc, yc, yaw = 0.0, 0.0, 0.0
            yaw_rate = 0.0

        # hull footprint sampling on wave surface
        hull_pts = self.sample_hull_points(xc, yc, yaw, t)

        # best-fit plane gives "target" normal + centroid
        n_world, c_world = pca_plane_normal(hull_pts)

        # desired roll/pitch relative to yaw-aligned frame
        R_yaw = euler_R(0.0, 0.0, yaw)
        n_body_yaw = R_yaw.T @ n_world
        roll_des  = math.atan2(n_body_yaw[1], max(n_body_yaw[2], 1e-6))
        pitch_des = -math.atan2(n_body_yaw[0], max(n_body_yaw[2], 1e-6))

        # desired heave: water height at centroid - draft
        heave_des = float(c_world[2] - self.draft)

        heave, heave_d, heave_dd = self.heave_sys.step(heave_des, dt)
        roll,  roll_d,  _        = self.roll_sys.step(roll_des, dt)
        pitch, pitch_d, _        = self.pitch_sys.step(pitch_des, dt)

        return {
            "xc": xc, "yc": yc, "yaw": yaw, "yaw_rate": yaw_rate,
            "heave": heave, "heave_d": heave_d, "heave_dd": heave_dd,
            "roll": roll, "roll_d": roll_d,
            "pitch": pitch, "pitch_d": pitch_d,
            "n_world": n_world, "c_world": c_world
        }

    def cb_imu_timer(self, _evt):
        # This IMU is driven by wall time; if you are on /use_sim_time, it will follow sim time.
        t = rospy.Time.now().to_sec()

        if self.last_t is None:
            self.last_t = t
            return

        dt = max(1e-4, t - self.last_t)
        self.last_t = t

        st = self.update_boat_state(t, dt)

        roll, pitch, yaw = st["roll"], st["pitch"], st["yaw"]
        roll_d, pitch_d, yaw_rate = st["roll_d"], st["pitch_d"], st["yaw_rate"]
        heave_dd = st["heave_dd"]

        # Orientation
        qx, qy, qz, qw = quat_from_euler(roll, pitch, yaw)

        # Angular velocity in body frame (approx: Euler rates -> body rates; small-angle is fine here)
        wx = roll_d
        wy = pitch_d
        wz = yaw_rate

        w = np.array([wx, wy, wz], dtype=np.float64)
        w += np.random.normal(0.0, self.gyro_noise_std, size=(3,))

        # Linear acceleration (specific force) in body frame:
        # world accel of boat ref: [0,0,heave_dd]; specific = a - g
        a_world_specific = np.array([0.0, 0.0, heave_dd - G], dtype=np.float64)
        R_bw = euler_R(roll, pitch, yaw)
        a_body = (R_bw.T @ a_world_specific)
        a_body += np.random.normal(0.0, self.acc_noise_std, size=(3,))

        msg = Imu()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.imu_frame
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.angular_velocity.x = float(w[0])
        msg.angular_velocity.y = float(w[1])
        msg.angular_velocity.z = float(w[2])
        msg.linear_acceleration.x = float(a_body[0])
        msg.linear_acceleration.y = float(a_body[1])
        msg.linear_acceleration.z = float(a_body[2])

        self.pub_imu.publish(msg)

    def cb_cloud(self, msg: PointCloud2):
        pts6, extra, step, pad_len = self.decode_points(msg)
        if pts6 is None:
            return

        t = msg.header.stamp.to_sec()
        if self.last_t is None:
            self.last_t = t

        dt = max(1e-4, t - self.last_t)

        st = self.update_boat_state(t, dt)

        roll, pitch, yaw = st["roll"], st["pitch"], st["yaw"]
        xc, yc, heave = st["xc"], st["yc"], st["heave"]
        n_world = st["n_world"]
        c_world = st["c_world"]

        R_bw = euler_R(roll, pitch, yaw)

        # Boat reference point in world (using heave). This is a minimal consistent reference.
        boat_ref_world = np.array([xc, yc, heave], dtype=np.float64)

        # LiDAR origin in world: boat_ref + R*[0,0,lidar_z]
        lidar_origin_world = boat_ref_world + (R_bw @ np.array([0.0, 0.0, self.lidar_z], dtype=np.float64))

        # Local water plane point (world): at (xc,yc) with z from best-fit centroid
        plane_point_world = np.array([xc, yc, float(c_world[2])], dtype=np.float64) + self.plane_bias * n_world

        # Transform plane into LiDAR/body frame
        n_lidar = (R_bw.T @ n_world)
        nn = np.linalg.norm(n_lidar)
        if nn < 1e-9:
            n_lidar = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            n_lidar = n_lidar / nn

        p0_lidar = (R_bw.T @ (plane_point_world - lidar_origin_world))

        # Ray-plane intersection occlusion
        xyz = pts6[:, :3].astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)

        valid = np.isfinite(r) & (r >= self.r_min) & (r <= self.r_max)

        d = np.zeros_like(xyz)
        d[valid] = xyz[valid] / r[valid, None]

        nd = d @ n_lidar
        np0 = float(np.dot(n_lidar, p0_lidar))

        good = valid & (np.abs(nd) > 1e-9)
        t_int = np.zeros(len(r), dtype=np.float64)
        t_int[good] = np0 / nd[good]

        occluded = np.zeros(len(r), dtype=bool)
        occluded[good] = (t_int[good] > 0.0) & (t_int[good] < r[good])

        keep = valid & (~occluded)

        pts_kept = pts6[keep]
        extra_kept = [extra[i] for i in np.where(keep)[0]] if extra is not None else None

        out = PointCloud2()
        out.header = msg.header
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.height = 1
        out.width = int(pts_kept.shape[0])
        out.row_step = out.width * out.point_step
        out.is_dense = False

        self.repack_pc2(out, pts_kept, extra_kept, step, pad_len)
        self.pub_cloud.publish(out)


if __name__ == "__main__":
    try:
        WaveLidarImuInjector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
