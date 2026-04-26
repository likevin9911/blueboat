#!/usr/bin/env python3
"""
CmdVel → PWM with selectable output channels + IO autodetect hints.

Matches start/stop behavior of the standalone test script:
- On idle (stale/zero cmd): do a forward "flash", then hold neutral briefly, then stop overriding.
- On shutdown: flash → neutral hold → disarm → clear overrides (do not reassert afterward).

Direction control:
- LEFT_THRUST_DIR / RIGHT_THRUST_DIR = +1 means hull-forward = ESC forward (>1500)
- ... = -1 means hull-forward = ESC reverse (<1500)

Arc shaping preserved (TURN_GAIN, inner floor / outer boost).
"""

import time, sys, signal, atexit
import rospy
from geometry_msgs.msg import Twist
from pymavlink import mavutil

# ===================== CONFIG =====================

# OUTPUT channels (physical ports where ESCs are connected)
RIGHT_SERVO_NUM = 1    # MAIN1 (typical)
LEFT_SERVO_NUM  = 3    # MAIN3; if on AUX3 use 11

# RCIN channels we feed via override (stay 1 & 3 for BlueBoat)
CH_RIGHT_IN = 1
CH_LEFT_IN  = 3

CH_RIGHT_IN2 = '-RIGHT-> 1'
CH_LEFT_IN2  = '-LEFT-> 3'

# ROS timing
RATE_HZ        = 30
CMD_TIMEOUT_S  = 0.35
DEBUG_HZ       = 5.0

# Normalize cmd_vel
MAX_LINEAR     = 1.0
MAX_ANGULAR    = 1.0
LINEAR_DB      = 0.02
ANGULAR_DB     = 0.02

# PWM limits & neutrals
PWM_MIN        = 1470
PWM_MAX        = 1540
NEUTRAL_LEFT   = 1500
NEUTRAL_RIGHT  = 1500

# Start thresholds (first values that actually move your ESCs)
LEFT_FWD_START   = max(NEUTRAL_LEFT,  1526)
LEFT_REV_START   = min(NEUTRAL_LEFT,  1486)
RIGHT_FWD_START  = max(NEUTRAL_RIGHT, 1536)  # your latest test file value
RIGHT_REV_START  = min(NEUTRAL_RIGHT, 1489)

# Mixing thresholds
TURN_ONLY_W_THRESH  = 0.15
LINEAR_LOCKOUT_TURN = 0.10

# Forward "flash" (EXACTLY like the test script)
FLASH_PWM        = 1532
BLIP_TIME_S      = 0.25
FLASH_HZ         = 20.0

# --- Direction (set to +1 or -1 per side) ---
# For LH(CW) and RH(CCW) props, hull-forward likely = ESC forward on L, ESC reverse on R:
LEFT_THRUST_DIR  = +1
RIGHT_THRUST_DIR = -1

# --- Arc shaping (tune here) ---
ARC_NO_NEUTRAL_WHEN_VPOS = True
ARC_NO_NEUTRAL_WHEN_VNEG = False
TURN_GAIN = 1.40

# Floors & boosts for forward arcs (microseconds)
ARC_INNER_MIN_US_FWD   = 3
ARC_OUTER_EXTRA_US_FWD = 2

# Optional reverse-arc shaping
ARC_INNER_MIN_US_REV   = 2
ARC_OUTER_EXTRA_US_REV = 0

# MAVLink link
CONN_STR    = "udpin:0.0.0.0:14550"
SOURCE_SYS  = 1
SOURCE_COMP = 190
TARGET_SYS  = 1
TARGET_COMP = 1

# ===================== HELPERS =====================

def clamp(v, lo, hi): return max(lo, min(hi, v))
def apply_deadband(x, db): return 0.0 if abs(x) < db else x

def norm_cmd(v, w):
    v_n = clamp(apply_deadband(v / MAX_LINEAR,  LINEAR_DB), -1.0, 1.0)
    w_n = clamp(apply_deadband(w / MAX_ANGULAR, ANGULAR_DB), -1.0, 1.0)
    return v_n, w_n

def cmd_to_pwm(cmd, neutral, fwd_start, rev_start):
    """ESC-domain: +cmd => >1500, -cmd => <1500."""
    if cmd > 0.0:
        return int(round(fwd_start + cmd * (PWM_MAX - fwd_start)))
    elif cmd < 0.0:
        mag = -cmd
        return int(round(rev_start - mag * (rev_start - PWM_MIN)))
    else:
        return neutral

# ===================== MAVLINK =====================

class MavRC:
    def __init__(self):
        self.m = mavutil.mavlink_connection(
            CONN_STR, source_system=SOURCE_SYS, source_component=SOURCE_COMP, dialect="ardupilotmega"
        )
        hb = self.m.wait_heartbeat(timeout=10.0)
        if not hb: raise RuntimeError("No MAVLink HEARTBEAT")
        self.m.target_system = TARGET_SYS
        self.m.target_component = TARGET_COMP
        self._next_dbg = 0.0

    # ---- params (safe: only touch the two output channels) ----
    def set_param(self, name, value, ptype=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                  name.encode('ascii'), float(value), ptype)

    def get_param_once(self, name, timeout=0.7):
        self.m.mav.param_request_read_send(self.m.target_system, self.m.target_component,
                                           name.encode('ascii'), -1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.m.recv_match(type='PARAM_VALUE', blocking=False)
            if not msg: time.sleep(0.02); continue
            if msg.param_id.strip('\x00') == name:
                return float(msg.param_value)
        return None

    def ensure_param_soft(self, name, wanted, tries=4):
        ok = False
        for _ in range(tries):
            cur = self.get_param_once(name)
            if cur is not None and abs(cur - float(wanted)) < 1e-3:
                ok = True; break
            self.set_param(name, wanted)
            time.sleep(0.15)
        if ok: rospy.loginfo(f"[PARAM] {name} = {wanted} (ok)")
        else:  rospy.logwarn(f"[PARAM] Tried to set {name} -> {wanted} (could not verify)")

    def map_output_to_rcin(self, servo_num, rcin_idx):
        # SERVOx_FUNCTION = 50 + n  (RCINn). Example: RCIN3 -> 53
        func = 50 + int(rcin_idx)
        self.ensure_param_soft(f"SERVO{servo_num}_FUNCTION", func)
        # Sane range/trim
        self.ensure_param_soft(f"SERVO{servo_num}_MIN", 1100.0)
        self.ensure_param_soft(f"SERVO{servo_num}_MAX", 1900.0)
        self.ensure_param_soft(f"SERVO{servo_num}_TRIM", 1500.0)
        # If present, keep not-reversed & PWM type (ignore if FW lacks these)
        if self.get_param_once(f"SERVO{servo_num}_REVERSED") is not None:
            self.ensure_param_soft(f"SERVO{servo_num}_REVERSED", 0.0)
        if self.get_param_once(f"SERVO{servo_num}_TYPE") is not None:
            self.ensure_param_soft(f"SERVO{servo_num}_TYPE", 1.0)  # 1=PWM

    # ---- mode/arm ----
    def set_mode_manual(self):
        self.m.mav.set_mode_send(TARGET_SYS, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0)

    def arm(self, val=True):
        self.m.mav.command_long_send(
            TARGET_SYS, TARGET_COMP,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1.0 if val else 0.0, 0,0,0,0,0,0
        )

    def disarm(self): self.arm(False)

    # ---- RC override ----
    def rc_override(self, rcin_vals):
        """rcin_vals: dict like {1:1500, 3:1520} -> RCIN1, RCIN3"""
        vals = [65535]*8
        for ch, pwm in rcin_vals.items():
            if 1 <= ch <= 8:
                vals[ch-1] = int(pwm)
        self.m.mav.rc_channels_override_send(self.m.target_system, self.m.target_component, *vals)

    def clear_override(self):
        self.m.mav.rc_channels_override_send(self.m.target_system, self.m.target_component,
                                             65535,65535,65535,65535,65535,65535,65535,65535)

    def release_override(self, repeats=5, dt=0.05):
        for _ in range(repeats):
            self.clear_override()
            time.sleep(dt)

    # ---- telemetry ----
    def poll_rcin_rcout(self):
        data = {}
        msg = self.m.recv_match(type=['RC_CHANNELS','RC_CHANNELS_RAW'], blocking=False)
        if msg:
            if msg.get_type() == 'RC_CHANNELS_RAW':
                for i in range(1,9): data[f'rcin{i}'] = getattr(msg, f'chan{i}_raw', None)
            else:
                data.update({
                    'rcin1': msg.chan1_raw, 'rcin2': msg.chan2_raw, 'rcin3': msg.chan3_raw, 'rcin4': msg.chan4_raw,
                    'rcin5': msg.chan5_raw, 'rcin6': msg.chan6_raw, 'rcin7': msg.chan7_raw, 'rcin8': msg.chan8_raw
                })
        out = self.m.recv_match(type='SERVO_OUTPUT_RAW', blocking=False)
        if out:
            for i in range(1,15):
                data[f'servo{i}'] = getattr(out, f'servo{i}_raw', None)
        return data

    def debug_print_io(self, tag=""):
        now = time.time()
        if now < self._next_dbg: return None
        self._next_dbg = now + (1.0 / max(0.1, DEBUG_HZ))
        d = self.poll_rcin_rcout()
        if not d: return None
        sR = d.get(f'servo{RIGHT_SERVO_NUM}')
        sL = d.get(f'servo{LEFT_SERVO_NUM}')
        r1 = d.get(f'rcin{CH_RIGHT_IN}')
        r3 = d.get(f'rcin{CH_LEFT_IN}')
        if sR is not None and sL is not None:
            rospy.loginfo(f"[IO]{tag} RCIN{CH_RIGHT_IN2}={r1} RCIN{CH_LEFT_IN2}={r3} | "
                          f"RCOUT{RIGHT_SERVO_NUM}={sR} RCOUT{LEFT_SERVO_NUM}={sL}")
        else:
            rospy.loginfo(f"[IO]{tag} RCIN{CH_RIGHT_IN2}={r1} RCIN{CH_LEFT_IN2}={r3}")
        return d

# ===================== NODE =====================

class CmdVelToPWMNode:
    def __init__(self):
        self.mav = MavRC()

        # Map chosen outputs to chosen RCIN sources
        self.mav.map_output_to_rcin(RIGHT_SERVO_NUM, CH_RIGHT_IN)
        self.mav.map_output_to_rcin(LEFT_SERVO_NUM,  CH_LEFT_IN)

        self.mav.set_mode_manual()
        self.mav.arm(True)

        self.des_v = 0.0
        self.des_w = 0.0
        self.last_cmd_time = 0.0

        self.last_left_pwm  = NEUTRAL_LEFT
        self.last_right_pwm = NEUTRAL_RIGHT
        self.was_neutral    = True       # to detect neutral-entry transitions
        self.stopping       = False

        self._next_hint_t   = 0.0  # auto-detect hint cadence

        rospy.Subscriber("cmd_vel", Twist, self.cb_cmdvel, queue_size=10)

        def _cleanup(*_):
            self.stopping = True
            try: self.safe_stop()
            except Exception: pass
            try: self.mav.m.close()
            except Exception: pass
            rospy.signal_shutdown("shutdown")

        atexit.register(_cleanup)
        signal.signal(signal.SIGINT,  _cleanup)
        signal.signal(signal.SIGTERM, _cleanup)

    # ---------- mix ----------
    def mix(self, v_n, w_n):
        if abs(w_n) > TURN_ONLY_W_THRESH and abs(v_n) < LINEAR_LOCKOUT_TURN:
            s = clamp(abs(w_n), 0.0, 1.0)
            if w_n > 0: left_cmd, right_cmd = -s, s
            else:       left_cmd, right_cmd =  s,-s
        else:
            left_cmd  = clamp(v_n - TURN_GAIN * w_n, -1.0, 1.0)
            right_cmd = clamp(v_n + TURN_GAIN * w_n, -1.0, 1.0)
        return left_cmd, right_cmd

    # ---------- start/stop helpers (EXACT like test script) ----------
    def forward_flash_once(self):
        pwm = clamp(FLASH_PWM, NEUTRAL_LEFT, PWM_MAX)  # single value on both, like test
        dt = 1.0 / max(FLASH_HZ, 1.0)
        t_end = time.time() + BLIP_TIME_S
        rospy.loginfo(f"[FLASH] brief forward at {pwm} for {BLIP_TIME_S:.2f}s")
        while time.time() < t_end:
            self.mav.rc_override({CH_RIGHT_IN: pwm, CH_LEFT_IN: pwm})
            time.sleep(dt)

    def neutral_hold(self, hold_s=0.5):
        rospy.loginfo("[NEUTRAL] Holding both = 1500")
        t_end = time.time() + hold_s
        while time.time() < t_end:
            self.mav.rc_override({CH_RIGHT_IN: NEUTRAL_RIGHT, CH_LEFT_IN: NEUTRAL_LEFT})
            time.sleep(0.05)
        # DO NOT reassert after this — we leave outputs to the FC until commands resume

    def neutral_entry_stop(self, hold_s=0.7, reason=""):
        self.forward_flash_once()
        self.neutral_hold(hold_s)
        if reason: rospy.loginfo(f"[NEUTRAL] flash+hold complete ({reason})")

    def safe_stop(self):
        """
        EXACT stop sequence as test script:
         1) neutral() = flash forward + neutral hold
         2) disarm
         3) clear overrides a few times
        """
        try:
            self.neutral_entry_stop(hold_s=0.5, reason="shutdown")
            time.sleep(0.05)
            self.mav.disarm()
            time.sleep(0.1)
            self.mav.release_override(repeats=5, dt=0.05)
        except Exception:
            pass

    # ---------- ROS ----------
    def cb_cmdvel(self, msg: Twist):
        self.des_v = float(msg.linear.x)
        self.des_w = float(msg.angular.z)
        self.last_cmd_time = time.time()

    def spin(self):
        rate = rospy.Rate(RATE_HZ)
        while not rospy.is_shutdown() and not self.stopping:
            io = self.mav.debug_print_io()

            now = time.time()
            stale = (now - self.last_cmd_time) > CMD_TIMEOUT_S
            neutral_cmd = (abs(self.des_v) < 1e-6 and abs(self.des_w) < 1e-6)

            # On entering idle, do EXACT neutral sequence and then stop overriding
            if stale or neutral_cmd:
                if not self.was_neutral:
                    self.neutral_entry_stop(hold_s=0.7, reason="stale/zero")
                    self.was_neutral = True
                # remain idle: do NOT keep sending neutral; let FC own outputs
                rate.sleep()
                continue

            v_n, w_n = norm_cmd(self.des_v, self.des_w)
            left_h, right_h = self.mix(v_n, w_n)

            pivot = (abs(w_n) > TURN_ONLY_W_THRESH and abs(v_n) < LINEAR_LOCKOUT_TURN)

            # Keep both sides hull-forward on arcs (no neutral inner)
            if (v_n > 0.0) and not pivot and ARC_NO_NEUTRAL_WHEN_VPOS:
                left_h  = max(0.0, left_h)
                right_h = max(0.0, right_h)

            # Convert HULL cmd to ESC-domain via thrust directions
            left_e  = left_h  * (1 if LEFT_THRUST_DIR  >= 0 else -1)
            right_e = right_h * (1 if RIGHT_THRUST_DIR >= 0 else -1)

            # Map to PWM
            L_pre = cmd_to_pwm(left_e,  NEUTRAL_LEFT,  LEFT_FWD_START,  LEFT_REV_START)
            R_pre = cmd_to_pwm(right_e, NEUTRAL_RIGHT, RIGHT_FWD_START, RIGHT_REV_START)
            L = clamp(L_pre, PWM_MIN, PWM_MAX)
            R = clamp(R_pre, PWM_MIN, PWM_MAX)

            # Forward-arc shaping (hull-forward floors/boosts)
            if (v_n > 0.0) and not pivot:
                if w_n > 0:  # turning left: left=inner, right=outer
                    if ARC_NO_NEUTRAL_WHEN_VPOS:
                        # push left a touch into hull-forward (dir-agnostic here since we already mapped)
                        if LEFT_THRUST_DIR >= 0: L = max(L, LEFT_FWD_START + ARC_INNER_MIN_US_FWD)
                        else:                    L = min(L, LEFT_REV_START  - ARC_INNER_MIN_US_FWD)
                    # outer boost on right in hull-forward sense (we already mapped, so +µs is fine)
                    R = min(PWM_MAX, R + ARC_OUTER_EXTRA_US_FWD)
                elif w_n < 0:  # turning right: right=inner, left=outer
                    if ARC_NO_NEUTRAL_WHEN_VPOS:
                        if RIGHT_THRUST_DIR >= 0: R = max(R, RIGHT_FWD_START + ARC_INNER_MIN_US_FWD)
                        else:                     R = min(R, RIGHT_REV_START  - ARC_INNER_MIN_US_FWD)
                    L = min(PWM_MAX, L + ARC_OUTER_EXTRA_US_FWD)

            self.mav.rc_override({CH_RIGHT_IN: R, CH_LEFT_IN: L})
            self.last_left_pwm, self.last_right_pwm = L, R
            self.was_neutral = False

            # Heuristic: hint if wrong physical output is moving
            if io and now > self._next_hint_t:
                self._next_hint_t = now + 1.0
                target = L
                best_ch, best_err = None, 999
                for i in range(1,15):
                    val = io.get(f"servo{i}")
                    if val is None: continue
                    err = abs(int(val) - int(target))
                    if err < best_err:
                        best_err, best_ch = err, i
                cur_val = io.get(f"servo{LEFT_SERVO_NUM}")
                if cur_val is not None and abs(int(cur_val) - target) > 6 and best_ch not in (None, LEFT_SERVO_NUM):
                    if best_err <= 4:
                        rospy.logwarn(f"[HINT] RCOUT{LEFT_SERVO_NUM}={cur_val} not tracking left PWM {target}. "
                                      f"RCOUT{best_ch}≈{io.get(f'servo{best_ch}')} matches. "
                                      f"Try set LEFT_SERVO_NUM={best_ch} (e.g., AUX3→11).")

            rate.sleep()

def main():
    rospy.init_node("cmdvel_to_pwm_selectable", anonymous=False)
    node = CmdVelToPWMNode()
    try:
        node.spin()
    finally:
        try: node.safe_stop()
        except Exception: pass

if __name__ == "__main__":
    main()
