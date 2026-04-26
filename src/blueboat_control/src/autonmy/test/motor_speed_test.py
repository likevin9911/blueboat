#!/usr/bin/env python3
# Four-phase BlueBoat test with forward "flash" before any stop/neutral.
# Phases (5s each, 1 PWM step/sec):
#   1) both forward
#   2) both reverse
#   3) left forward + right reverse (simultaneous)
#   4) left reverse + right forward (simultaneous)

import time, signal, atexit, sys
from pymavlink import mavutil

# ================= CONFIG =================
CONN_STR     = "udpin:0.0.0.0:14550"
SOURCE_SYS   = 1
SOURCE_COMP  = 190
TARGET_SYS   = 1
TARGET_COMP  = 1

# Servo function mode per motor:
#   1 = RCPassThru
#   2 = RCIN (right uses RCIN1, left uses RCIN3)
MODE_RIGHT = 2
MODE_LEFT  = 2

# Forward thresholds (where they *start* moving)
LEFT_START_FWD  = 1526
RIGHT_START_FWD = 1532

# Reverse thresholds (where they *start* moving)
LEFT_START_REV  = 1486
RIGHT_START_REV = 1489

# PWM bounds
PWM_NEUTRAL = 1500
PWM_FWD_MAX = 1540
PWM_REV_MIN = 1470

# Channels (ArduRover BlueBoat: SERVO1=right, SERVO3=left)
CH_RIGHT, CH_LEFT = 1, 3

# Param values
FUNC = {"RCPASS": 1, "RCIN1": 51, "RCIN3": 53}

# Phase timing
PHASE_SEC  = 5          # each sub-phase lasts 5 seconds
STEP_SEC   = 1.0        # 1 PWM increment per second
RAMP_STEPS = int(PHASE_SEC / STEP_SEC)

# Forward "flash" before any stop (helps ESCs exit reverse/brake)
FLASH_PWM   = 1532
FLASH_TIME  = 0.25      # seconds
FLASH_HZ    = 20.0

# Extra-robust shutdown tuning
CLEAR_REPEATS_BEFORE_DISARM = 10
CLEAR_REPEATS_AFTER_DISARM  = 10
CLEAR_DT_S                  = 0.05
FINAL_NEUTRAL_HOLD_S        = 0.4

# ==========================================

def clamp(v, lo, hi): return max(lo, min(hi, v))

def connect():
    print("==============================================================")
    print("[STEP 1] CONNECT")
    print("==============================================================")
    print(f"[CONNECT] {CONN_STR}")
    m = mavutil.mavlink_connection(CONN_STR, source_system=SOURCE_SYS,
                                   source_component=SOURCE_COMP, dialect="ardupilotmega")
    hb = m.wait_heartbeat(timeout=10.0)
    if not hb: raise RuntimeError("No HEARTBEAT")
    print(f"[CONNECT] HEARTBEAT from system={hb.get_srcSystem()} component={hb.get_srcComponent()}")
    m.target_system = TARGET_SYS
    m.target_component = TARGET_COMP
    return m

def set_param(master, name, value, ptype=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
    master.mav.param_set_send(master.target_system, master.target_component,
                              name.encode('ascii'), float(value), ptype)

def apply_servo_functions(master, mode_right, mode_left):
    # Only two options: RCPassThru or RCIN (right=RCIN1, left=RCIN3)
    fr = FUNC["RCPASS"] if mode_right == 1 else FUNC["RCIN1"]
    fl = FUNC["RCPASS"] if mode_left  == 1 else FUNC["RCIN3"]
    print("==============================================================")
    print("[STEP 2] SERVO FUNCTION MODE")
    print("==============================================================")
    print(f"[SERVO-FUNC] SERVO1 (right) = {fr}  |  SERVO3 (left) = {fl}")
    set_param(master, "SERVO1_FUNCTION", fr)
    set_param(master, "SERVO3_FUNCTION", fl)

def set_mode_manual(m):
    print("==============================================================")
    print("[STEP 3] MODE + ARM")
    print("==============================================================")
    print("[MODE] Request MANUAL …")
    m.mav.set_mode_send(TARGET_SYS, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0)
    # best-effort confirm
    t0 = time.time()
    ok = False
    while time.time() - t0 < 3.0:
        hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
        if hb and getattr(hb, "custom_mode", None) == 0:
            ok = True
            break
        m.mav.set_mode_send(TARGET_SYS, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0)
    print(f"[MODE] MANUAL set -> {ok}")

def arm(m, val=True):
    print(f"[ARM] Request Arm({val}) …")
    m.mav.command_long_send(TARGET_SYS, TARGET_COMP,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                            1.0 if val else 0.0,0,0,0,0,0,0)
    # best-effort confirm
    t0 = time.time()
    while time.time() - t0 < 5.0:
        hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
        if not hb: continue
        armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
        if armed == val:
            print(f"[ARM] Confirmed armed={armed}")
            return
    print("[ARM] Timed out (check failsafes)")

def rc_override(m, ch1=None, ch3=None):
    vals = [65535]*8
    if ch1 is not None: vals[0] = int(ch1)  # right
    if ch3 is not None: vals[2] = int(ch3)  # left
    m.mav.rc_channels_override_send(m.target_system, m.target_component, *vals)

def clear_overrides(m, repeats, dt):
    for _ in range(repeats):
        rc_override(m, None, None)  # 65535 to both -> release
        time.sleep(dt)

def forward_flash(m, pwm=FLASH_PWM, secs=FLASH_TIME):
    pwm = clamp(pwm, PWM_NEUTRAL, PWM_FWD_MAX)
    print(f"[FLASH] brief forward at {pwm} for {secs:.2f}s")
    dt = 1.0 / max(FLASH_HZ, 1.0)
    t_end = time.time() + secs
    while time.time() < t_end:
        rc_override(m, pwm, pwm)
        time.sleep(dt)

def neutral(m, secs=0.5):
    # ALWAYS flash forward before neutral (original behavior)
    forward_flash(m)
    print("[NEUTRAL] Returning both to 1500")
    t_end = time.time() + secs
    while time.time() < t_end:
        rc_override(m, PWM_NEUTRAL, PWM_NEUTRAL)
        time.sleep(0.05)

def stop_all(m):
    print("==============================================================")
    print("[STOP] ROBUST SHUTDOWN: neutral → clear → disarm → clear → final neutral")
    print("==============================================================")
    try:
        # 1) Original neutral sequence
        neutral(m, 0.7)

        # 2) Clear overrides BEFORE disarm (gives FC back control cleanly)
        clear_overrides(m, CLEAR_REPEATS_BEFORE_DISARM, CLEAR_DT_S)

        # 3) Disarm
        arm(m, False)

        # 4) Clear overrides AFTER disarm (some firmwares keep last override latched)
        clear_overrides(m, CLEAR_REPEATS_AFTER_DISARM, CLEAR_DT_S)

        # 5) Final explicit neutral hold (quiet some ESCs that sing at 1500 edge)
        t_end = time.time() + FINAL_NEUTRAL_HOLD_S
        while time.time() < t_end:
            rc_override(m, PWM_NEUTRAL, PWM_NEUTRAL)
            time.sleep(0.05)

        # 6) And release once more
        clear_overrides(m, 3, CLEAR_DT_S)

    except Exception as e:
        print(f"[STOP] Exception during shutdown: {e}")

# ================= TEST PHASES =================

def ramp_both_forward(m):
    print("\n[TEST 1] BOTH FORWARD (5s)")
    for i in range(RAMP_STEPS):
        L = clamp(LEFT_START_FWD  + i, LEFT_START_FWD,  PWM_FWD_MAX)
        R = clamp(RIGHT_START_FWD + i, RIGHT_START_FWD, PWM_FWD_MAX)
        print(f"[RC] L={L}  R={R}")
        rc_override(m, R, L)
        time.sleep(STEP_SEC)

def ramp_both_reverse(m):
    print("\n[TEST 2] BOTH REVERSE (5s)")
    for i in range(RAMP_STEPS):
        L = clamp(LEFT_START_REV  - i, PWM_REV_MIN,  LEFT_START_REV)
        R = clamp(RIGHT_START_REV - i, PWM_REV_MIN, RIGHT_START_REV)
        print(f"[RC] L={L}  R={R}")
        rc_override(m, R, L)
        time.sleep(STEP_SEC)

def ramp_left_fwd_right_rev(m):
    print("\n[TEST 3] LEFT FWD  + RIGHT REV (simultaneous, 5s)")
    for i in range(RAMP_STEPS):
        L = clamp(LEFT_START_FWD  + i, LEFT_START_FWD,  PWM_FWD_MAX)
        R = clamp(RIGHT_START_REV - i, PWM_REV_MIN,    RIGHT_START_REV)
        print(f"[RC] L={L}  R={R}")
        rc_override(m, R, L)
        time.sleep(STEP_SEC)

def ramp_left_rev_right_fwd(m):
    print("\n[TEST 4] LEFT REV  + RIGHT FWD (simultaneous, 5s)")
    for i in range(RAMP_STEPS):
        L = clamp(LEFT_START_REV  - i, PWM_REV_MIN,   LEFT_START_REV)
        R = clamp(RIGHT_START_FWD + i, RIGHT_START_FWD, PWM_FWD_MAX)
        print(f"[RC] L={L}  R={R}")
        rc_override(m, R, L)
        time.sleep(STEP_SEC)

# ================= MAIN =================

def main():
    m = connect()
    apply_servo_functions(m, MODE_RIGHT, MODE_LEFT)
    set_mode_manual(m)
    arm(m, True)

    def _cleanup(*_):
        stop_all(m)
        try: m.close()
        except Exception: pass
        sys.exit(0)

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        # Run tests; BEFORE any stop/neutral, we do forward_flash inside neutral()
        ramp_both_forward(m);        neutral(m, 0.7)
        ramp_both_reverse(m);        neutral(m, 0.7)
        ramp_left_fwd_right_rev(m);  neutral(m, 0.7)
        ramp_left_rev_right_fwd(m);  neutral(m, 0.7)
    finally:
        _cleanup()

if __name__ == "__main__":
    main()
