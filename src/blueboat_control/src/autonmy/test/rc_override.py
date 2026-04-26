#!/usr/bin/env python3
# bb_1535_simple.py — set both motors to 1535 for 3s with built-in sanity checks (fixed param_id handling)
import time, argparse
from pymavlink import mavutil

PWM_MIN, PWM_NEUTRAL, PWM_MAX = 1100, 1500, 1900
CMD_PWM = 1535
LEFT_CH, RIGHT_CH = 3, 1  # BlueBoat mapping

def clamp(v, lo, hi): return max(lo, min(hi, v))

def _param_id_to_str(x):
    # pymavlink may give bytes or str depending on transport/version
    if isinstance(x, bytes):
        return x.decode('utf-8', errors='ignore').strip('\x00')
    return str(x).strip('\x00')

def connect(conn, sysid, compid, hb_timeout):
    print(f"[CONNECT] {conn}")
    m = mavutil.mavlink_connection(conn, source_system=sysid, source_component=compid, dialect="ardupilotmega")
    print("[CONNECT] Waiting for HEARTBEAT…")
    hb = m.wait_heartbeat(timeout=hb_timeout)
    if not hb: raise RuntimeError("No HEARTBEAT")
    print(f"[CONNECT] HEARTBEAT from system={hb.get_srcSystem()} component={hb.get_srcComponent()} type={getattr(hb,'type',None)}")
    # Prefer autopilot (compid 1) if we see it shortly
    t0 = time.time()
    while time.time() - t0 < 2.0:
        hb2 = m.recv_match(type='HEARTBEAT', blocking=True, timeout=0.3)
        if not hb2: continue
        if hb2.get_srcComponent() == 1:  # MAV_COMP_ID_AUTOPILOT1
            print(f"[CONNECT] Autopilot heartbeat seen (sys={hb2.get_srcSystem()} comp=1) — selecting it.")
            m.target_system = hb2.get_srcSystem()
            m.target_component = 1
            return m
    # fallback to the first heartbeat source
    m.target_system = hb.get_srcSystem()
    m.target_component = hb.get_srcComponent()
    return m

def set_mode_manual(m, target_sys=1, retry_sec=3.0):
    MANUAL_ID = 0
    print("[MODE] Request MANUAL…")
    m.mav.set_mode_send(target_sys, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, MANUAL_ID)
    t0 = time.time(); ok = False
    while time.time()-t0 < retry_sec:
        hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
        if hb and getattr(hb, "custom_mode", None) == MANUAL_ID:
            ok = True; break
        m.mav.set_mode_send(target_sys, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, MANUAL_ID)
    print(f"[MODE] MANUAL set -> {ok}")
    return ok

def arm(m, value, target_sys=1, target_comp=1, timeout=6.0):
    print(f"[ARM] Request Arm({value})…")
    m.mav.command_long_send(target_sys, target_comp, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                            0, 1.0 if value else 0.0, 0,0,0,0,0,0)
    t0 = time.time()
    while time.time()-t0 < timeout:
        hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
        if not hb: continue
        armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
        if armed == value:
            print(f"[ARM] Confirmed armed={armed}")
            return armed
    print("[ARM] Timed out (check prearm/failsafes).")
    return False

def get_param(m, name, timeout=2.0):
    m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
    t0 = time.time()
    while time.time()-t0 < timeout:
        msg = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.4)
        if not msg: continue
        if _param_id_to_str(msg.param_id) == name:
            return int(msg.param_value) if msg.param_type in (6,7,9) else msg.param_value
    return None

def set_param_int(m, name, val):
    print(f"[PARAM] {name} -> {val}")
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(), float(val),
                         mavutil.mavlink.MAV_PARAM_TYPE_INT32)

def send_rc(m, left_pwm, right_pwm):
    vals = [65535]*8
    vals[LEFT_CH-1]  = left_pwm
    vals[RIGHT_CH-1] = right_pwm
    m.mav.rc_channels_override_send(m.target_system, m.target_component, *vals)
    print(f"[RC ] ch{LEFT_CH}={left_pwm}  ch{RIGHT_CH}={right_pwm}")

def send_servo(m, ch, pwm):
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                            0, float(ch), float(pwm), 0,0,0,0,0)
    print(f"[SER] DO_SET_SERVO ch{ch} -> {pwm}")

def main():
    ap = argparse.ArgumentParser(description="BlueBoat ultra-simple 1535 test with checks")
    ap.add_argument("--conn", default="udpin:0.0.0.0:14550")
    ap.add_argument("--gcs-sysid", type=int, default=1)
    ap.add_argument("--gcs-compid", type=int, default=190)
    ap.add_argument("--target-sys", type=int, default=1)
    ap.add_argument("--target-comp", type=int, default=1)
    ap.add_argument("--hb-timeout", type=float, default=8.0)
    ap.add_argument("--duration", type=float, default=3.0)
    args = ap.parse_args()

    pwm = clamp(CMD_PWM, PWM_MIN, 1540)

    print("==============================================================")
    print("[STEP 1] CONNECT")
    print("==============================================================")
    m = connect(args.conn, args.gcs_sysid, args.gcs_compid, args.hb_timeout)

    # Ensure we’re pointing at the flight controller (if user passed explicit targets, respect them)
    m.target_system = args.target_sys or m.target_system
    m.target_component = args.target_comp or m.target_component
    print(f"[TARGET] sys={m.target_system} comp={m.target_component}")

    print("==============================================================")
    print("[STEP 2] ID + PARAM CHECKS")
    print("==============================================================")
    sysid_mygcs = get_param(m, "SYSID_MYGCS")
    if sysid_mygcs is not None:
        print(f"[CHECK] SYSID_MYGCS = {sysid_mygcs}")
        if sysid_mygcs != args.gcs_sysid:
            set_param_int(m, "SYSID_MYGCS", args.gcs_sysid)
            time.sleep(0.3)
    else:
        print("[WARN] Could not read SYSID_MYGCS (continuing).")

    set_param_int(m, "MOT_PWM_TYPE", 0)           # normal PWM
    set_param_int(m, "SERVO1_FUNCTION", 74)       # ThrottleRight
    set_param_int(m, "SERVO3_FUNCTION", 73)       # ThrottleLeft
    time.sleep(0.4)

    print("==============================================================")
    print("[STEP 3] MODE + ARM")
    print("==============================================================")
    set_mode_manual(m, target_sys=m.target_system)
    arm(m, True, target_sys=m.target_system, target_comp=m.target_component)

    print("==============================================================")
    print("[STEP 4] WAKE @1500/1500 (~0.6s)")
    print("==============================================================")
    for _ in range(6):
        send_servo(m, LEFT_CH,  PWM_NEUTRAL)
        send_servo(m, RIGHT_CH, PWM_NEUTRAL)
        send_rc(m, PWM_NEUTRAL, PWM_NEUTRAL)
        time.sleep(0.1)

    print("==============================================================")
    print("[STEP 5] COMMAND BOTH @1535 for 3s (RC + DO_SET_SERVO)")
    print("==============================================================")
    # Temporarily flip to passthrough so DO_SET_SERVO isn't rejected by function ownership
    set_param_int(m, "SERVO1_FUNCTION", 1)
    set_param_int(m, "SERVO3_FUNCTION", 1)
    time.sleep(0.2)

    t_end = time.time() + args.duration
    while time.time() < t_end:
        send_servo(m, LEFT_CH,  pwm)
        send_servo(m, RIGHT_CH, pwm)
        send_rc(m, pwm, pwm)
        time.sleep(0.1)

    print("==============================================================")
    print("[STEP 6] NEUTRAL + RESTORE + DISARM")
    print("==============================================================")
    for _ in range(6):
        send_servo(m, LEFT_CH,  PWM_NEUTRAL)
        send_servo(m, RIGHT_CH, PWM_NEUTRAL)
        send_rc(m, PWM_NEUTRAL, PWM_NEUTRAL)
        time.sleep(0.1)

    set_param_int(m, "SERVO1_FUNCTION", 74)  # restore BlueBoat mapping
    set_param_int(m, "SERVO3_FUNCTION", 73)
    time.sleep(0.2)

    arm(m, False, target_sys=m.target_system, target_comp=m.target_component)
    print("[DONE] Complete. Closing link.")
    try: m.close()
    except Exception: pass

if __name__ == "__main__":
    main()
