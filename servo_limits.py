import os, time, termios, yaml
from ament_index_python.packages import get_package_share_directory
PORT = '/dev/ttyACM0'
JOINTS = ['Shoulder_Rotation','Shoulder_Pitch','Elbow','Wrist_Pitch','Wrist_Roll','Gripper']
ADDR_MIN, ADDR_MAX = 9, 11
def open_serial(port):
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    a = termios.tcgetattr(fd)
    cc = list(a[6])
    cflag = termios.CS8 | termios.CLOCAL | termios.CREAD
    cc[termios.VMIN] = 0; cc[termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW,
                      [0, 0, cflag, 0, termios.B1000000, termios.B1000000, cc])
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd
def chk(b): return (~sum(b)) & 0xFF
def read_word(fd, sid, addr):
    body = [sid, 4, 0x02, addr, 2]
    termios.tcflush(fd, termios.TCIFLUSH)
    os.write(fd, bytes([0xFF, 0xFF] + body + [chk(body)]))
    r = b''; t = time.time()
    while len(r) < 8 and time.time() - t < 0.3:
        try: c = os.read(fd, 8 - len(r))
        except OSError: c = b''
        if c: r += c
        else: time.sleep(0.005)
    if len(r) >= 7 and r[0] == 0xFF and r[1] == 0xFF:
        return r[5] + (r[6] << 8)
    return None
def main():
    cfg = get_package_share_directory('so_arm_100_hardware') + '/config/calibration.yaml'
    cal = yaml.safe_load(open(cfg))['joints']
    fd = open_serial(PORT)
    for i, name in enumerate(JOINTS):
        sid = i + 1
        mn = read_word(fd, sid, ADDR_MIN)
        mx = read_word(fd, sid, ADDR_MAX)
        c = cal.get(name, {})
        cmin = c.get('min', {}).get('ticks')
        cmax = c.get('max', {}).get('ticks')
        if None in (mn, mx, cmin, cmax):
            print(name, "id", sid, "limit", mn, mx, "calib", cmin, cmax, "нет данных")
            continue
        lo, hi = min(cmin, cmax), max(cmin, cmax)
        flag = "CLAMPS" if (mn > lo or mx < hi) else "ok"
        print("%-18s id%d  limit[%s..%s]  calib[%d..%d]  %s"
              % (name, sid, mn, mx, lo, hi, flag))
    os.close(fd)
if __name__ == '__main__':
    main()