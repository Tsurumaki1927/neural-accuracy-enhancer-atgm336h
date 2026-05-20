import socket
import re
import csv
import time
from datetime import datetime
import serial # 如果GPS模块接电脑USB串口，需要安装pyserial库

# ---------------------- 配置项 ----------------------
# UDP接收手机数据配置
UDP_IP = "0.0.0.0"
UDP_PORT = 6666
# 串口GPS配置（如果GPS是UDP上报就删掉这部分，改成UDP接收）
GPS_SERIAL_PORT = "COM6" # 替换为你的GPS串口号
GPS_BAUDRATE = 9600
# 训练集存储路径
SAVE_PATH = "gps_train_dataset.csv"
# 时间对齐阈值：两个数据时间差小于500ms则配对
TIME_THRESHOLD = 0.5

# ---------------------- 初始化 ----------------------
# 初始化UDP服务
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_sock.bind((UDP_IP, UDP_PORT))
udp_sock.setblocking(False)
# 初始化GPS串口
gps_serial = serial.Serial(GPS_SERIAL_PORT, GPS_BAUDRATE, timeout=0.1)
# 初始化CSV文件
with open(SAVE_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "gps_lon", "gps_lat", "gps_alt", "gps_sat_num", "true_lon", "true_lat", "true_alt"])

# 缓存最近的GPS数据和手机数据
latest_gps = None
latest_phone = None

# ---------------------- 解析函数 ----------------------
# 解析手机数据
def parse_phone_data(raw):
    match = re.match(r"@(.*?)@#(.*?)#\*(.*?)\*", raw.decode("utf-8", errors="ignore"))
    if match:
        return {
            "time": time.time(),
            "lon": float(match.group(1)),
            "lat": float(match.group(2)),
            "alt": float(match.group(3))
        }
    return None

# 解析NMEA GGA语句
def parse_nmea_gga(raw):
    raw = raw.decode("utf-8", errors="ignore").strip()
    if not raw.startswith("$GNGGA") and not raw.startswith("$GPGGA"):
        return None
    parts = raw.split(",")
    if len(parts) < 15 or parts[6] == "0": # 定位无效
        return None
    # 转十进制度
    lat = float(parts[2][:2]) + float(parts[2][2:])/60
    if parts[3] == "S": lat = -lat
    lon = float(parts[4][:3]) + float(parts[4][3:])/60
    if parts[5] == "W": lon = -lon
    alt = float(parts[9])
    sat_num = int(parts[7])
    return {
        "time": time.time(),
        "lon": lon,
        "lat": lat,
        "alt": alt,
        "sat_num": sat_num
    }

# ---------------------- 主循环 ----------------------
print("采集已启动，按Ctrl+C停止...")
while True:
    try:
        # 读GPS数据
        if gps_serial.in_waiting:
            nmea_line = gps_serial.readline()
            gps_data = parse_nmea_gga(nmea_line)
            if gps_data:
                latest_gps = gps_data

        # 读手机UDP数据
        try:
            phone_data, _ = udp_sock.recvfrom(1024)
            phone_data = parse_phone_data(phone_data)
            if phone_data:
                latest_phone = phone_data
        except BlockingIOError:
            pass

        # 时间对齐配对
        if latest_gps and latest_phone:
            time_diff = abs(latest_gps["time"] - latest_phone["time"])
            if time_diff < TIME_THRESHOLD:
                # 写入CSV
                with open(SAVE_PATH, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.fromtimestamp(latest_gps["time"]).strftime("%Y-%m-%d %H:%M:%S.%f"),
                        round(latest_gps["lon"],7), round(latest_gps["lat"],7), round(latest_gps["alt"],3), latest_gps["sat_num"],
                        round(latest_phone["lon"],7), round(latest_phone["lat"],7), round(latest_phone["alt"],3)
                    ])
                print(f"配对成功：GPS[{latest_gps['lon']},{latest_gps['lat']}] | 真值[{latest_phone['lon']},{latest_phone['lat']}]")
                # 清空缓存等待下一组
                latest_gps = None
                latest_phone = None

    except KeyboardInterrupt:
        print(f"采集停止，数据已保存到{SAVE_PATH}")
        break