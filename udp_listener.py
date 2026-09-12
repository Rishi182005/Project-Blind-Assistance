import socket

UDP_IP = "0.0.0.0"
UDP_PORT = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

latest_distance_cm = None

print("Listening for ESP32 ultrasonic data...")

while True:
    data, addr = sock.recvfrom(1024)

    message = data.decode().strip()

    # Expected:
    # distance_cm:13.1,ts:18631

    try:
        parts = message.split(",")

        distance_cm = float(parts[0].split(":")[1])
        timestamp = int(parts[1].split(":")[1])

        latest_distance_cm = distance_cm

        print(
            f"Ultrasonic: {latest_distance_cm:.1f} cm | "
            f"ESP32 timestamp: {timestamp}"
        )

    except (ValueError, IndexError):
        print(f"Invalid packet: {message}")