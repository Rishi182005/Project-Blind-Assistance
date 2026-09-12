import socket
import threading

UDP_IP = "0.0.0.0"
UDP_PORT = 4210

latest_distance_cm = None
latest_timestamp = None

def udp_receiver():
    global latest_distance_cm, latest_timestamp

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Ultrasonic receiver listening on UDP {UDP_PORT}")

    while True:
        data, addr = sock.recvfrom(1024)

        try:
            message = data.decode().strip()

            parts = message.split(",")

            distance_cm = float(parts[0].split(":")[1])
            timestamp = int(parts[1].split(":")[1])

            latest_distance_cm = distance_cm
            latest_timestamp = timestamp

        except (ValueError, IndexError):
            print("Invalid packet:", data)


receiver_thread = threading.Thread(
    target=udp_receiver,
    daemon=True
)

receiver_thread.start()

print("Receiver started.")

while True:
    if latest_distance_cm is not None:
        print(f"Latest ultrasonic distance: {latest_distance_cm:.1f} cm")
    
    # Don't print too quickly
    import time
    time.sleep(0.05)