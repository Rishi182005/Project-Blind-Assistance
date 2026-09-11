import socket

UDP_IP = "0.0.0.0"
UDP_PORT = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Listening for ESP32 UDP packets on port 4210...")

while True:
    data, addr = sock.recvfrom(1024)
    print(f"From {addr}: {data.decode()}")