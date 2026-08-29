"""
UDP Listener — Receives HC-SR04 Distance Readings from ESP32
================================================================

WHAT THIS DOES:
Listens on your laptop for the UDP packets your ESP32 firmware sends
once the HC-SR04 is wired in (this Saturday). Each packet looks like:

  distance_cm:23.4,ts:154823

This script receives that, parses it, and prints a clean running log --
this is the first piece of the ESP32 <-> laptop communication pipeline
you'll build on for the full system (right now it just listens and
prints; later this same pattern feeds into your YOLO+GRU pipeline).

HOW TO USE:
1. Make sure your laptop and ESP32 are on the same WiFi network
   (confirmed already: both on 192.168.29.x)
2. Run this script BEFORE wiring in the HC-SR04, so it's already
   listening the moment real data starts flowing
3. Once HC-SR04 is wired Saturday, readings should start appearing here
   automatically -- no code changes needed on either side

TESTING RIGHT NOW (before HC-SR04 arrives):
You can test this listener works even without the sensor, using the
included test sender at the bottom of this file. Run this script in
one terminal, then in a SECOND terminal run:
  python udp_listener.py --test
This sends a few fake packets to itself so you can confirm the parsing
logic works before real hardware is involved.

No extra packages needed -- uses only Python's built-in socket module.
"""

import socket
import sys
import time
from datetime import datetime

# ---------- Config ----------
LISTEN_IP = "0.0.0.0"   # listen on all network interfaces
LISTEN_PORT = 4210      # MUST match LAPTOP_PORT in the ESP32 firmware
BUFFER_SIZE = 1024


def parse_message(raw_message):
    """Parses 'distance_cm:23.4,ts:154823' into a dict.
    Returns None if the message doesn't match the expected format
    (e.g. corrupted packet, or something unexpected sent to this port)."""
    try:
        parts = raw_message.strip().split(",")
        parsed = {}
        for part in parts:
            key, value = part.split(":")
            parsed[key] = float(value)
        if "distance_cm" not in parsed:
            return None
        return parsed
    except (ValueError, IndexError):
        return None


def run_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))

    print(f"Listening for ESP32 UDP packets on port {LISTEN_PORT}...")
    print("(Waiting for HC-SR04 readings -- nothing will appear until")
    print(" the sensor is wired in and sending real data)")
    print("Press Ctrl+C to stop.\n")

    packet_count = 0
    last_packet_time = None

    try:
        while True:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            raw_message = data.decode("utf-8", errors="replace")
            now = time.time()

            parsed = parse_message(raw_message)
            timestamp_str = datetime.now().strftime("%H:%M:%S")

            if parsed is None:
                print(f"[{timestamp_str}] Received unparseable packet from {addr[0]}: {raw_message!r}")
                continue

            packet_count += 1
            gap_str = ""
            if last_packet_time is not None:
                gap_ms = (now - last_packet_time) * 1000
                gap_str = f"  (+{gap_ms:.0f}ms since last)"
            last_packet_time = now

            distance = parsed["distance_cm"]
            print(f"[{timestamp_str}] #{packet_count:5d}  distance={distance:6.1f}cm  from {addr[0]}{gap_str}")

    except KeyboardInterrupt:
        print(f"\nStopped. Received {packet_count} packets total.")
    finally:
        sock.close()


def run_test_sender():
    """Sends a few fake packets to localhost, to verify the listener's
    parsing logic works correctly BEFORE the HC-SR04 is wired in.
    Run 'python udp_listener.py' in one terminal first, then
    'python udp_listener.py --test' in a second terminal."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    test_distances = [150.0, 120.5, 90.2, 60.0, 30.5, 10.2]

    print("Sending 6 fake test packets to localhost:4210...")
    for i, dist in enumerate(test_distances):
        message = f"distance_cm:{dist},ts:{int(time.time() * 1000)}"
        sock.sendto(message.encode("utf-8"), ("127.0.0.1", LISTEN_PORT))
        print(f"  Sent: {message}")
        time.sleep(0.3)

    sock.close()
    print("\nDone. Check the OTHER terminal (running the listener) --")
    print("you should see 6 parsed distance readings appear there.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_test_sender()
    else:
        run_listener()
