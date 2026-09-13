"""
YOLOv8n webcam/video scaffold for the wearable nav system -- MULTI-OBJECT
TRACKING VERSION.

Difference from the single-nearest-object scaffold: instead of collapsing
to "whichever object is closest right now" every single frame, this keeps
a small set of tracked objects across frames (matched by centroid distance
+ class), computes closing speed PER OBJECT, estimates time-to-collision
(TTC) per object, and selects whichever tracked object currently has the
LOWEST TTC as "the" obstacle fed to the GRU.

Why this matters: a fast object far away (e.g. a car approaching quickly)
can be more urgent than a slow/stationary object that's physically closer
(e.g. a pole 2m away). Picking by raw nearest-distance misses this --
picking by TTC catches it.

Still 100% software -- no new hardware required. YOLO distance uses the
existing calibrated bbox-size heuristic. MiDaS-only distance uses raw MiDaS
inverse-depth with a separate online metric calibration learned from matched
YOLO+MiDaS objects.

pip install ultralytics opencv-python --break-system-packages
"""

import time
import threading
from collections import deque
import numpy as np
import cv2
from tensorflow import keras
import openvino as ov
from pathlib import Path
import socket
import queue
import pyttsx3
import subprocess
import os
import sys

try:
    from groq import Groq
except Exception:
    Groq = None

# ---- laptop audio safety layer ----
SPEECH_ENABLED = True
SPEECH_RATE = 150
# Piper local neural TTS settings. Voice files are downloaded once into the project folder.
PIPER_MODEL_FILE = "en_US-lessac-medium.onnx"
PIPER_LENGTH_SCALE = 1.075
SPEECH_MIN_REPEAT_S = 1.8
SPEECH_MEDIUM_REPEAT_S = 2.5
SPEECH_DIRECTION_STABLE_S = 0.8
SPEECH_LEFT_ZONE_FRAC = 0.34
SPEECH_RIGHT_ZONE_FRAC = 0.66

# ---- event-triggered LLM navigation layer ----
# Groq is a secondary natural-language layer. It NEVER makes the safety
# decision; the deterministic sensor-fusion + GRU path remains authoritative.
GROQ_ENABLED = True
GROQ_MODEL = "qwen/qwen3.6-27b"
GROQ_TIMEOUT_S = 1.2
GROQ_MAX_TOKENS = 50
GROQ_DISTANCE_EVENT_M = 0.60
GROQ_MIN_EVENT_INTERVAL_S = 2.00


class SpeechManager:
    """Non-blocking local neural TTS that always speaks the newest warning."""

    def __init__(self):
        # Only one pending warning is useful: stale warnings must never build up.
        self.queue = queue.Queue(maxsize=1)
        self.last_message = None
        self.last_bucket = "LOW"
        self.last_direction = None
        self.last_spoken_time = 0.0
        self.direction_candidate = None
        self.direction_candidate_since = 0.0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._worker,
            name="SpeechWorker",
            daemon=True,
        )
        self.thread.start()
        print(f"Speech: Piper neural voice ready | model={PIPER_MODEL_FILE} | length_scale={PIPER_LENGTH_SCALE}")

    def _start_speak(self, message):
        """Start one local Piper neural-TTS utterance and return its process."""
        if os.name == "nt":
            env = os.environ.copy()
            env["WEARABLE_TTS_TEXT"] = str(message)

            # Keep synthesis + playback inside one child process so the existing
            # speech worker can still terminate an old warning immediately when
            # a higher-severity warning arrives.
            piper_code = """
import os
import tempfile
import wave
import winsound
from pathlib import Path
from piper import PiperVoice, SynthesisConfig

text = os.environ.get("WEARABLE_TTS_TEXT", "").strip()
model_path = Path(os.environ.get("WEARABLE_PIPER_MODEL", "en_US-lessac-medium.onnx"))
length_scale = float(os.environ.get("WEARABLE_PIPER_LENGTH_SCALE", "1.075"))

if not text:
    raise SystemExit(0)

if not model_path.exists():
    raise FileNotFoundError(f"Piper voice model not found: {model_path}")

voice = PiperVoice.load(model_path)
fd, wav_path = tempfile.mkstemp(prefix="wearable_piper_", suffix=".wav")
os.close(fd)

try:
    with wave.open(wav_path, "wb") as wav_file:
        voice.synthesize_wav(
            text,
            wav_file,
            syn_config=SynthesisConfig(length_scale=length_scale),
        )
    winsound.PlaySound(wav_path, winsound.SND_FILENAME)
finally:
    try:
        os.remove(wav_path)
    except OSError:
        pass
"""

            env["WEARABLE_PIPER_MODEL"] = str(Path(PIPER_MODEL_FILE))
            env["WEARABLE_PIPER_LENGTH_SCALE"] = str(PIPER_LENGTH_SCALE)

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            return subprocess.Popen(
                [sys.executable, "-c", piper_code],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )

        # Non-Windows fallback: not used on the target Windows laptop.
        engine = pyttsx3.init()
        engine.setProperty("rate", SPEECH_RATE)
        engine.say(message)
        engine.runAndWait()
        engine.stop()
        return None

    def _worker(self):
        current_process = None
        current_item = None

        while self.running:
            if current_process is None:
                try:
                    item = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break

                message, bucket, direction = item
                try:
                    print(f"Speech: speaking -> {message}")
                    # Start the cooldown when speech starts, so repeated frames
                    # cannot create a backlog while this sentence is playing.
                    with self.lock:
                        self.last_spoken_time = time.monotonic()
                    current_process = self._start_speak(message)
                    current_item = item
                except Exception as exc:
                    print(f"Speech ERROR: {exc}")
                    current_process = None
                    current_item = None
                continue

            # While speech is playing, continuously look for a newer warning.
            # If one arrives, stop the old utterance and immediately speak the new one.
            if current_process.poll() is not None:
                print("Speech: done")
                current_process = None
                current_item = None
                continue

            try:
                newer = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if newer is None:
                try:
                    current_process.terminate()
                    current_process.wait(timeout=0.5)
                except Exception:
                    pass
                current_process = None
                break

            # A newer feed exists. Keep the current sentence intact so it is
            # not cut off after only the first word. The newest warning stays
            # in the one-item queue and will be spoken as soon as the current
            # sentence finishes. If severity increases, interrupt immediately.
            _old_message, old_bucket, _old_direction = current_item
            _new_message, new_bucket, _new_direction = newer
            severity = {"MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

            if severity.get(new_bucket, 0) > severity.get(old_bucket, 0):
                try:
                    current_process.terminate()
                    current_process.wait(timeout=0.5)
                except Exception:
                    try:
                        current_process.kill()
                    except Exception:
                        pass

                current_process = None
                current_item = None
                message, bucket, direction = newer
                try:
                    print(f"Speech: urgent latest -> {message}")
                    with self.lock:
                        self.last_spoken_time = time.monotonic()
                    current_process = self._start_speak(message)
                    current_item = newer
                except Exception as exc:
                    print(f"Speech ERROR: {exc}")
            else:
                # Put the newest warning back so it replaces any stale item
                # and is spoken immediately after the current sentence.
                try:
                    while True:
                        self.queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.queue.put_nowait(newer)
                    print(f"Speech: pending latest -> {newer[0]}")
                except queue.Full:
                    pass

        if current_process is not None:
            try:
                current_process.terminate()
                current_process.wait(timeout=0.5)
            except Exception:
                pass

    def _direction(self, selected, frame_width, unknown_zone=""):
        if selected is not None and getattr(selected, "box", None) is not None:
            x1, _, x2, _ = selected.box
            center_x = (float(x1) + float(x2)) / 2.0
            frac = center_x / max(float(frame_width), 1.0)
            if frac < SPEECH_LEFT_ZONE_FRAC:
                return "left"
            if frac > SPEECH_RIGHT_ZONE_FRAC:
                return "right"
            return "ahead"
        zone = str(unknown_zone).upper()
        if zone == "LEFT":
            return "left"
        if zone == "RIGHT":
            return "right"
        return "ahead"

    def request(self, final_bucket, selected, frame_width, unknown_zone="", forced_direction=None):
        if not SPEECH_ENABLED or not self.running:
            return

        bucket = str(final_bucket).upper()

        # LOW clears the active warning. The next real hazard is a new event.
        if bucket == "LOW":
            with self.lock:
                self.last_bucket = "LOW"
                self.last_direction = None
                self.last_message = None
                self.direction_candidate = None
                self.direction_candidate_since = 0.0
            return

        if bucket not in ("CRITICAL", "HIGH", "MEDIUM"):
            return

        direction = (forced_direction or self._direction(selected, frame_width, unknown_zone))
        now = time.monotonic()

        # Ignore frame-to-frame left/right jitter. A direction must remain stable
        # briefly before it can create a new spoken event.
        if direction != self.direction_candidate:
            self.direction_candidate = direction
            self.direction_candidate_since = now
            if self.last_bucket != "LOW" and bucket == self.last_bucket:
                return
        elif now - self.direction_candidate_since < SPEECH_DIRECTION_STABLE_S:
            if bucket == self.last_bucket:
                return

        if bucket == "CRITICAL":
            message = f"Stop. Critical obstacle on your {direction}."
        elif bucket == "HIGH":
            message = f"Stop. Obstacle on your {direction}."
        else:
            message = f"Caution. Obstacle on your {direction}."

        state_changed = (
            bucket != self.last_bucket
            or direction != self.last_direction
        )

        # IMPORTANT: do NOT repeat a persistent warning. Speech happens only
        # when severity or a stable direction actually changes. LOW resets it.
        if not state_changed:
            return

        # Replace any pending warning with the newest feed.
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass

        self.last_bucket = bucket
        self.last_direction = direction
        self.last_message = message

        try:
            self.queue.put_nowait((message, bucket, direction))
            print(f"Speech: queued -> {message}")
        except queue.Full:
            pass

    def request_message(self, message, bucket, direction=None):
        """Queue an already-generated navigation sentence without re-running
        the deterministic speech-state gate. Used by the event-triggered LLM.
        """
        if not SPEECH_ENABLED or not self.running:
            return
        message = str(message).strip()
        if not message:
            return
        bucket = str(bucket).upper()
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait((message, bucket, direction))
            print(f"Speech: LLM queued -> {message}")
        except queue.Full:
            pass

    def deterministic_fallback_message(self, bucket, direction):
        bucket = str(bucket).upper()
        direction = direction or "ahead"
        if bucket == "CRITICAL":
            return f"Stop. Critical obstacle on your {direction}."
        if bucket == "HIGH":
            return f"Stop. Obstacle on your {direction}."
        if bucket == "MEDIUM":
            return f"Caution. Obstacle on your {direction}."
        return f"Caution. Obstacle on your {direction}."

    def stop(self):
        self.running = False
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


class GroqNavigationManager:
    """Event-triggered Groq navigation assistant.

    Only meaningful state changes create an LLM request. The latest request
    replaces stale pending context, and the deterministic safety path remains
    available as a fallback.
    """

    def __init__(self, speech_manager):
        self.speech = speech_manager

        # Prefer the current process environment. If the key was added to the
        # Windows User environment after this terminal was opened, also read
        # the User-level environment value directly.
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key and os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Environment",
                ) as key:
                    groq_api_key, _ = winreg.QueryValueEx(
                        key, "GROQ_API_KEY"
                    )
            except Exception:
                groq_api_key = None

        if groq_api_key:
            os.environ["GROQ_API_KEY"] = str(groq_api_key)

        self.enabled = bool(
            GROQ_ENABLED
            and Groq is not None
            and groq_api_key
        )
        self.client = None
        self.lock = threading.Lock()
        self.pending = None
        self.last_context = None
        self.last_event_time = 0.0
        self.last_spoken_message = None
        self.bucket_change_candidate = None
        self.bucket_change_start = None
        self.running = True

        # Build the Groq client before starting the worker. This removes a
        # startup race where the worker could see enabled=True with client=None.
        if self.enabled:
            try:
                self.client = Groq(timeout=GROQ_TIMEOUT_S)
                print(f"Groq: enabled | model={GROQ_MODEL}")
            except Exception as exc:
                self.enabled = False
                print(f"Groq: disabled (client init failed: {exc})")
        else:
            print("Groq: disabled (missing package or GROQ_API_KEY)")

        self.thread = threading.Thread(
            target=self._worker,
            name="GroqNavigationWorker",
            daemon=True,
        )
        self.thread.start()

    @staticmethod
    def _approach_state(closing_speed):
        if closing_speed > 0.08:
            return "approaching"
        if closing_speed < -0.08:
            return "moving away"
        return "stationary"

    def _make_context(self, bucket, obj_name, distance, direction, closing_speed, ttc):
        return {
            "bucket": str(bucket).upper(),
            "object": str(obj_name),
            "distance": float(distance),
            "direction": str(direction or "ahead"),
            "closing_speed": float(closing_speed),
            "ttc": float(ttc) if ttc is not None and np.isfinite(ttc) and ttc < 900 else None,
            "approach_state": self._approach_state(closing_speed),
        }

    def _meaningful_event(self, ctx, now):
        previous = self.last_context
        if previous is None:
            return True

        if now - self.last_event_time < GROQ_MIN_EVENT_INTERVAL_S:
            return False

        # A different physical obstacle or direction is a new event.
        if ctx["object"] != previous["object"]:
            return True
        if ctx["direction"] != previous["direction"]:
            return True

        # Do not retrigger merely because the GRU flickers between MEDIUM/HIGH
        # from frame to frame. Only a sustained increase in severity creates a
        # new Groq event. A decrease never retriggers speech.
        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        old_rank = rank.get(previous["bucket"], 0)
        new_rank = rank.get(ctx["bucket"], 0)

        if new_rank > old_rank:
            candidate = (ctx["object"], ctx["direction"], ctx["bucket"])
            if candidate != self.bucket_change_candidate:
                self.bucket_change_candidate = candidate
                self.bucket_change_start = now
                return False
            if (
                self.bucket_change_start is None
                or now - self.bucket_change_start < 1.0
            ):
                return False
            self.bucket_change_candidate = None
            self.bucket_change_start = None
            return True

        # Same severity: only a substantial distance change is meaningful.
        if ctx["bucket"] == previous["bucket"]:
            if abs(ctx["distance"] - previous["distance"]) >= GROQ_DISTANCE_EVENT_M:
                return True

        # Severity decrease / normal approach-state jitter: stay silent.
        self.bucket_change_candidate = None
        self.bucket_change_start = None
        return False

    def reset(self):
        # Called when there is no selected obstacle. The next obstacle is then
        # treated as a fresh event even if it has the same class/direction as
        # the previous one.
        with self.lock:
            self.last_context = None
            self.pending = None
            self.bucket_change_candidate = None
            self.bucket_change_start = None

    def request(self, bucket, obj_name, distance, direction, closing_speed, ttc):
        """Submit only meaningful navigation-state changes to Groq.

        CRITICAL is intentionally not sent to Groq: immediate deterministic
        safety speech must never wait for a network response.
        """
        bucket = str(bucket).upper()
        if bucket == "CRITICAL":
            return False
        ctx = self._make_context(
            bucket, obj_name, distance, direction, closing_speed, ttc
        )
        now = time.monotonic()
        with self.lock:
            if not self._meaningful_event(ctx, now):
                return False
            if self.pending is not None and ctx == self.pending:
                return False
            self.last_context = ctx
            self.last_event_time = now
            self.pending = ctx
        print(
            f"Groq: event queued | {ctx['bucket']} | {ctx['object']} | "
            f"{ctx['distance']:.2f}m | {ctx['direction']}"
        )
        return True

    def _prompt(self, ctx):
        ttc_text = (f"{ctx['ttc']:.1f} s" if ctx["ttc"] is not None else "unavailable")
        return f"""You are a wearable navigation assistant for a visually impaired user.
Give ONE short spoken instruction, maximum 18 words.
Use ONLY the supplied facts. Never invent an obstacle, distance, direction, movement, or safe escape route.
Never tell the user to turn left/right, step back, or choose an escape route unless that safe route is explicitly supplied.
Always include distance and direction when available. If approaching, say it is approaching.
CRITICAL: start with Stop and tell the user not to move forward.
HIGH: start with Stop and give a cautious instruction.
MEDIUM: start with Caution and tell the user to slow down.
LOW: brief awareness only.

Object: {ctx['object']}
Distance: {ctx['distance']:.2f} m
Direction: {ctx['direction']}
Closing speed: {ctx['closing_speed']:+.2f} m/s
TTC: {ttc_text}
Risk: {ctx['bucket']}
"""

    def _call_groq(self, ctx):
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": self._prompt(ctx)}],
            temperature=0,
            max_tokens=GROQ_MAX_TOKENS,
            reasoning_effort="none",
        )
        return response.choices[0].message.content.strip()

    def _worker(self):
        while self.running:
            with self.lock:
                ctx = self.pending
                self.pending = None
            if ctx is None:
                time.sleep(0.05)
                continue

            if not self.enabled or self.client is None:
                direction = ctx["direction"]
                self.speech.request_message(
                    self.speech.deterministic_fallback_message(ctx["bucket"], direction),
                    ctx["bucket"],
                    direction,
                )
                continue

            try:
                message = self._call_groq(ctx)
                if not message:
                    raise RuntimeError("empty Groq response")
                print(f"Groq: response -> {message}")
                self.speech.request_message(
                    message, ctx["bucket"], ctx["direction"]
                )
            except Exception as exc:
                print(f"Groq ERROR: {exc}")
                self.speech.request_message(
                    self.speech.deterministic_fallback_message(
                        ctx["bucket"], ctx["direction"]
                    ),
                    ctx["bucket"],
                    ctx["direction"],
                )

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


class UltrasonicReceiver:
    """Receives front/rear HC-SR04 distance packets from the ESP32 over UDP."""

    def __init__(self, host="0.0.0.0", port=4210):
        self.port = int(port)
        self.lock = threading.Lock()
        self.running = True

        # Each sensor has its own independent filtering/history state.
        self.channels = {
            "front": {
                "latest_distance_cm": None,
                "last_received_time": 0.0,
                "history": deque(maxlen=8),
                "raw_history": deque(maxlen=ULTRASONIC_MEDIAN_WINDOW),
                "filtered_distance_cm": None,
                "far_jump_candidate_cm": None,
                "far_jump_start_time": None,
            },
            "rear": {
                "latest_distance_cm": None,
                "last_received_time": 0.0,
                "history": deque(maxlen=8),
                "raw_history": deque(maxlen=ULTRASONIC_MEDIAN_WINDOW),
                "filtered_distance_cm": None,
                "far_jump_candidate_cm": None,
                "far_jump_start_time": None,
            },
        }

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, self.port))
        self.sock.settimeout(0.2)

        self.thread = threading.Thread(
            target=self._receive_loop,
            name="UltrasonicReceiver",
            daemon=True,
        )
        self.thread.start()

        print(f"Ultrasonic UDP receiver listening on port {self.port}")
        print("Ultrasonic channels: FRONT + REAR")

    def _update_channel(self, channel, distance_cm, now):
        state = self.channels[channel]
        if not (ULTRASONIC_MIN_CM <= distance_cm <= ULTRASONIC_MAX_CM):
            return

        state["raw_history"].append(distance_cm)
        median_cm = float(np.median(np.asarray(state["raw_history"], dtype=np.float32)))

        if state["filtered_distance_cm"] is None:
            state["filtered_distance_cm"] = median_cm
            state["far_jump_candidate_cm"] = None
            state["far_jump_start_time"] = None
        else:
            current_cm = float(state["filtered_distance_cm"])
            jump_cm = median_cm - current_cm

            if jump_cm > ULTRASONIC_FAR_JUMP_CM:
                if state["far_jump_candidate_cm"] is None:
                    state["far_jump_candidate_cm"] = median_cm
                    state["far_jump_start_time"] = now
                else:
                    state["far_jump_candidate_cm"] = (
                        0.5 * state["far_jump_candidate_cm"] + 0.5 * median_cm
                    )

                candidate_age = (
                    now - state["far_jump_start_time"]
                    if state["far_jump_start_time"] is not None
                    else 0.0
                )
                candidate_stability = abs(
                    median_cm - state["far_jump_candidate_cm"]
                )

                if (
                    candidate_age >= ULTRASONIC_FAR_JUMP_CONFIRM_S
                    and candidate_stability <= ULTRASONIC_FAR_JUMP_STABILITY_CM
                ):
                    state["filtered_distance_cm"] = (
                        ULTRASONIC_EMA_ALPHA * median_cm
                        + (1.0 - ULTRASONIC_EMA_ALPHA) * current_cm
                    )
                    state["far_jump_candidate_cm"] = None
                    state["far_jump_start_time"] = None
                else:
                    state["filtered_distance_cm"] = current_cm
            else:
                state["far_jump_candidate_cm"] = None
                state["far_jump_start_time"] = None
                state["filtered_distance_cm"] = (
                    ULTRASONIC_EMA_ALPHA * median_cm
                    + (1.0 - ULTRASONIC_EMA_ALPHA) * current_cm
                )

        filtered_cm = float(state["filtered_distance_cm"])
        state["latest_distance_cm"] = filtered_cm
        state["last_received_time"] = now
        state["history"].append((now, filtered_cm))

    def _receive_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                text = data.decode("utf-8").strip()
                parts = {}
                for item in text.split(","):
                    if ":" in item:
                        key, value = item.split(":", 1)
                        parts[key.strip()] = value.strip()

                now = time.monotonic()

                # Preferred dual-sensor packet:
                # front_cm:123.4,rear_cm:234.5,ts:123456
                # Also accept front: / rear: aliases.
                if "front_cm" in parts or "front" in parts:
                    front_value = parts.get("front_cm", parts.get("front"))
                    if front_value is not None:
                        self._update_channel("front", float(front_value), now)

                if "rear_cm" in parts or "rear" in parts:
                    rear_value = parts.get("rear_cm", parts.get("rear"))
                    if rear_value is not None:
                        self._update_channel("rear", float(rear_value), now)

                # Backward compatibility with the original one-sensor firmware.
                elif "distance_cm" in parts:
                    self._update_channel("front", float(parts["distance_cm"]), now)

            except (UnicodeDecodeError, ValueError, KeyError):
                continue

    def _get_channel_with_speed(self, channel, max_age_s):
        with self.lock:
            state = self.channels[channel]
            distance_cm = state["latest_distance_cm"]
            received_time = state["last_received_time"]
            samples = list(state["history"])

        if distance_cm is None or time.monotonic() - received_time > max_age_s:
            return None, 0.0

        if len(samples) >= 2:
            t0, d0 = samples[0]
            t1, d1 = samples[-1]
            dt = max(t1 - t0, 1e-3)
            closing_speed = (d0 - d1) / 100.0 / dt
            if abs(closing_speed) < SPEED_DEADBAND:
                closing_speed = 0.0
        else:
            closing_speed = 0.0

        return float(distance_cm), float(closing_speed)

    def get_latest_with_speed(self, max_age_s=0.5):
        """Backward-compatible front-sensor accessor."""
        return self._get_channel_with_speed("front", max_age_s)

    def get_latest_pair_with_speed(self, max_age_s=0.5):
        """Return (front_cm, front_speed, rear_cm, rear_speed)."""
        front = self._get_channel_with_speed("front", max_age_s)
        rear = self._get_channel_with_speed("rear", max_age_s)
        return front[0], front[1], rear[0], rear[1]

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

class LatestFrameReader:
    """Reads frames from a VideoCapture in a background thread, always
    keeping only the MOST RECENT frame. This prevents the delay-creep you
    get from cv2's internal buffer queuing up frames faster than the main
    loop (YOLO inference) can process them -- without this, every frame
    processed is older than the last, and the lag grows over time."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source {source}")
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        """Returns (ok, latest_frame) -- non-blocking, always the freshest
        frame available, never a stale queued one."""
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# ---- config ----
SOURCE = "http://192.168.29.115:8080/video"   # phone IP-cam stream; 0 = default webcam
MODEL = "yolov8n.pt"  # kept for reference/documentation
YOLO_OPENVINO_MODEL = "yolov8n_openvino_model/yolov8n.xml"
YOLO_DEVICE = "GPU"
YOLO_INPUT_SIZE = 256

# Run YOLO every 2nd frame to reduce inference load.
# Tracking keeps the latest detected objects between YOLO passes.
YOLO_INFERENCE_EVERY_N_FRAMES = 2

# Match the confidence/NMS style of a normal YOLO detection pipeline.
YOLO_CONF_THRESHOLD = 0.25
YOLO_NMS_IOU = 0.45
MAX_RANGE = 4.0
# Keep YOLO's native semantic classes. Every YOLO class is allowed through
# the navigation pipeline; only the GRU input is collapsed later to its legacy
# six-class representation. This means chair stays "chair", laptop stays
# "laptop", bottle stays "bottle", etc.
CLASS_MAP = {}

# The existing GRU was trained with six categorical values. Keep that mapping
# isolated to the GRU boundary so it does not alter YOLO/tracking labels.
GRU_CLASS_MAP = {
    "person": "person",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "traffic light": "pole", "fire hydrant": "pole", "stop sign": "pole",
    "chair": "pole", "bench": "pole", "potted plant": "pole",
    "couch": "wall",
    "cell phone": "pole",
    "dining table": "wall",
    "bottle": "pole",
    "tv": "wall",
}
# Note: classes outside COCO still cannot be detected by this pretrained model,
# but every COCO class it does detect is now retained with its native label.

INFERENCE_IMGSZ = 256             # resolution YOLO actually runs inference at --
                                   # lower = faster, less accurate on small objects.
                                   # OpenVINO decoding rescales detected boxes back to the
                                   # ORIGINAL frame size automatically, so area_frac
                                   # (and CALIBRATION_K) stay valid at native res --
                                   # this only speeds up inference, doesn't touch
                                   # the resolution used for distance calibration.
SHOW_ALL_DETECTIONS = False        # set True to also draw every raw YOLO detection
                                   # (any class, dim gray) for debugging -- off by
                                   # default so the view only shows the nearest
                                   # MAX_TRACKS tracked objects
# Keep YOLO's semantic class names intact throughout detection/tracking/display.
# The legacy 6-class representation is used ONLY when constructing the GRU's
# existing 4-feature input, so the trained GRU schema remains compatible.
FEATURE_CLASSES = ["none", "person", "pole", "wall", "vehicle", "curb"]
AGENT_SPEED_ASSUMED = 1.2

# MiDaS has no semantic class label. For the existing trained 4-feature GRU,
# use the already-trained "curb" feature as the generic unknown-obstacle
# proxy. This keeps the GRU input shape/schema unchanged (30 x 4).
UNKNOWN_GRU_CLASS = "curb"

# ---- live GRU config ----
GRU_MODEL_FILE = "risk_gru_model_final.keras"
GRU_SEQ_LEN = 30
GRU_WARMUP_RISK = 0.0

# Keep these identical to the offline evaluation.
LOW_MED_BOUNDARY = 0.33
MED_HIGH_BOUNDARY = 0.66

# Live inference does not need to run the neural network on every camera
# frame. Running every N frames reduces CPU load while retaining a smooth
# risk display.
GRU_INFERENCE_EVERY_N_FRAMES = 3

# ---- MiDaS Small + OpenVINO ----
# MiDaS does NOT change the GRU inputs. Confirmed unknown protrusions are
# integrated afterward as a deterministic safety override.
MIDAS_MODEL_FILE = "MiDaS/weights/openvino/openvino_midas_v21_small_256.xml"
MIDAS_DEVICE = "GPU"
MIDAS_INPUT_SIZE = 256
MIDAS_INFERENCE_EVERY_N_FRAMES = 2       # Async; main loop never waits for it.

# ---- immediate proximity safety layer ----
# These are deliberately NOT fed back into the GRU. The GRU remains the
# trained 4-feature model; this deterministic layer handles objects that
# are already extremely close even when closing_speed is near zero.
PROXIMITY_CRITICAL_M = 0.35
PROXIMITY_HIGH_M = 0.45

# ---- unknown-obstacle risk integration ----
# MiDaS unknown obstacles are NOT fed into the GRU because the GRU was
# trained on the original 4-feature schema. Instead, a confirmed MiDaS
# protrusion acts as a deterministic safety layer on top of GRU/proximity.
UNKNOWN_OBSTACLE_MIN_RISK = LOW_MED_BOUNDARY + 0.02
UNKNOWN_OBSTACLE_HIGH_SCORE = 0.72
UNKNOWN_OBSTACLE_HIGH_DEPTH = 0.68
UNKNOWN_OBSTACLE_HIGH_BOTTOM_FRAC = 0.88

# Bottom-of-frame warning: an object extending into this fraction of the
# image height is likely very close to the camera/user.
BOTTOM_ZONE_START_FRAC = 0.78
BOTTOM_ZONE_CRITICAL_FRAC = 0.95

# ---- MiDaS object-region growth / temporal stability ----
# The protrusion mask often contains only the strongest depth edges of a
# real object. Grow each confirmed seed into the surrounding depth plateau
# so the displayed bbox covers the object rather than a tiny edge fragment.
# V14: MiDaS bbox geometry comes from the protrusion mask itself.
# Do NOT grow a seed through a depth plateau: that was causing large
# background/laptop/floor regions to become giant false boxes.
DEPTH_GROW_RADIUS_FRAC = 0.0
DEPTH_GROW_TOLERANCE = 0.0
DEPTH_GROW_MIN_COMPONENT_FRAC = 0.0
DEPTH_GROW_MAX_COMPONENT_FRAC = 0.08
BOX_SMOOTH_ALPHA = 0.22

# Retained for Track geometry bookkeeping. The V18 fusion pipeline no longer
# uses the old first-bbox reference-distance mechanism, but Track.update()
# still uses these guards for safe area calculations.
TRACK_REFERENCE_MIN_AREA_FRAC = 1e-4
TRACK_REFERENCE_MAX_AREA_FRAC = 0.50
TRACK_AREA_EMA_ALPHA = 0.20
# Fragment merging is intentionally conservative. Only edge fragments that
# are close in image space and overlap strongly in one axis may be combined.
MIDAS_MERGE_GAP_FRAC_V14 = 0.018
MIDAS_MERGE_MIN_OVERLAP_V14 = 0.30
MIDAS_MERGE_CENTER_FRAC_V14 = 0.07



EMA_ALPHA = 0.12
CLOSING_SPEED_EMA_ALPHA = 0.14
RISK_EMA_ALPHA = 0.16
SPEED_WINDOW = 8                 # frames of history kept per tracked object

# ---- distance fusion ----
# Keep the original calibrated YOLO bbox-area model as the metric anchor.
# MiDaS is used as a RELATIVE correction signal, not as a second absolute
# metre measurement. This avoids treating raw MiDaS values as metres.
#
# Frame-level fusion:
#     d_yolo = existing calibrated YOLO distance
#     r      = median(MiDaS depth inside object) /
#              median(MiDaS depth around object)
#     d_fused = d_yolo * correction(r)
#
# The fused value is then aggregated over a short temporal window before
# closing speed/TTC are calculated.
# ---- HC-SR04 metric-distance fusion ----
# The ultrasonic sensor is used as the authoritative metric distance only
# for a YOLO obstacle whose image position is close to the sensor's forward
# axis. YOLO still supplies semantic identity and bounding-box position.
ULTRASONIC_FUSION_ENABLED = True
ULTRASONIC_MAX_AGE_S = 0.5
ULTRASONIC_CENTER_TOL_FRAC = 0.15
# Front HC-SR04 is matched to the closest YOLO/MiDaS bbox by metric distance.
# If no visual bbox is close enough, the ultrasonic obstacle is classified FRONT.
ULTRASONIC_BBOX_MATCH_TOL_M = 0.50
ULTRASONIC_MIN_CM = 2.0
ULTRASONIC_MAX_CM = 400.0
ULTRASONIC_MEDIAN_WINDOW = 5
ULTRASONIC_EMA_ALPHA = 0.25

# Reject sudden jumps to a much FARTHER reading (often the background wall).
# A closer reading is accepted immediately for safety. A farther reading is
# accepted only after it remains stable for this many seconds.
ULTRASONIC_FAR_JUMP_CM = 35.0
ULTRASONIC_FAR_JUMP_CONFIRM_S = 1.0
ULTRASONIC_FAR_JUMP_STABILITY_CM = 12.0

# ---- long-range vision-assisted ultrasonic failsafe ----
ULTRASONIC_VISION_OVERRIDE_START_M = 3.20
ULTRASONIC_VISION_OVERRIDE_MARGIN_M = 0.20
ULTRASONIC_VISION_CONFIRM_S = 1.00
ULTRASONIC_VISION_MIN_UPDATES = 3
ULTRASONIC_VISION_CENTER_TOL_FRAC = 0.20

# Conservative vision confidence gate for the long-range ultrasonic failsafe.
# The absolute YOLO calibration was obtained primarily from person-distance
# data, so other classes are allowed but receive a small reliability penalty.
VISION_CONF_HIGH = 0.68
VISION_CONF_MEDIUM = 0.45
VISION_MIN_YOLO_CONF = 0.30
VISION_HIGH_YOLO_CONF = 0.50
VISION_MIN_MIDAS_STRENGTH = 0.12
VISION_HIGH_MIDAS_STRENGTH = 0.35
VISION_STABILITY_CV_HIGH = 0.10
VISION_STABILITY_CV_MEDIUM = 0.20
VISION_NON_PERSON_PENALTY = 0.88

DIST_FUSION_ENABLED = True
MIDAS_CORRECTION_GAMMA = 0.65
MIDAS_CORRECTION_MIN = 0.70
MIDAS_CORRECTION_MAX = 1.30
MIDAS_RATIO_MIN = 0.70
MIDAS_RATIO_MAX = 1.60
DIST_FUSION_MEDIAN_WINDOW = 5
TRACK_DISTANCE_EMA_ALPHA = 0.22

# The old per-track reference-distance mechanism is disabled because a wrong
# first bbox can permanently anchor a track to the wrong absolute distance.
TRACK_REFERENCE_LOCK = False

SPEED_DEADBAND = 0.05
DIST_C = 0.4327                  # recalibrated from 12 fresh person-distance points (0.25m..3.00m)
DIST_EXPONENT = 0.7530

# Retained for MiDaS-only fallback objects. These values are NOT used for
# YOLO+MiDaS fused objects.
MIDAS_CALIBRATION_MIN_SAMPLES = 6
MIDAS_CALIBRATION_MAX_SAMPLES = 120
MIDAS_CALIBRATION_MIN_RAW_SPREAD = 1e-4
MIDAS_DISTANCE_MIN = 0.20
MIDAS_DISTANCE_MAX = MAX_RANGE
MIDAS_DISTANCE_FALLBACK = 4.0
MIDAS_DISTANCE_EMA_ALPHA = 0.18

ROTATE = False                   # keep native landscape orientation
PROCESS_WIDTH = None             # keep native resolution -- set to an int to force resize
DISPLAY_MAX_WIDTH = 960          # display window is capped to this width so it fits on
                                  # screen -- purely visual, does NOT affect detection/
                                  # calibration, which still run on the native frame

# ---- multi-object tracking config ----
MAX_TRACKS = 3                   # keep at most this many simultaneous tracks
                                  # (top-3 covers "several obstacles at once"
                                  # without unbounded cost per frame)
MATCH_MAX_DIST_FRAC = 0.25       # max centroid movement (as a fraction of frame
                                  # width) between frames to count as "the same
                                  # object" -- tune up if fast objects lose their
                                  # track ID, down if separate objects get merged
TRACK_TIMEOUT_S = 1.5            # keep a track through short YOLO dropouts
                                  # (object left frame / occluded)
TTC_SAFE_VALUE = 999.0
UNKNOWN_REPLACEMENT_MARGIN_M = 0.15           # TTC assigned when an object isn't closing

# V11: geometry/fusion guards.  MiDaS mask fragments are treated as one
# physical obstacle when they are spatially close and have similar depth.
MIDAS_MERGE_GAP_FRAC = 0.035
MIDAS_MERGE_MIN_OVERLAP = 0.20
MIDAS_MERGE_CENTER_FRAC = 0.11
YOLO_MIDAS_CENTER_FRAC = 0.10
YOLO_MIDAS_MIN_CONTAINMENT = 0.18

                                  # in (moving away or stationary) -- effectively
                                  # "infinite time," so it never wins the
                                  # lowest-TTC selection over a real threat


def bbox_area_to_distance(box_area_frac):
    box_area_frac = max(box_area_frac, 1e-4)
    est = DIST_C * (box_area_frac ** (-DIST_EXPONENT))
    return float(np.clip(est, 0.2, MAX_RANGE))


class MidasMetricCalibrator:
    """Learn raw MiDaS inverse-depth -> metric distance from YOLO matches."""
    def __init__(self):
        self.samples = deque(maxlen=MIDAS_CALIBRATION_MAX_SAMPLES)
        self.a = None
        self.b = None
        self.last_prediction = None

    def add(self, raw_depth, metric_distance):
        raw_depth = float(raw_depth)
        metric_distance = float(metric_distance)
        if not np.isfinite(raw_depth) or not np.isfinite(metric_distance):
            return
        if metric_distance < MIDAS_DISTANCE_MIN or metric_distance > MIDAS_DISTANCE_MAX:
            return
        self.samples.append((raw_depth, metric_distance))
        self._fit()

    def _fit(self):
        if len(self.samples) < MIDAS_CALIBRATION_MIN_SAMPLES:
            return
        x = np.asarray([p[0] for p in self.samples], dtype=np.float64)
        y = np.asarray([1.0 / p[1] for p in self.samples], dtype=np.float64)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return
        if float(np.ptp(x)) < MIDAS_CALIBRATION_MIN_RAW_SPREAD:
            return
        try:
            a, b = np.polyfit(x, y, 1)
            pred = a * x + b
            resid = np.abs(pred - y)
            med = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med)))
            keep = resid <= max(3.0 * mad, 0.015)
            if int(np.count_nonzero(keep)) >= MIDAS_CALIBRATION_MIN_SAMPLES:
                a, b = np.polyfit(x[keep], y[keep], 1)
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                self.a = float(a)
                self.b = float(b)
        except Exception:
            pass

    def predict(self, raw_depth):
        raw_depth = float(raw_depth)
        if self.a is None or self.b is None or not np.isfinite(raw_depth):
            return MIDAS_DISTANCE_FALLBACK, False
        inv_d = self.a * raw_depth + self.b
        if not np.isfinite(inv_d) or inv_d <= 1e-6:
            return MIDAS_DISTANCE_FALLBACK, False
        dist = float(np.clip(1.0 / inv_d, MIDAS_DISTANCE_MIN, MIDAS_DISTANCE_MAX))
        if self.last_prediction is None:
            self.last_prediction = dist
        else:
            self.last_prediction = (
                MIDAS_DISTANCE_EMA_ALPHA * dist
                + (1.0 - MIDAS_DISTANCE_EMA_ALPHA) * self.last_prediction
            )
        return float(self.last_prediction), True

    @property
    def ready(self):
        return self.a is not None and self.b is not None


def resize_fixed(frame, width):
    if width is None:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


def _depth_core_and_surrounding(raw_depth, box, frame_shape):
    """Compute object/surrounding depth directly on the MiDaS grid.

    This avoids resizing the raw 256x256 depth map to the full camera frame for
    every tracked object.
    """
    if (
        raw_depth is None
        or not isinstance(raw_depth, np.ndarray)
        or raw_depth.size == 0
        or raw_depth.ndim != 2
    ):
        return float("nan"), float("nan")

    fh, fw = frame_shape[:2]
    dh, dw = raw_depth.shape[:2]
    sx = dw / max(float(fw), 1.0)
    sy = dh / max(float(fh), 1.0)

    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(fw - 2, x1))
    y1 = max(0, min(fh - 2, y1))
    x2 = max(x1 + 2, min(fw, x2))
    y2 = max(y1 + 2, min(fh, y2))

    dx1 = max(0, min(dw - 2, int(round(x1 * sx))))
    dy1 = max(0, min(dh - 2, int(round(y1 * sy))))
    dx2 = max(dx1 + 2, min(dw, int(round(x2 * sx))))
    dy2 = max(dy1 + 2, min(dh, int(round(y2 * sy))))

    bw = dx2 - dx1
    bh = dy2 - dy1

    ix1 = dx1 + int(0.20 * bw)
    ix2 = dx2 - int(0.20 * bw)
    iy1 = dy1 + int(0.20 * bh)
    iy2 = dy2 - int(0.20 * bh)
    if ix2 <= ix1 or iy2 <= iy1:
        ix1, iy1, ix2, iy2 = dx1, dy1, dx2, dy2

    depth = raw_depth.astype(np.float32, copy=False)
    core = depth[iy1:iy2, ix1:ix2]
    core_vals = core[np.isfinite(core)]

    pad_x = max(3, int(0.55 * bw))
    pad_y = max(3, int(0.55 * bh))
    ox1 = max(0, dx1 - pad_x)
    oy1 = max(0, dy1 - pad_y)
    ox2 = min(dw, dx2 + pad_x)
    oy2 = min(dh, dy2 + pad_y)

    outer = depth[oy1:oy2, ox1:ox2]
    inner_x1 = dx1 - ox1
    inner_y1 = dy1 - oy1
    inner_x2 = dx2 - ox1
    inner_y2 = dy2 - oy1

    ring_mask = np.ones(outer.shape, dtype=bool)
    ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    surround_vals = outer[ring_mask & np.isfinite(outer)]

    if core_vals.size < 8 or surround_vals.size < 20:
        return float("nan"), float("nan")

    return float(np.median(core_vals)), float(np.median(surround_vals))

def midas_relative_correction(raw_depth, box, frame_shape):
    """Convert MiDaS relative depth contrast into a bounded distance correction.

    A larger MiDaS depth in the object core than in the surrounding region
    means the object is locally closer. The correction is intentionally bounded
    so MiDaS cannot catastrophically override the calibrated YOLO anchor.
    """
    obj_depth, bg_depth = _depth_core_and_surrounding(
        raw_depth,
        box,
        frame_shape,
    )

    if (
        not np.isfinite(obj_depth)
        or not np.isfinite(bg_depth)
        or bg_depth <= 1e-6
    ):
        return 1.0, obj_depth, bg_depth, 0.0

    ratio = float(
        np.clip(
            obj_depth / bg_depth,
            MIDAS_RATIO_MIN,
            MIDAS_RATIO_MAX,
        )
    )

    # Stronger local protrusion => stronger correction, but still bounded.
    correction = float(
        np.clip(
            ratio ** (-MIDAS_CORRECTION_GAMMA),
            MIDAS_CORRECTION_MIN,
            MIDAS_CORRECTION_MAX,
        )
    )

    strength = float(
        np.clip(
            abs(np.log(max(ratio, 1e-6)))
            / abs(np.log(1.45)),
            0.0,
            1.0,
        )
    )

    return correction, obj_depth, bg_depth, strength


class Track:
    """One tracked object's identity + rolling history across frames."""
    _next_id = 1

    def __init__(self, cls_name, centroid, dist, now, box=None, frame_area=None):
        self.id = Track._next_id
        Track._next_id += 1
        self.cls_name = cls_name
        self.source = "YOLO"
        self.centroid = centroid
        self.box = box

        # Reference established from the first stable observations. The
        # calibrated initial distance remains the metric anchor; later
        # distance changes come from tracked apparent-size change.
        self.reference_area_frac = None
        self.reference_distance = None
        self.area_ema_frac = None
        self.reference_observations = []
        self.reference_locked = False

        if box is not None and frame_area:
            x1, y1, x2, y2 = box
            area_frac = max(
                ((x2 - x1) * (y2 - y1)) / max(frame_area, 1),
                TRACK_REFERENCE_MIN_AREA_FRAC,
            )
            if TRACK_REFERENCE_MIN_AREA_FRAC <= area_frac <= TRACK_REFERENCE_MAX_AREA_FRAC:
                self.reference_observations.append((area_frac, dist))
                self.area_ema_frac = area_frac

        self.smoothed_dist = float(dist)
        self.distance_history = deque(
            [float(dist)],
            maxlen=DIST_FUSION_MEDIAN_WINDOW,
        )
        self.history = [(now, float(dist))]
        self.filtered_closing_speed = 0.0
        self.last_seen = now
        self.last_yolo_distance = float(dist)
        self.last_yolo_confidence = 0.0
        self.last_midas_ratio = 1.0
        self.last_midas_correction = 1.0
        self.last_midas_strength = 0.0

        # Independent vision-only metric history. The main `history` can be
        # replaced by HC-SR04, so it must never be reused for the long-range
        # vision-vs-ultrasonic comparison.
        self.vision_distance_history = deque(
            [float(dist)],
            maxlen=DIST_FUSION_MEDIAN_WINDOW,
        )
        self.vision_smoothed_dist = float(dist)
        self.vision_history = [(now, float(dist))]
        self.vision_filtered_closing_speed = 0.0

    def _update_reference(self, area_frac, calibrated_dist):
        if self.reference_locked:
            return

        self.reference_observations.append(
            (area_frac, calibrated_dist)
        )

        if len(self.reference_observations) < TRACK_REFERENCE_FRAMES:
            return

        areas = np.asarray(
            [a for a, _ in self.reference_observations],
            dtype=np.float64,
        )
        dists = np.asarray(
            [d for _, d in self.reference_observations],
            dtype=np.float64,
        )

        self.reference_area_frac = float(np.median(areas))
        self.reference_distance = float(np.median(dists))
        self.reference_locked = True

    def _tracked_distance(self, area_frac, fallback_dist):
        if self.reference_locked and self.reference_area_frac is not None:
            # Same empirical exponent as the existing calibration, but applied
            # to the CHANGE in apparent area relative to this object's traced
            # reference box. This is the key difference from recalibrating a
            # fresh absolute distance from every noisy detection box.
            ratio = self.reference_area_frac / max(
                area_frac,
                TRACK_REFERENCE_MIN_AREA_FRAC,
            )
            estimated = self.reference_distance * (ratio ** DIST_EXPONENT)
            return float(
                np.clip(
                    estimated,
                    0.2,
                    MAX_RANGE,
                )
            )
        return float(fallback_dist)

    def update(self, cls_name, centroid, box, raw_dist, now, frame_area):
        self.cls_name = cls_name
        self.centroid = centroid

        if self.box is None:
            self.box = box
        else:
            px1, py1, px2, py2 = map(float, self.box)
            nx1, ny1, nx2, ny2 = map(float, box)
            pw, ph = max(px2-px1, 1.0), max(py2-py1, 1.0)
            nw, nh = max(nx2-nx1, 1.0), max(ny2-ny1, 1.0)
            nw = float(np.clip(nw, pw*0.78, pw*1.28))
            nh = float(np.clip(nh, ph*0.78, ph*1.28))
            ncx = 0.30*((nx1+nx2)/2.0) + 0.70*((px1+px2)/2.0)
            ncy = 0.30*((ny1+ny2)/2.0) + 0.70*((py1+py2)/2.0)
            self.box = (
                int(ncx - nw/2.0), int(ncy - nh/2.0),
                int(ncx + nw/2.0), int(ncy + nh/2.0),
            )

        x1, y1, x2, y2 = self.box
        area_frac = max(
            ((x2 - x1) * (y2 - y1)) / max(frame_area, 1),
            TRACK_REFERENCE_MIN_AREA_FRAC,
        )
        area_frac = float(
            np.clip(
                area_frac,
                TRACK_REFERENCE_MIN_AREA_FRAC,
                TRACK_REFERENCE_MAX_AREA_FRAC,
            )
        )

        if self.area_ema_frac is None:
            self.area_ema_frac = area_frac
        else:
            self.area_ema_frac = (
                TRACK_AREA_EMA_ALPHA * area_frac
                + (1.0 - TRACK_AREA_EMA_ALPHA) * self.area_ema_frac
            )

        # raw_dist is now the already-fused metric estimate. The old
        # first-bbox reference mechanism is deliberately disabled because it
        # could permanently anchor a track to an incorrect initial distance.
        fused_dist = float(raw_dist)
        self.last_yolo_distance = fused_dist

        # Maintain a vision-only distance stream independent of HC-SR04.
        self.vision_distance_history.append(fused_dist)
        vision_median_dist = float(
            np.median(
                np.asarray(
                    self.vision_distance_history,
                    dtype=np.float64,
                )
            )
        )
        self.vision_smoothed_dist = (
            TRACK_DISTANCE_EMA_ALPHA * vision_median_dist
            + (1.0 - TRACK_DISTANCE_EMA_ALPHA) * self.vision_smoothed_dist
        )
        self.vision_history.append((now, self.vision_smoothed_dist))
        if len(self.vision_history) > SPEED_WINDOW:
            self.vision_history.pop(0)

        self.distance_history.append(fused_dist)

        # Robust temporal aggregation first, then a light EMA.
        median_dist = float(
            np.median(
                np.asarray(
                    self.distance_history,
                    dtype=np.float64,
                )
            )
        )

        self.smoothed_dist = (
            TRACK_DISTANCE_EMA_ALPHA * median_dist
            + (1.0 - TRACK_DISTANCE_EMA_ALPHA) * self.smoothed_dist
        )
        self.history.append((now, self.smoothed_dist))
        if len(self.history) > SPEED_WINDOW:
            self.history.pop(0)
        self.last_seen = now

    def closing_speed(self):
        if len(self.history) < 2:
            return 0.0
        t0, d0 = self.history[0]
        t1, d1 = self.history[-1]
        dt = max(t1 - t0, 1e-3)
        speed = (d0 - d1) / dt   # positive = getting closer
        if abs(speed) < SPEED_DEADBAND:
            speed = 0.0
        # Low-pass the derivative. Distance is already EMA-smoothed, but a
        # derivative amplifies small frame-to-frame box jitter.
        self.filtered_closing_speed = (
            CLOSING_SPEED_EMA_ALPHA * speed
            + (1.0 - CLOSING_SPEED_EMA_ALPHA) * self.filtered_closing_speed
        )
        if abs(self.filtered_closing_speed) < SPEED_DEADBAND:
            return 0.0
        return self.filtered_closing_speed

    def vision_closing_speed(self):
        """Closing speed computed only from the independent vision history."""
        if len(self.vision_history) < 2:
            return 0.0
        t0, d0 = self.vision_history[0]
        t1, d1 = self.vision_history[-1]
        dt = max(t1 - t0, 1e-3)
        speed = (d0 - d1) / dt
        if abs(speed) < SPEED_DEADBAND:
            speed = 0.0
        self.vision_filtered_closing_speed = (
            CLOSING_SPEED_EMA_ALPHA * speed
            + (1.0 - CLOSING_SPEED_EMA_ALPHA) * self.vision_filtered_closing_speed
        )
        if abs(self.vision_filtered_closing_speed) < SPEED_DEADBAND:
            return 0.0
        return self.vision_filtered_closing_speed

    def vision_ttc(self):
        """TTC computed from the independent vision metric stream."""
        speed = self.vision_closing_speed()
        if speed <= 0:
            return TTC_SAFE_VALUE
        return self.vision_smoothed_dist / speed

    def ttc(self):
        """Time-to-collision estimate. Lower = more urgent.
        Objects not closing in get a large 'safe' value so they never
        outrank a genuine approaching threat."""
        speed = self.closing_speed()
        if speed <= 0:
            return TTC_SAFE_VALUE
        return self.smoothed_dist / speed


def match_detections_to_tracks(detections, tracks, frame_width, frame_height, now):
    """Greedy centroid+class matching: each detection claims the closest
    unclaimed track of the same class within MATCH_MAX_DIST_FRAC, else
    spawns a new track."""
    max_dist_px = MATCH_MAX_DIST_FRAC * frame_width
    unmatched_tracks = list(tracks)
    updated = []

    for cls_name, centroid, box, raw_dist in detections:
        best_track, best_dist = None, None
        for tr in unmatched_tracks:
            if tr.cls_name != cls_name:
                continue
            d = np.hypot(centroid[0] - tr.centroid[0], centroid[1] - tr.centroid[1])
            if d <= max_dist_px and (best_dist is None or d < best_dist):
                best_track, best_dist = tr, d

        if best_track is not None:
            best_track.update(cls_name, centroid, box, raw_dist, now, frame_width * frame_height)
            unmatched_tracks.remove(best_track)
            updated.append(best_track)
        else:
            new_track = Track(cls_name, centroid, raw_dist, now, box=box, frame_area=frame_width * frame_height)
            new_track.box = box
            updated.append(new_track)

    # keep still-alive-but-unmatched tracks too (object briefly occluded)
    for tr in unmatched_tracks:
        if now - tr.last_seen <= TRACK_TIMEOUT_S:
            updated.append(tr)

    # cap total tracks -- keep the ones with lowest current distance
    updated.sort(key=lambda t: t.smoothed_dist)
    return updated[:MAX_TRACKS]


def assign_yolo_confidence_to_tracks(tracks, yolo_detections, frame_width):
    """Attach raw YOLO confidence to the already-matched Track objects."""
    if not yolo_detections:
        return
    max_dist_px = MATCH_MAX_DIST_FRAC * frame_width
    for tr in tracks:
        if tr.box is None:
            continue
        best_conf = None
        best_dist = None
        for cls_name_raw, centroid, _box, confidence in yolo_detections:
            mapped = CLASS_MAP.get(cls_name_raw, cls_name_raw)
            if mapped != tr.cls_name:
                continue
            d = np.hypot(
                centroid[0] - tr.centroid[0],
                centroid[1] - tr.centroid[1],
            )
            if d <= max_dist_px and (best_dist is None or d < best_dist):
                best_dist = d
                best_conf = float(confidence)
        if best_conf is not None:
            tr.last_yolo_confidence = best_conf


def vision_confidence(track):
    """Return (score, level, details) for the long-range vision failsafe."""
    if track is None or getattr(track, "box", None) is None:
        return 0.0, "LOW", "no visual bbox"

    yolo_conf = float(np.clip(getattr(track, "last_yolo_confidence", 0.0), 0.0, 1.0))
    midas_strength = float(np.clip(getattr(track, "last_midas_strength", 0.0), 0.0, 1.0))
    hist = np.asarray(list(getattr(track, "vision_distance_history", [])), dtype=np.float64)

    if hist.size >= 3:
        mean_d = max(float(np.mean(hist)), 1e-6)
        cv = float(np.std(hist) / mean_d)
    else:
        cv = 1.0

    if yolo_conf < VISION_MIN_YOLO_CONF:
        yolo_score = 0.0
    elif yolo_conf >= VISION_HIGH_YOLO_CONF:
        yolo_score = 1.0
    else:
        yolo_score = (yolo_conf - VISION_MIN_YOLO_CONF) / max(
            VISION_HIGH_YOLO_CONF - VISION_MIN_YOLO_CONF, 1e-6
        )

    if cv <= VISION_STABILITY_CV_HIGH:
        stability_score = 1.0
    elif cv <= VISION_STABILITY_CV_MEDIUM:
        stability_score = (VISION_STABILITY_CV_MEDIUM - cv) / max(
            VISION_STABILITY_CV_MEDIUM - VISION_STABILITY_CV_HIGH, 1e-6
        )
    else:
        stability_score = 0.0

    if midas_strength <= VISION_MIN_MIDAS_STRENGTH:
        midas_score = 0.0
    elif midas_strength >= VISION_HIGH_MIDAS_STRENGTH:
        midas_score = 1.0
    else:
        midas_score = (midas_strength - VISION_MIN_MIDAS_STRENGTH) / max(
            VISION_HIGH_MIDAS_STRENGTH - VISION_MIN_MIDAS_STRENGTH, 1e-6
        )

    score = 0.50 * yolo_score + 0.25 * stability_score + 0.25 * midas_score
    if getattr(track, "cls_name", "") != "person":
        score *= VISION_NON_PERSON_PENALTY
    score = float(np.clip(score, 0.0, 1.0))

    if score >= VISION_CONF_HIGH:
        level = "HIGH"
    elif score >= VISION_CONF_MEDIUM:
        level = "MEDIUM"
    else:
        level = "LOW"

    details = f"conf={yolo_conf:.2f} cv={cv:.2f} MiDaS={midas_strength:.2f} score={score:.2f}"
    return score, level, details


def current_yolo_anchor_distance(track, frame_area):
    """Recover the original calibrated YOLO bbox-area estimate for diagnostics."""
    if track is None or track.box is None:
        return float(MAX_RANGE)
    x1, y1, x2, y2 = map(int, track.box)
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    area_frac = (
        (bw * bh)
        / max(float(frame_area), 1.0)
    )
    return bbox_area_to_distance(area_frac)


def risk_bucket(risk):
    """Convert continuous GRU risk into the same three buckets used offline."""
    if risk < LOW_MED_BOUNDARY:
        return "LOW"
    if risk < MED_HIGH_BOUNDARY:
        return "MEDIUM"
    return "HIGH"



def proximity_override(tr, frame_height, distance_override=None):
    """
    Deterministic immediate-proximity check.

    Returns:
        (level, reason)

    This is intentionally separate from GRU prediction:
      - VERY close distance can force HIGH
      - A box reaching the bottom of the image can raise proximity risk
    """
    if tr is None or tr.box is None:
        return "NONE", "no selected obstacle"

    x1, y1, x2, y2 = tr.box
    bottom_frac = float(y2) / max(frame_height, 1)
    dist = (
        float(distance_override)
        if distance_override is not None
        else float(tr.smoothed_dist)
    )

    if dist <= PROXIMITY_CRITICAL_M:
        return "CRITICAL", f"distance {dist:.2f}m"

    if dist <= PROXIMITY_HIGH_M:
        return "HIGH", f"distance {dist:.2f}m"

    if bottom_frac >= BOTTOM_ZONE_CRITICAL_FRAC:
        return "HIGH", f"box bottom {bottom_frac:.0%}"

    if bottom_frac >= BOTTOM_ZONE_START_FRAC:
        return "MEDIUM", f"box bottom {bottom_frac:.0%}"

    return "NONE", "outside immediate zone"


def combine_risk(gru_risk, proximity_level):
    """Combine predictive GRU risk with the deterministic proximity override."""
    if proximity_level == "CRITICAL":
        return 1.0, "CRITICAL"

    if proximity_level == "HIGH":
        return max(float(gru_risk), MED_HIGH_BOUNDARY), "HIGH"

    if proximity_level == "MEDIUM":
        # Do not force a full high risk, but don't let the GRU call an
        # immediately foregrounded object LOW.
        return max(float(gru_risk), LOW_MED_BOUNDARY + 0.02), "MEDIUM"

    return float(gru_risk), risk_bucket(gru_risk)

def unknown_obstacle_risk(unknown_candidates, frame_height):
    """Return a deterministic safety floor for confirmed MiDaS obstacles.

    Metric distance/TTC for MiDaS-only obstacles are computed from raw MiDaS
    depth using the separate online MiDaS metric calibrator. MiDaS bbox area is
    never passed through the YOLO distance calibration.
    """
    confirmed = [
        c for c in unknown_candidates
        if c.get("confirmed", c.get("stable_confirmed", False))
    ]
    if not confirmed:
        return 0.0, "NONE", "none", "no confirmed unknown obstacle"

    best = max(confirmed, key=lambda c: c.get("score", 0.0))
    x, y, bw, bh = best["box"]
    bottom_frac = (y + bh) / max(float(frame_height), 1.0)
    score = float(best.get("score", 0.0))
    depth_level = float(best.get("depth_level", 0.0))

    high = (
        score >= UNKNOWN_OBSTACLE_HIGH_SCORE
        or depth_level >= UNKNOWN_OBSTACLE_HIGH_DEPTH
        or bottom_frac >= UNKNOWN_OBSTACLE_HIGH_BOTTOM_FRAC
    )

    if high:
        return (
            MED_HIGH_BOUNDARY, "HIGH", best.get("zone", "UNKNOWN"),
            f"confirmed MiDaS protrusion score={score:.2f} "
            f"depth={depth_level:.2f} bottom={bottom_frac:.2f}",
        )

    return (
        UNKNOWN_OBSTACLE_MIN_RISK, "MEDIUM", best.get("zone", "UNKNOWN"),
        f"confirmed MiDaS protrusion score={score:.2f} "
        f"depth={depth_level:.2f} bottom={bottom_frac:.2f}",
    )


def load_gru_model():
    """Load the exact fine-tuned model selected during validation."""
    try:
        gru = keras.models.load_model(GRU_MODEL_FILE)
    except Exception as e:
        raise RuntimeError(
            f"Could not load GRU model '{GRU_MODEL_FILE}'. "
            "Make sure risk_gru_model_final.keras is in the same folder "
            f"as this script. Original error: {e}"
        ) from e

    # Confirm the expected input shape.
    expected = (None, GRU_SEQ_LEN, 4)
    if len(gru.input_shape) != 3 or gru.input_shape[1:] != expected[1:]:
        raise RuntimeError(
            f"GRU input shape is {gru.input_shape}, but this live pipeline "
            f"expects {expected}."
        )

    print(f"Loaded live GRU model: {GRU_MODEL_FILE}")
    print(f"GRU input shape: {gru.input_shape}")
    return gru


def predict_live_risk(gru_model, sequence_buffer):
    """
    Run one live GRU prediction.

    sequence_buffer contains exactly 30 feature vectors with the same
    four features used by the offline model:
        normalized distance
        closing speed
        normalized class index
        assumed agent speed
    """
    if len(sequence_buffer) < GRU_SEQ_LEN:
        return GRU_WARMUP_RISK

    x = np.asarray(sequence_buffer, dtype=np.float32)
    x = x[np.newaxis, ...]  # (1, 30, 4)

    pred = gru_model.predict(x, verbose=0)

    # Model output is (1, 30, 1); use the LAST timestep because the
    # live system needs the current risk.
    risk = float(np.asarray(pred)[0, -1, 0])
    return float(np.clip(risk, 0.0, 1.0))



COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


def load_yolo_openvino():
    """Load the exported YOLOv8n OpenVINO model on the Intel GPU."""
    model_path = Path(YOLO_OPENVINO_MODEL)

    if not model_path.exists():
        raise RuntimeError(
            f"YOLO OpenVINO model not found:\n{model_path}\n"
            "Export yolov8n.pt to OpenVINO at 256x256 first."
        )

    core = ov.Core()

    if YOLO_DEVICE not in core.available_devices:
        raise RuntimeError(
            f"OpenVINO device '{YOLO_DEVICE}' is unavailable. "
            f"Available devices: {core.available_devices}"
        )

    model = core.read_model(model_path)
    compiled = core.compile_model(model, YOLO_DEVICE)

    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    print(f"Loaded YOLOv8n OpenVINO: {model_path}")
    print(f"YOLO device: {YOLO_DEVICE}")
    print(f"YOLO input shape: {input_layer.shape}")
    print(f"YOLO output shape: {output_layer.shape}")

    return compiled, input_layer, output_layer


def preprocess_yolo_openvino(frame):
    """BGR OpenCV frame -> normalized NCHW tensor."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(
        rgb,
        (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = rgb.astype(np.float32) / 255.0

    tensor = np.transpose(
        rgb,
        (2, 0, 1),
    )[None, ...]

    return tensor.astype(np.float32)


def infer_yolo_openvino(compiled_model, output_layer, frame):
    """
    Decode the standard YOLOv8 detect output:
        [1, 84, 1344]
    = 4 box coordinates + 80 class scores.

    Returns detections as:
        (raw_class_name, centroid, box_px, confidence)
    """
    result = compiled_model(
        [preprocess_yolo_openvino(frame)]
    )

    output = np.asarray(
        result[output_layer],
        dtype=np.float32,
    )

    if output.ndim != 3 or output.shape[1] < 5:
        raise RuntimeError(
            f"Unexpected YOLO OpenVINO output shape: {output.shape}"
        )

    # [1, 84, 1344] -> [1344, 84]
    predictions = output[0].T

    # First 4 values are cx, cy, w, h. Remaining values are class scores.
    boxes_cxcywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1,
    )
    confidences = class_scores[
        np.arange(class_scores.shape[0]),
        class_ids,
    ]

    keep = confidences >= YOLO_CONF_THRESHOLD

    boxes_cxcywh = boxes_cxcywh[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    if len(boxes_cxcywh) == 0:
        return []

    h, w = frame.shape[:2]

    sx = w / float(YOLO_INPUT_SIZE)
    sy = h / float(YOLO_INPUT_SIZE)

    boxes = []
    score_list = []

    for (cx, cy, bw, bh), conf in zip(
        boxes_cxcywh,
        confidences,
    ):
        x1 = int((cx - bw / 2.0) * sx)
        y1 = int((cy - bh / 2.0) * sy)
        x2 = int((cx + bw / 2.0) * sx)
        y2 = int((cy + bh / 2.0) * sy)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        bw_px = max(0, x2 - x1)
        bh_px = max(0, y2 - y1)

        boxes.append([x1, y1, bw_px, bh_px])
        score_list.append(float(conf))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        score_list,
        YOLO_CONF_THRESHOLD,
        YOLO_NMS_IOU,
    )

    if indices is None or len(indices) == 0:
        return []

    indices = np.asarray(
        indices,
        dtype=np.int32,
    ).reshape(-1)

    detections = []

    for idx in indices:
        x, y, bw_px, bh_px = boxes[int(idx)]
        x2 = x + bw_px
        y2 = y + bh_px

        cls_id = int(class_ids[int(idx)])
        cls_name = (
            COCO_NAMES[cls_id]
            if 0 <= cls_id < len(COCO_NAMES)
            else f"class_{cls_id}"
        )

        centroid = (
            (x + x2) / 2.0,
            (y + y2) / 2.0,
        )

        detections.append(
            (
                cls_name,
                centroid,
                (x, y, x2, y2),
                float(score_list[int(idx)]),
            )
        )

    return detections

def load_midas_openvino():
    """Load the official MiDaS Small OpenVINO model on Intel GPU."""
    model_path = Path(MIDAS_MODEL_FILE)

    if not model_path.exists():
        raise RuntimeError(
            f"MiDaS OpenVINO model not found:\n{model_path}"
        )

    core = ov.Core()

    if MIDAS_DEVICE not in core.available_devices:
        raise RuntimeError(
            f"OpenVINO device '{MIDAS_DEVICE}' is unavailable. "
            f"Available devices: {core.available_devices}"
        )

    model = core.read_model(model_path)
    compiled = core.compile_model(model, MIDAS_DEVICE)

    print(f"Loaded MiDaS Small: {model_path}")
    print(f"MiDaS device: {MIDAS_DEVICE}")
    print(f"MiDaS input shape: {compiled.input(0).shape}")
    print(f"MiDaS output shape: {compiled.output(0).shape}")

    return compiled


def preprocess_midas(frame):
    """Preprocess BGR OpenCV frame for MiDaS v2.1 Small 256."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(
        rgb,
        (MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = rgb.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    rgb = (rgb - mean) / std

    return np.transpose(
        rgb,
        (2, 0, 1),
    )[None, ...].astype(np.float32)


def midas_infer(compiled_model, frame):
    result = compiled_model([preprocess_midas(frame)])
    depth_raw = np.asarray(
        result[compiled_model.output(0)],
        dtype=np.float32,
    ).squeeze()

    # Per-frame normalization is only for visualization/detection. Preserve the
    # raw model output for metric calibration.
    lo = float(np.percentile(depth_raw, 2))
    hi = float(np.percentile(depth_raw, 98))
    depth_norm = np.clip(
        (depth_raw - lo) / max(hi - lo, 1e-6),
        0.0,
        1.0,
    )
    return depth_norm, depth_raw


def midas_visual(depth_norm, width, height):
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(
        depth_u8,
        cv2.COLORMAP_TURBO,
    )
    return cv2.resize(
        vis,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


class AsyncMidasWorker:
    """
    Runs MiDaS on a background thread and keeps ONLY the newest frame/result.

    The YOLO + GRU main loop never waits for MiDaS. If MiDaS is still busy,
    the main loop simply uses the most recently completed depth map.
    """

    def __init__(self, compiled_model):
        self.compiled_model = compiled_model

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_depth = None
        self._latest_raw_depth = None
        self._result_id = 0
        self._consumed_result_id = 0

        self._new_frame = threading.Event()
        self._stop = threading.Event()

        self.thread = threading.Thread(
            target=self._run,
            name="MiDaSWorker",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def submit(self, frame):
        with self._lock:
            self._latest_frame = frame.copy()
        self._new_frame.set()

    def get_latest(self):
        """Return the latest completed depth map for display/inspection."""
        with self._lock:
            if self._latest_depth is None:
                return None
            return self._latest_depth.copy()

    def get_new_result(self):
        """Return (normalized_depth, raw_depth) once per MiDaS inference."""
        with self._lock:
            if self._latest_depth is None or self._latest_raw_depth is None:
                return None
            if self._result_id == self._consumed_result_id:
                return None
            self._consumed_result_id = self._result_id
            return self._latest_depth.copy(), self._latest_raw_depth.copy()

    def stop(self):
        self._stop.set()
        self._new_frame.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            self._new_frame.wait(timeout=0.1)
            self._new_frame.clear()

            if self._stop.is_set():
                break

            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                continue

            try:
                depth_norm, depth_raw = midas_infer(
                    self.compiled_model,
                    frame,
                )

                with self._lock:
                    self._latest_depth = depth_norm
                    self._latest_raw_depth = depth_raw
                    self._result_id += 1

            except Exception as e:
                print(f"MiDaS worker error: {e}")


# ---- PROVEN MiDaS UNKNOWN-OBSTACLE DETECTOR ----

def grow_depth_region(depth, seed_box, seed_depth, frame_shape):
    """V14: disabled depth-plateau growth.

    The protrusion mask is the trusted geometry source. Growing a seed through
    a broad relative-depth plateau can absorb desks, laptops, walls or floors
    that happen to have similar MiDaS values. Return the actual mask component
    bbox unchanged; fragment grouping is handled separately by
    merge_midas_candidates().
    """
    x, y, bw, bh = [int(v) for v in seed_box]
    return (x, y, bw, bh), float((bw * bh) / max(float(frame_shape[0] * frame_shape[1]), 1.0))

def analyze_depth(depth_norm, frame_shape):
    """Fast MiDaS protrusion detector operating at MiDaS native resolution.

    The previous version resized the 256x256 MiDaS map to the full camera
    resolution before every blur/Sobel/morphology operation. That made the
    detector much more expensive than necessary. We now do all numerical
    processing in the native MiDaS grid and upscale only the final mask once.
    Candidate boxes are converted back to native camera coordinates before
    returning, so downstream tracking/fusion logic remains unchanged.
    """
    if (not isinstance(depth_norm, np.ndarray)
            or depth_norm.size == 0
            or depth_norm.ndim != 2):
        h, w = frame_shape[:2]
        return np.zeros((h, w), dtype=np.uint8), [], 0.0, {}

    h, w = frame_shape[:2]
    depth = depth_norm.astype(np.float32, copy=False)
    dh, dw = depth.shape[:2]

    # Forward/ground region, evaluated on the native 256x256 grid.
    y0 = int(dh * 0.32)
    y1 = int(dh * 0.94)
    y1 = max(y0 + 1, min(dh, y1))
    roi = depth[y0:y1, :]

    if roi.size == 0:
        return np.zeros((h, w), dtype=np.uint8), [], 0.0, {}

    # Scale the old full-frame blur sizes down to MiDaS native resolution.
    scale = min(dw / 640.0, dh / 480.0)
    sigma_large = max(3.0, 18.0 * scale)
    sigma_small = max(1.0, 4.0 * scale)

    smooth = cv2.GaussianBlur(
        roi, (0, 0), sigmaX=sigma_large, sigmaY=sigma_large
    )
    residual = roi - smooth

    residual_smooth = cv2.GaussianBlur(
        residual, (0, 0), sigmaX=sigma_small, sigmaY=sigma_small
    )

    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    grad_scale = float(np.percentile(gradient, 90))
    if grad_scale > 1e-6:
        gradient_n = np.clip(gradient / grad_scale, 0.0, 1.0)
    else:
        gradient_n = np.zeros_like(gradient)

    near_threshold = float(np.percentile(roi, 70.0))
    near = roi >= near_threshold

    abs_residual = np.abs(residual_smooth)
    residual_med = float(np.median(abs_residual))
    residual_mad = float(
        np.median(np.abs(abs_residual - residual_med))
    )
    residual_threshold = max(
        0.055,
        residual_med + 3.0 * residual_mad,
    )

    protrusion = (
        (residual_smooth >= residual_threshold)
        & near
        & (gradient_n >= 0.08)
    ).astype(np.uint8) * 255

    # Keep morphology intentionally light at native resolution.
    kernel = np.ones((3, 3), np.uint8)
    protrusion = cv2.morphologyEx(
        protrusion, cv2.MORPH_OPEN, kernel
    )
    protrusion = cv2.morphologyEx(
        protrusion, cv2.MORPH_CLOSE, kernel
    )

    # Connect small broken edge pieces, but don't bridge large regions.
    bridge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    protrusion = cv2.morphologyEx(
        protrusion,
        cv2.MORPH_CLOSE,
        bridge_kernel,
        iterations=1,
    )

    # One upscale operation for display/mask persistence. Numerical processing
    # above remains entirely on the cheap native MiDaS grid.
    full_mask = cv2.resize(
        protrusion,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        protrusion, connectivity=8
    )

    candidates = []

    # Scale minimum area from the old 640x480 processing grid to native grid.
    reference_area = 640.0 * 480.0
    native_area = float(dw * dh)
    min_area_native = max(
        25,
        int(round(250.0 * native_area / reference_area)),
    )
    min_height_native = max(6, int(round(dh * 0.035)))
    min_width_native = max(5, int(round(dw * 0.03)))

    sx = w / float(dw)
    sy = h / float(dh)

    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bw_native = int(stats[label, cv2.CC_STAT_WIDTH])
        bh_native = int(stats[label, cv2.CC_STAT_HEIGHT])
        x_native = int(stats[label, cv2.CC_STAT_LEFT])
        y_native = int(stats[label, cv2.CC_STAT_TOP])

        if area < min_area_native:
            continue
        if bh_native < min_height_native:
            continue
        if bw_native < min_width_native:
            continue

        fill = area / max(float(bw_native * bh_native), 1.0)
        if fill < 0.10:
            continue

        component = labels == label
        component_residual = residual_smooth[component]
        component_gradient = gradient_n[component]
        component_depth = roi[component]

        if component_residual.size == 0:
            continue

        contrast = float(np.median(component_residual))
        edge_support = float(np.mean(component_gradient))
        depth_level = float(np.median(component_depth))

        if contrast < residual_threshold:
            continue

        aspect = bh_native / max(float(bw_native), 1.0)
        if aspect < 0.12 and bw_native > int(dw * 0.20):
            continue

        # Convert native ROI/component coordinates back to full-frame pixels.
        x = int(round(x_native * sx))
        y = int(round((y_native + y0) * sy))
        bw = max(1, int(round(bw_native * sx)))
        bh = max(1, int(round(bh_native * sy)))

        cx = x + bw / 2.0
        if cx < w * 0.34:
            zone = "LEFT"
        elif cx < w * 0.66:
            zone = "CENTER"
        else:
            zone = "RIGHT"

        final_box = (x, y, bw, bh)
        grown_area_frac = (bw * bh) / max(float(w * h), 1.0)

        score = (
            min(contrast / 0.16, 1.0) * 0.50
            + min(edge_support / 0.50, 1.0) * 0.20
            + min(fill / 0.70, 1.0) * 0.15
            + depth_level * 0.15
        )

        fx, fy, fw, fh = final_box
        if fw < int(w * 0.030) or fh < int(h * 0.040):
            continue

        candidates.append({
            "zone": zone,
            "box": final_box,
            "seed_box": final_box,
            "area": area,
            "grown_area_frac": grown_area_frac,
            "component_fill": fill,
            "depth_contrast": contrast,
            "edge_support": edge_support,
            "depth_level": depth_level,
            "score": float(score),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    zone_stats = {
        "threshold": residual_threshold,
        "mean_abs_residual": float(np.mean(abs_residual)),
        "max_positive_residual": float(np.max(residual_smooth)),
        "near_fraction": float(np.mean(near)),
        "candidates": len(candidates),
        "processing_grid": f"{dw}x{dh}",
    }

    return full_mask, candidates, residual_threshold, zone_stats

# ---- TEMPORAL STABILITY ----

STABILITY_WINDOW = 5
STABILITY_HITS_REQUIRED = 3
MAX_MISSED_CONFIRMED_FRAMES = 2

def stabilize_candidates(candidates, state):
    """
    Convert flickering per-frame candidates into stable obstacle tracks.

    A zone is confirmed when a candidate is present in >=3 of the last 5
    frames. The box is exponentially smoothed, so it does not jump around.
    """
    by_zone = {}

    # Keep only the strongest candidate per navigation zone.
    for candidate in candidates:
        zone = candidate["zone"]

        if (
            zone not in by_zone
            or candidate["score"]
            > by_zone[zone]["score"]
        ):
            by_zone[zone] = candidate

    stable = []

    for zone in ("LEFT", "CENTER", "RIGHT"):
        s = state.setdefault(
            zone,
            {
                "history": deque(maxlen=STABILITY_WINDOW),
                "smooth_box": None,
                "confirmed": False,
                "missed": 0,
                "last_candidate": None,
            },
        )

        candidate = by_zone.get(zone)

        # Record whether this frame has valid evidence in this zone.
        s["history"].append(candidate is not None)

        if candidate is not None:
            new_box = candidate["box"]

            # Reject implausible one-frame bbox explosions/shrinks. MiDaS can
            # momentarily attach to a nearby depth edge; temporal stability
            # should not let that redefine the tracked object immediately.
            if s["smooth_box"] is not None:
                px, py, pw, ph = s["smooth_box"]
                nx, ny, nw, nh = new_box
                max_w = max(pw * 1.45, 40)
                max_h = max(ph * 1.45, 40)
                min_w = min(pw * 0.70, max(8, pw - 12))
                min_h = min(ph * 0.70, max(8, ph - 12))
                nx = int(np.clip(nx, px - max(pw * 0.22, 30), px + max(pw * 0.22, 30)))
                ny = int(np.clip(ny, py - max(ph * 0.22, 30), py + max(ph * 0.22, 30)))
                nw = int(np.clip(nw, min_w, max_w))
                nh = int(np.clip(nh, min_h, max_h))
                new_box = (nx, ny, nw, nh)

            if s["smooth_box"] is None:
                smooth = tuple(
                    int(v)
                    for v in new_box
                )
            else:
                old = s["smooth_box"]

                smooth = tuple(
                    int(
                        BOX_SMOOTH_ALPHA * new
                        + (1.0 - BOX_SMOOTH_ALPHA) * prev
                    )
                    for new, prev in zip(
                        new_box,
                        old,
                    )
                )

            s["smooth_box"] = smooth
            s["last_candidate"] = candidate
            s["missed"] = 0

        else:
            s["missed"] += 1

        hits = sum(s["history"])

        if hits >= STABILITY_HITS_REQUIRED:
            s["confirmed"] = True
        elif (
            not s["confirmed"]
            and hits == 0
        ):
            s["confirmed"] = False

        # Once confirmed, tolerate only a very short gap.
        if (
            s["confirmed"]
            and candidate is None
            and s["missed"] > MAX_MISSED_CONFIRMED_FRAMES
        ):
            s["confirmed"] = False
            s["smooth_box"] = None
            s["last_candidate"] = None
            s["missed"] = 0

        if s["confirmed"] and s["last_candidate"] is not None:
            out = dict(s["last_candidate"])

            out["box"] = s["smooth_box"]
            out["stable_hits"] = hits
            out["stable_window"] = STABILITY_WINDOW
            out["confirmed"] = True
            out["stale"] = candidate is None

            stable.append(out)

    return stable

def raw_midas_depth_for_box(raw_depth, box, frame_shape):
    """Robust raw MiDaS depth statistic without full-frame upscaling."""
    if (
        raw_depth is None
        or not isinstance(raw_depth, np.ndarray)
        or raw_depth.size == 0
        or raw_depth.ndim != 2
    ):
        return float("nan")

    h, w = frame_shape[:2]
    dh, dw = raw_depth.shape[:2]
    sx = dw / max(float(w), 1.0)
    sy = dh / max(float(h), 1.0)

    x, y, bw, bh = [int(v) for v in box]
    x0 = max(0, min(w - 1, x))
    y0 = max(0, min(h - 1, y))
    x1 = max(x0 + 1, min(w, x + bw))
    y1 = max(y0 + 1, min(h, y + bh))

    dx0 = max(0, min(dw - 1, int(round(x0 * sx))))
    dy0 = max(0, min(dh - 1, int(round(y0 * sy))))
    dx1 = max(dx0 + 1, min(dw, int(round(x1 * sx))))
    dy1 = max(dy0 + 1, min(dh, int(round(y1 * sy))))

    patch = raw_depth[dy0:dy1, dx0:dx1].astype(np.float32, copy=False)
    if patch.size == 0:
        return float("nan")

    py0 = int(patch.shape[0] * 0.15)
    py1 = max(py0 + 1, int(patch.shape[0] * 0.85))
    px0 = int(patch.shape[1] * 0.15)
    px1 = max(px0 + 1, int(patch.shape[1] * 0.85))
    core = patch[py0:py1, px0:px1]
    vals = core[np.isfinite(core)]
    if vals.size < 8:
        vals = patch[np.isfinite(patch)]
    return float(np.median(vals)) if vals.size else float("nan")

def update_unknown_tracks(raw_candidates, stable_candidates, unknown_tracks, now, frame_area, raw_midas_depth, frame_shape, midas_calibrator):
    """Maintain one metric-distance/TTC track per MiDaS navigation zone.

    Raw MiDaS candidates update the distance history every depth result, so
    once temporal stability confirms an obstacle we already have a short
    distance history for closing-speed/TTC estimation. Only confirmed zones
    are returned as active unknown tracks.
    """
    stable_by_zone = {
        c.get("zone"): c
        for c in stable_candidates
        if c.get("confirmed", c.get("stable_confirmed", False))
    }
    stable_zones = set(stable_by_zone)

    for candidate in raw_candidates:
        zone = candidate.get("zone")
        if zone not in ("LEFT", "CENTER", "RIGHT"):
            continue

        # Once a MiDaS obstacle is temporally confirmed, use the SAME smoothed
        # box that is drawn on screen for its distance estimate. This prevents
        # the displayed box and the GRU distance from disagreeing.
        measurement = stable_by_zone.get(zone, candidate)
        x, y, bw, bh = measurement["box"]
        centroid = (x + bw / 2.0, y + bh / 2.0)
        raw_midas = raw_midas_depth_for_box(
            raw_midas_depth, (x, y, bw, bh), frame_shape
        )
        measured_dist, metric_ready = midas_calibrator.predict(raw_midas)

        tr = unknown_tracks.get(zone)
        if tr is None:
            tr = Track(UNKNOWN_GRU_CLASS, centroid, measured_dist, now)
            tr.source = "MiDaS"
            tr.box = (x, y, x + bw, y + bh)
            unknown_tracks[zone] = tr
        else:
            tr.update(
                UNKNOWN_GRU_CLASS,
                centroid,
                (x, y, x + bw, y + bh),
                measured_dist,
                now,
                frame_area,
            )
            tr.source = "MiDaS"

    active = []
    for zone, tr in list(unknown_tracks.items()):
        if zone in stable_zones and tr.box is not None:
            stable = next(
                (c for c in stable_candidates if c.get("zone") == zone),
                None,
            )
            if stable is not None:
                x, y, bw, bh = stable["box"]
                tr.box = (x, y, x + bw, y + bh)
            active.append(tr)
        elif now - tr.last_seen > TRACK_TIMEOUT_S:
            del unknown_tracks[zone]

    return active


def safe_imshow(window_name, image):
    if not isinstance(image, np.ndarray):
        return
    if image.ndim < 2 or image.size == 0:
        return
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return

    # Downscale only the GUI image. Processing remains at native resolution.
    # This keeps the three-panel OpenCV window responsive on the laptop.
    display = image
    max_width = int(DISPLAY_MAX_WIDTH)
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        display = cv2.resize(
            image,
            (max_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    try:
        cv2.imshow(
            window_name,
            np.ascontiguousarray(display),
        )
    except cv2.error as exc:
        print(f"OpenCV display warning: {exc}")

def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ab = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(aa + ab - inter, 1.0)


def boxes_same_object(a, b):
    """Conservative physical-object association for YOLO <-> MiDaS.

    IoU alone is unreliable because YOLO can see only a semantic fragment
    while MiDaS can recover a much larger depth silhouette.  We therefore use
    overlap/containment plus centroid proximity, but *not* a simple
    center-inside rule: that rule was responsible for unrelated floor/table
    regions being fused to nearby objects.
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iou = box_iou_xyxy(a, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (ax2-ax1)*(ay2-ay1))
    area_b = max(1.0, (bx2-bx1)*(by2-by1))
    containment = max(inter/area_a, inter/area_b)

    acx, acy = (ax1+ax2)/2.0, (ay1+ay2)/2.0
    bcx, bcy = (bx1+bx2)/2.0, (by1+by2)/2.0
    aw, ah = ax2-ax1, ay2-ay1
    bw, bh = bx2-bx1, by2-by1
    diag = max(1.0, ((aw+ah+ bw+bh)/4.0))
    center_dist = ((acx-bcx)**2 + (acy-bcy)**2) ** 0.5

    x_overlap = max(0.0, min(ax2,bx2)-max(ax1,bx1)) / max(1.0, min(aw,bw))
    y_overlap = max(0.0, min(ay2,by2)-max(ay1,by1)) / max(1.0, min(ah,bh))

    return (
        iou >= 0.15
        or containment >= YOLO_MIDAS_MIN_CONTAINMENT
        or (x_overlap >= 0.45 and y_overlap >= 0.30)
        or (center_dist <= YOLO_MIDAS_CENTER_FRAC * max(aw,ah,bw,bh,diag)
            and x_overlap >= 0.18 and y_overlap >= 0.18)
    )


def union_box(a, b):
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def merge_midas_candidates(candidates, frame_w, frame_h):
    """Merge disconnected MiDaS mask fragments into ONE physical obstacle.

    The protrusion mask is an edge-like representation, so a single object
    can produce a top edge, side edge and bottom edge as separate connected
    components.  We merge those components before tracking.  A merge needs
    overlap/proximity evidence *and* similar depth; this prevents nearby but
    separate floor/table regions from becoming one giant box.
    """
    if not candidates:
        return []

    work = [dict(c) for c in candidates]

    def xyxy(c):
        x,y,bw,bh=c["box"]
        return (float(x),float(y),float(x+bw),float(y+bh))

    def depth_similar(a,b):
        da=float(a.get("depth_level",0.0))
        db=float(b.get("depth_level",0.0))
        # Relative MiDaS values are not metric; compare locally.
        return abs(da-db) <= 0.16

    def should_merge(a,b):
        ab=xyxy(a); bb=xyxy(b)
        # Never allow fragment chaining to create a giant physical-object box.
        ux1=min(ab[0],bb[0]); uy1=min(ab[1],bb[1])
        ux2=max(ab[2],bb[2]); uy2=max(ab[3],bb[3])
        uw=ux2-ux1; uh=uy2-uy1
        if (uw > frame_w*0.58 or uh > frame_h*0.62 or
                uw*uh > frame_w*frame_h*0.20):
            return False
        if not depth_similar(a,b):
            return False
        iou=box_iou_xyxy(ab,bb)
        if iou >= 0.03:
            return True

        ax1,ay1,ax2,ay2=ab; bx1,by1,bx2,by2=bb
        xgap=max(0.0,max(bx1-ax2,ax1-bx2))
        ygap=max(0.0,max(by1-ay2,ay1-by2))
        xover=max(0.0,min(ax2,bx2)-max(ax1,bx1)) / max(1.0,min(ax2-ax1,bx2-bx1))
        yover=max(0.0,min(ay2,by2)-max(ay1,by1)) / max(1.0,min(ay2-ay1,by2-by1))

        # Pieces of the same object commonly touch/approach each other with
        # substantial overlap in one axis.
        if xgap <= frame_w*MIDAS_MERGE_GAP_FRAC_V14 and yover >= MIDAS_MERGE_MIN_OVERLAP_V14:
            return True
        if ygap <= frame_h*MIDAS_MERGE_GAP_FRAC_V14 and xover >= MIDAS_MERGE_MIN_OVERLAP_V14:
            return True

        acx=(ax1+ax2)/2; acy=(ay1+ay2)/2
        bcx=(bx1+bx2)/2; bcy=(by1+by2)/2
        center_dist=((acx-bcx)**2+(acy-bcy)**2)**0.5
        scale=max(ax2-ax1,ay2-ay1,bx2-bx1,by2-by1,1.0)
        return center_dist <= MIDAS_MERGE_CENTER_FRAC_V14*scale and (xover>=0.10 or yover>=0.10)

    changed=True
    while changed and len(work)>1:
        changed=False
        best_pair=None
        best_score=-1.0
        for i in range(len(work)):
            for j in range(i+1,len(work)):
                if not should_merge(work[i],work[j]):
                    continue
                score=box_iou_xyxy(xyxy(work[i]),xyxy(work[j])) + 0.01*max(float(work[i].get("score",0)),float(work[j].get("score",0)))
                if score>best_score:
                    best_score=score; best_pair=(i,j)
        if best_pair is None:
            break
        i,j=best_pair
        a=work[i]; b=work[j]
        u=union_box(xyxy(a),xyxy(b))
        ux1,uy1,ux2,uy2=u
        new_box = (int(ux1), int(uy1), int(ux2-ux1), int(uy2-uy1))
        a["box"] = new_box
        for k in ("score","depth_level","grown_area_frac","component_fill","depth_contrast","edge_support"):
            a[k]=max(float(a.get(k,0.0)),float(b.get(k,0.0)))
        a["area"]=int(a.get("area",0))+int(b.get("area",0))
        cx=(ux1+ux2)/2.0
        a["zone"]="LEFT" if cx < frame_w*0.34 else ("CENTER" if cx < frame_w*0.66 else "RIGHT")
        work.pop(j)
        changed=True

    result=[]
    for c in work:
        x,y,bw,bh=c["box"]
        # Reject thin strips and tiny specks after merging.
        if bw < frame_w*0.055 or bh < frame_h*0.065:
            continue
        if y+bh > frame_h*0.98 and bh < frame_h*0.12:
            continue
        result.append(c)

    result.sort(key=lambda c: float(c.get("score",0.0)), reverse=True)
    return result


def main():
    yolo_model, yolo_input, yolo_output = load_yolo_openvino()
    gru_model = load_gru_model()
    midas_model = load_midas_openvino()

    midas_worker = AsyncMidasWorker(midas_model)
    midas_worker.start()
    midas_calibrator = MidasMetricCalibrator()

    reader = LatestFrameReader(SOURCE)
    ultrasonic = UltrasonicReceiver(host="0.0.0.0", port=4210)
    speech = SpeechManager()
    groq_navigation = GroqNavigationManager(speech)

    try:
        reader.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )
    except Exception:
        pass

    tracks = []
    feature_buffer = deque(maxlen=GRU_SEQ_LEN)

    # Keep the most recent YOLO results available on frames where YOLO is
    # intentionally skipped. This prevents detections/all_boxes from becoming
    # undefined when YOLO runs only every Nth frame.
    detections = []
    all_boxes = []
    yolo_detections = []

    # Persistent fallback obstacle used when HC-SR04 has a valid reading but
    # neither YOLO nor MiDaS provides a corresponding object.
    ultrasonic_virtual_track = None
    rear_ultrasonic_virtual_track = None
    front_ultrasonic_match = None
    ultrasonic_direction = None

    # Long-range vision-assisted ultrasonic override state.
    ultrasonic_vision_candidate_key = None
    ultrasonic_vision_candidate_start = None
    ultrasonic_vision_candidate_updates = 0
    ultrasonic_vision_override_active = False
    ultrasonic_vision_override_track = None

    live_risk = GRU_WARMUP_RISK
    live_risk_target = GRU_WARMUP_RISK

    frame_counter = 0
    midas_counter = 0
    pipeline_frame_counter = 0

    latest_midas_depth = None
    latest_midas_raw_depth = None
    latest_unknown_mask = None
    unknown_candidates = []
    raw_unknown_candidates = []
    unknown_threshold = 0.0

    obstacle_state = {}
    unknown_tracks = {}

    # Once a depth result exists, keep the display panels alive using the
    # last valid result instead of hiding them on intermittent worker timing.
    have_depth_result = False

    fps = 0.0
    last_loop_time = time.perf_counter()
    FPS_SMOOTHING = 0.08

    print("MiDaS unknown-obstacle detector: ENABLED")
    print("Detector: proven local depth protrusion")
    print("Temporal stability: 3/5 frames")
    print("Unknown obstacle: unified distance/TTC/GRU path + safety floor.")
    print("Depth/BW panel: persistent after first valid MiDaS result.")
    print("MiDaS bbox: native-grid protrusion geometry + balanced fragment merge.")
    print("YOLO labels: full semantic classes preserved; 6-class mapping only at GRU input.")
    print("MiDaS processing: native-grid analysis; full-resolution upscale only for display.")
    print(
        "Long-range failsafe: HC-SR04 <= "
        f"{ULTRASONIC_VISION_OVERRIDE_START_M:.2f}m authoritative; "
        "stable HIGH-confidence forward vision may override beyond this."
    )
    print(
        "Vision confidence: HIGH required; "
        f"margin={ULTRASONIC_VISION_OVERRIDE_MARGIN_M:.2f}m, "
        f"confirmation={ULTRASONIC_VISION_CONFIRM_S:.1f}s/"
        f"{ULTRASONIC_VISION_MIN_UPDATES} updates."
    )
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = reader.read()

            if not ok or frame is None:
                time.sleep(0.002)
                continue

            if (
                not isinstance(frame, np.ndarray)
                or frame.size == 0
                or frame.ndim < 2
                or frame.shape[0] <= 0
                or frame.shape[1] <= 0
            ):
                continue

            if ROTATE:
                frame = cv2.rotate(
                    frame,
                    cv2.ROTATE_90_CLOCKWISE,
                )

            frame = resize_fixed(
                frame,
                PROCESS_WIDTH,
            )

            if (
                frame is None
                or frame.size == 0
                or frame.shape[0] <= 0
                or frame.shape[1] <= 0
            ):
                continue

            h, w = frame.shape[:2]
            pipeline_frame_counter += 1

            # ---------------------------------------------------------
            # Loop FPS
            # ---------------------------------------------------------
            loop_now = time.perf_counter()
            dt = loop_now - last_loop_time
            last_loop_time = loop_now

            if dt > 0:
                instant_fps = 1.0 / dt
                fps = (
                    instant_fps
                    if fps <= 0
                    else FPS_SMOOTHING * instant_fps
                    + (1.0 - FPS_SMOOTHING) * fps
                )

            # ---------------------------------------------------------
            # Async MiDaS
            # ---------------------------------------------------------
            midas_counter += 1

            if (
                midas_counter
                % MIDAS_INFERENCE_EVERY_N_FRAMES
                == 0
            ):
                midas_worker.submit(frame)

            depth_result = midas_worker.get_new_result()

            if depth_result is not None:
                latest_midas_depth, latest_midas_raw_depth = depth_result
                if (
                    not isinstance(latest_midas_depth, np.ndarray)
                    or not isinstance(latest_midas_raw_depth, np.ndarray)
                    or latest_midas_depth.size == 0
                    or latest_midas_raw_depth.size == 0
                ):
                    continue
                have_depth_result = True

                # Calculate/update the B/W mask every time we receive a new
                # valid depth map. Store the mask BEFORE stability logic so a
                # temporary stabilizer error cannot make the panel disappear.
                try:
                    (
                        new_mask,
                        raw_unknown_candidates,
                        new_threshold,
                        _unknown_stats,
                    ) = analyze_depth(
                        latest_midas_depth,
                        frame.shape,
                    )

                    latest_unknown_mask = new_mask
                    unknown_threshold = new_threshold

                    # IMPORTANT: the protrusion mask can split one physical
                    # object into several disconnected components. Collapse
                    # those fragments BEFORE stability/tracking and BEFORE
                    # YOLO-vs-MiDaS association.
                    raw_unknown_candidates = merge_midas_candidates(
                        raw_unknown_candidates,
                        frame.shape[1],
                        frame.shape[0],
                    )

                    # Temporal stability cannot invalidate the underlying
                    # mask. Only the yellow stable boxes depend on this step.
                    unknown_candidates = (
                        stabilize_candidates(
                            raw_unknown_candidates,
                            obstacle_state,
                        )
                    )

                except Exception as exc:
                    # Preserve the last valid depth/mask/candidates.
                    print(
                        f"MiDaS diagnostic warning: {exc}"
                    )

            # ---------------------------------------------------------
            # YOLO
            # ---------------------------------------------------------
            # The tracker needs a current timestamp on EVERY frame,
            # including frames where YOLO inference is skipped.
            now = time.time()

            # Run YOLO every N frames. On skipped frames, reuse the most
            # recent detections instead of performing another GPU inference.
            fresh_yolo = (
                pipeline_frame_counter % YOLO_INFERENCE_EVERY_N_FRAMES == 0
            )

            if fresh_yolo:
                yolo_detections = infer_yolo_openvino(
                    yolo_model,
                    yolo_output,
                    frame,
                )

                detections = []
                all_boxes = []

                for (
                    cls_name_raw,
                    centroid,
                    box_px,
                    confidence,
                ) in yolo_detections:
                    all_boxes.append(
                        (
                            cls_name_raw,
                            box_px,
                            confidence,
                        )
                    )

                    mapped = CLASS_MAP.get(
                        cls_name_raw,
                        cls_name_raw,
                    )

                    x1, y1, x2, y2 = box_px

                    area_frac = (
                        (
                            (x2 - x1)
                            * (y2 - y1)
                        )
                        / max(
                            w * h,
                            1,
                        )
                    )

                    yolo_dist = bbox_area_to_distance(
                        area_frac
                    )

                    # Fuse the calibrated YOLO metric anchor with local MiDaS
                    # depth evidence. MiDaS is never interpreted as metres here.
                    fused_dist = yolo_dist
                    midas_corr = 1.0
                    midas_ratio = 1.0
                    midas_strength = 0.0

                    if (
                        DIST_FUSION_ENABLED
                        and latest_midas_raw_depth is not None
                    ):
                        (
                            midas_corr,
                            obj_depth,
                            bg_depth,
                            midas_strength,
                        ) = midas_relative_correction(
                            latest_midas_raw_depth,
                            box_px,
                            frame.shape,
                        )

                        if (
                            np.isfinite(obj_depth)
                            and np.isfinite(bg_depth)
                            and bg_depth > 1e-6
                        ):
                            midas_ratio = float(
                                np.clip(
                                    obj_depth / bg_depth,
                                    MIDAS_RATIO_MIN,
                                    MIDAS_RATIO_MAX,
                                )
                            )

                            effective_corr = (
                                1.0
                                + (
                                    midas_corr - 1.0
                                ) * midas_strength
                            )

                            fused_dist = float(
                                np.clip(
                                    yolo_dist * effective_corr,
                                    0.2,
                                    MAX_RANGE,
                                )
                            )

                    detections.append(
                        (
                            mapped,
                            centroid,
                            box_px,
                            fused_dist,
                        )
                    )

                # Update the tracker only when YOLO has produced fresh boxes.
                tracks = match_detections_to_tracks(
                    detections,
                    tracks,
                    w,
                    h,
                    now,
                )
                assign_yolo_confidence_to_tracks(
                    tracks,
                    yolo_detections,
                    w,
                )
            
            # ---------------------------------------------------------
            # PRIORITY FUSION / TOP-3 POLICY
            # ---------------------------------------------------------
            # Policy:
            #   1) YOLO owns the three primary obstacle slots.
            #   2) MiDaS is allowed to improve the geometry of a YOLO box
            #      when both describe the same physical object.
            #   3) A confirmed MiDaS-only obstacle can replace ONLY the
            #      farthest YOLO slot, and only when it is genuinely closer.
            #   4) Never display separate YOLO + MiDaS boxes for one object.
            #
            # This is deliberately different from simply concatenating both
            # detector outputs. The final navigation layer always contains
            # at most MAX_TRACKS physical obstacles.
            fused_zones = set()

            # First: keep only the three nearest valid YOLO detections.
            detections.sort(key=lambda d: d[3])
            detections = detections[:MAX_TRACKS]

            # Second: use MiDaS only as geometry enhancement for those YOLO
            # objects. If a depth region overlaps/contains a YOLO box, it is
            # considered the SAME physical obstacle.
            if unknown_candidates:
                fused_detections = []
                used_midas = set()

                for raw_cls_name, centroid, box_px, raw_dist in detections:
                    best = None
                    best_score = -1.0

                    for idx, uc in enumerate(unknown_candidates):
                        if idx in used_midas:
                            continue

                        ub = (
                            int(uc["box"][0]),
                            int(uc["box"][1]),
                            int(uc["box"][0] + uc["box"][2]),
                            int(uc["box"][1] + uc["box"][3]),
                        )

                        if boxes_same_object(box_px, ub):
                            score = box_iou_xyxy(box_px, ub)
                            # Prefer the candidate with the strongest overlap;
                            # containment also counts through boxes_same_object.
                            if score > best_score:
                                best_score = score
                                best = (idx, uc, ub)

                    if best is not None:
                        idx, uc, ub = best
                        used_midas.add(idx)
                        fused_zones.add(uc["zone"])

                        # YOLO supplies semantic identity AND metric distance.
                        # MiDaS supplies geometry only for this fused object.
                        # Learn the separate MiDaS metric mapping from this pair.
                        if latest_midas_raw_depth is not None:
                            midas_raw = raw_midas_depth_for_box(
                                latest_midas_raw_depth, uc["box"], frame.shape
                            )
                            midas_calibrator.add(midas_raw, raw_dist)

                        box_px = union_box(box_px, ub)
                        x1, y1, x2, y2 = box_px
                        centroid = (
                            (x1 + x2) / 2.0,
                            (y1 + y2) / 2.0,
                        )

                    fused_detections.append(
                        (raw_cls_name, centroid, box_px, raw_dist)
                    )

                detections = fused_detections

            # Keep frame-local fusion diagnostics keyed by bbox.
            fusion_diag = {}
            for _mapped, _centroid, _box, _fused_dist in detections:
                _area_frac = (
                    ((_box[2] - _box[0]) * (_box[3] - _box[1]))
                    / max(float(w * h), 1.0)
                )
                _yolo_anchor = bbox_area_to_distance(_area_frac)
                _corr = 1.0
                _ratio = 1.0
                _strength = 0.0
                if (
                    DIST_FUSION_ENABLED
                    and latest_midas_raw_depth is not None
                ):
                    (
                        _corr_raw,
                        _obj_d,
                        _bg_d,
                        _strength,
                    ) = midas_relative_correction(
                        latest_midas_raw_depth,
                        _box,
                        frame.shape,
                    )
                    if np.isfinite(_obj_d) and np.isfinite(_bg_d) and _bg_d > 1e-6:
                        _ratio = float(
                            np.clip(
                                _obj_d / _bg_d,
                                MIDAS_RATIO_MIN,
                                MIDAS_RATIO_MAX,
                            )
                        )
                        _corr = float(
                            1.0
                            + (_corr_raw - 1.0) * _strength
                        )
                fusion_diag[_box] = (
                    _yolo_anchor,
                    _corr,
                    _ratio,
                    _strength,
                )

            tracks = match_detections_to_tracks(
                detections,
                tracks,
                w,
                h,
                now,
            )

            for tr in tracks:
                if tr.box is not None:
                    diag = fusion_diag.get(
                        tuple(map(int, tr.box))
                    )
                    if diag is not None:
                        (
                            tr.last_yolo_distance,
                            tr.last_midas_correction,
                            tr.last_midas_ratio,
                            tr.last_midas_strength,
                        ) = diag

            for tr in tracks:
                tr.source = "YOLO"

            # ---------------------------------------------------------
            # MiDaS-only fallback candidates.
            # ---------------------------------------------------------
            # Only candidates NOT fused with a YOLO obstacle can become
            # unknown obstacles. Require temporal confirmation here so a
            # one-frame protrusion never displaces a YOLO object.
            midas_only_candidates = [
                c for c in (raw_unknown_candidates if have_depth_result else [])
                if c.get("zone") not in fused_zones
            ]
            midas_only_stable = [
                c for c in unknown_candidates
                if c.get("zone") not in fused_zones
            ]

            unknown_tracks_list = update_unknown_tracks(
                midas_only_candidates,
                midas_only_stable,
                unknown_tracks,
                now,
                w * h,
                latest_midas_raw_depth,
                frame.shape,
                midas_calibrator,
            )

            # A final geometric duplicate check protects against cases where
            # the MiDaS candidate came from a neighboring zone but is still the
            # same physical object as a YOLO box.
            known_boxes = [
                tr.box for tr in tracks
                if tr.box is not None
            ]
            unknown_tracks_list = [
                tr for tr in unknown_tracks_list
                if tr.box is not None
                and not any(
                    boxes_same_object(tr.box, kb)
                    for kb in known_boxes
                )
            ]

            # ---------------------------------------------------------
            # REPLACEMENT RULE
            # ---------------------------------------------------------
            # YOLO gets the first three slots. MiDaS can occupy a slot only
            # when its confirmed obstacle is closer than the current farthest
            # YOLO obstacle. This gives the exact priority requested:
            #
            #       YOLO #1, #2, #3
            #               ↓
            #       MiDaS unknown closer?
            #          YES → replace #3
            #          NO  → keep YOLO #3
            # ---------------------------------------------------------
            yolo_tracks = sorted(
                tracks,
                key=lambda tr: tr.smoothed_dist,
            )[:MAX_TRACKS]

            confirmed_unknown_tracks = sorted(
                unknown_tracks_list,
                key=lambda tr: tr.smoothed_dist,
            )

            final_obstacles = list(yolo_tracks)

            for utr in confirmed_unknown_tracks:
                if len(final_obstacles) < MAX_TRACKS:
                    final_obstacles.append(utr)
                    continue

                farthest_yolo = max(
                    final_obstacles,
                    key=lambda tr: tr.smoothed_dist,
                )

                # MiDaS must be meaningfully closer before it is allowed to
                # displace a YOLO slot. The margin avoids rapid A/B swapping
                # when both distance estimates are almost identical.
                if utr.smoothed_dist < (
                    farthest_yolo.smoothed_dist - UNKNOWN_REPLACEMENT_MARGIN_M
                ):
                    final_obstacles.remove(farthest_yolo)
                    final_obstacles.append(utr)

            # ---------------------------------------------------------
            # FINAL PHYSICAL-OBJECT DEDUPLICATION
            # ---------------------------------------------------------
            # A tracker can still temporarily retain a MiDaS track after the
            # association step, especially when the MiDaS box changes shape.
            # Never allow two final slots to represent the same physical
            # object.  If there is a conflict, YOLO always wins.
            deduped = []
            for candidate in sorted(final_obstacles, key=lambda tr: tr.smoothed_dist):
                duplicate_idx = None
                for idx, kept in enumerate(deduped):
                    if kept.box is not None and candidate.box is not None and boxes_same_object(candidate.box, kept.box):
                        duplicate_idx = idx
                        break
                if duplicate_idx is None:
                    deduped.append(candidate)
                    continue

                kept = deduped[duplicate_idx]
                kept_is_yolo = getattr(kept, "source", "YOLO") == "YOLO"
                cand_is_yolo = getattr(candidate, "source", "YOLO") == "YOLO"

                # YOLO has absolute priority for the displayed physical box.
                # If both are MiDaS, retain the one with the larger geometry.
                if cand_is_yolo and not kept_is_yolo:
                    deduped[duplicate_idx] = candidate
                elif cand_is_yolo == kept_is_yolo and candidate.box is not None and kept.box is not None:
                    carea = max(1, (candidate.box[2]-candidate.box[0]) * (candidate.box[3]-candidate.box[1]))
                    karea = max(1, (kept.box[2]-kept.box[0]) * (kept.box[3]-kept.box[1]))
                    if carea > karea * 1.10:
                        deduped[duplicate_idx] = candidate

            final_obstacles = sorted(
                deduped,
                key=lambda tr: tr.smoothed_dist,
            )[:MAX_TRACKS]

            # ---------------------------------------------------------
            # Known-object display
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            if SHOW_ALL_DETECTIONS:
                for (
                    cls_name_raw,
                    box_px,
                    confidence,
                ) in all_boxes:
                    cv2.rectangle(
                        frame,
                        box_px[:2],
                        box_px[2:],
                        (80, 80, 80),
                        1,
                    )


            # ---------------------------------------------------------
            # DUAL HC-SR04 FUSION / DIRECTION
            # ---------------------------------------------------------
            # FRONT sensor:
            #   - supplies authoritative metric distance
            #   - find the visual bbox (YOLO or confirmed MiDaS) whose
            #     estimated distance is closest to that reading
            #   - if a match exists, use that bbox's horizontal zone:
            #       FRONT-LEFT / FRONT / FRONT-RIGHT
            #   - if no match exists, classify the ultrasonic obstacle FRONT
            #
            # REAR sensor:
            #   - independent metric safety channel
            #   - always classified BACK (no camera matching)
            # ---------------------------------------------------------
            (
                ultrasonic_distance_cm,
                ultrasonic_closing_speed,
                rear_ultrasonic_distance_cm,
                rear_ultrasonic_closing_speed,
            ) = (
                ultrasonic.get_latest_pair_with_speed(
                    max_age_s=ULTRASONIC_MAX_AGE_S
                )
                if ULTRASONIC_FUSION_ENABLED
                else (None, 0.0, None, 0.0)
            )

            ultrasonic_track = None
            rear_ultrasonic_track = None
            front_ultrasonic_match = None
            ultrasonic_direction = None

            def _apply_ultrasonic_measurement(track, distance_cm, closing_speed, label):
                if distance_cm is None:
                    return track

                distance_m = float(
                    np.clip(distance_cm / 100.0, 0.02, MAX_RANGE)
                )

                created_virtual = track is None
                if created_virtual:
                    track = Track(
                        f"{label} ultrasonic obstacle",
                        (w / 2.0, h * 0.70),
                        distance_m,
                        now,
                        box=None,
                        frame_area=w * h,
                    )

                if not getattr(track, "ultrasonic_active", False):
                    track.history = []
                    track.distance_history.clear()
                    track.filtered_closing_speed = 0.0
                    track.ultrasonic_active = True

                track.smoothed_dist = distance_m
                track.distance_history.append(distance_m)
                track.history.append((now, distance_m))
                if len(track.history) > SPEED_WINDOW:
                    track.history.pop(0)
                track.filtered_closing_speed = closing_speed
                track.last_seen = now
                track.source = "HC-SR04"
                if created_virtual:
                    track.box = None
                    track.centroid = (w / 2.0, h * 0.70)
                    track.cls_name = f"{label.lower()} ultrasonic obstacle"
                track.last_yolo_distance = distance_m
                track.last_midas_ratio = 1.0
                track.last_midas_correction = 1.0
                track.last_midas_strength = 0.0
                return track

            # ---------- FRONT HC-SR04 ----------
            if ultrasonic_distance_cm is not None:
                front_m = float(ultrasonic_distance_cm / 100.0)
                visual_candidates = [
                    tr for tr in final_obstacles
                    if tr.box is not None
                ]

                if visual_candidates:
                    # Match by metric distance, not by image center. This lets
                    # the bbox determine left/front/right after the ultrasonic
                    # sensor identifies which visual obstacle it corresponds to.
                    front_ultrasonic_match = min(
                        visual_candidates,
                        key=lambda tr: abs(float(tr.smoothed_dist) - front_m),
                    )
                    match_error = abs(
                        float(front_ultrasonic_match.smoothed_dist) - front_m
                    )
                    if match_error <= max(
                        ULTRASONIC_BBOX_MATCH_TOL_M,
                        0.25 * max(front_m, 1.0),
                    ):
                        ultrasonic_track = front_ultrasonic_match
                        ultrasonic_direction = "front"
                        box_center_x = (
                            float(front_ultrasonic_match.box[0])
                            + float(front_ultrasonic_match.box[2])
                        ) / 2.0
                        frac = box_center_x / max(float(w), 1.0)
                        if frac < SPEECH_LEFT_ZONE_FRAC:
                            ultrasonic_direction = "front-left"
                        elif frac > SPEECH_RIGHT_ZONE_FRAC:
                            ultrasonic_direction = "front-right"
                    else:
                        front_ultrasonic_match = None

                if ultrasonic_track is None:
                    ultrasonic_track = ultrasonic_virtual_track
                    if ultrasonic_track is None:
                        ultrasonic_track = Track(
                            "front ultrasonic obstacle",
                            (w / 2.0, h * 0.70),
                            front_m,
                            now,
                            box=None,
                            frame_area=w * h,
                        )
                        ultrasonic_virtual_track = ultrasonic_track
                    ultrasonic_direction = "front"

                ultrasonic_track = _apply_ultrasonic_measurement(
                    ultrasonic_track,
                    ultrasonic_distance_cm,
                    ultrasonic_closing_speed,
                    "Front",
                )

                # Preserve the matched visual bbox/semantic identity when the
                # front sensor corresponds to YOLO/MiDaS. The ultrasonic
                # distance remains authoritative.
                if front_ultrasonic_match is not None:
                    ultrasonic_track = front_ultrasonic_match
                    ultrasonic_track.smoothed_dist = front_m
                    ultrasonic_track.filtered_closing_speed = ultrasonic_closing_speed
                    ultrasonic_track.last_seen = now

            # ---------- REAR HC-SR04 ----------
            if rear_ultrasonic_distance_cm is not None:
                rear_ultrasonic_track = _apply_ultrasonic_measurement(
                    rear_ultrasonic_virtual_track,
                    rear_ultrasonic_distance_cm,
                    rear_ultrasonic_closing_speed,
                    "Rear",
                )
                rear_ultrasonic_virtual_track = rear_ultrasonic_track

            # ---------------------------------------------------------
            # LONG-RANGE VISION-ASSISTED ULTRASONIC FAILSAFE
            # ---------------------------------------------------------
            # HC-SR04 remains authoritative at close/normal range. Beyond the
            # threshold, a background return is possible. Vision can override
            # only when a forward-axis obstacle is meaningfully closer, has HIGH
            # confidence, and remains consistent across fresh updates.
            # Raw MiDaS depth is NEVER compared directly with metres.
            # ---------------------------------------------------------
            vision_override_track = None
            vision_override_reason = "inactive"
            vision_override_score = 0.0
            vision_override_level = "LOW"
            vision_override_details = "no candidate"

            if ultrasonic_distance_cm is not None:
                ultrasonic_m = float(ultrasonic_distance_cm / 100.0)

                if ultrasonic_m >= ULTRASONIC_VISION_OVERRIDE_START_M:
                    center_limit = w * ULTRASONIC_VISION_CENTER_TOL_FRAC
                    vision_candidates = []

                    for tr in final_obstacles:
                        if tr.box is None:
                            continue

                        x1, _, x2, _ = tr.box
                        box_center_x = (x1 + x2) / 2.0
                        center_error = abs(box_center_x - (w / 2.0))
                        if center_error > center_limit:
                            continue

                        if getattr(tr, "source", "YOLO") == "YOLO":
                            vision_dist = float(
                                getattr(tr, "vision_smoothed_dist", tr.last_yolo_distance)
                            )
                        else:
                            vision_dist = float(tr.smoothed_dist)

                        if not np.isfinite(vision_dist):
                            continue

                        score, level, details = vision_confidence(tr)

                        if (
                            vision_dist < (ultrasonic_m - ULTRASONIC_VISION_OVERRIDE_MARGIN_M)
                            and level == "HIGH"
                        ):
                            vision_candidates.append(
                                (vision_dist, center_error, score, level, details, tr)
                            )

                    if vision_candidates:
                        _, _, score, level, details, candidate = min(
                            vision_candidates,
                            key=lambda item: (item[0], item[1]),
                        )

                        candidate_key = (
                            getattr(candidate, "source", "YOLO"),
                            getattr(candidate, "cls_name", "unknown"),
                            getattr(candidate, "id", id(candidate)),
                        )
                        fresh_vision_update = bool(fresh_yolo or depth_result is not None)

                        if candidate_key != ultrasonic_vision_candidate_key:
                            ultrasonic_vision_candidate_key = candidate_key
                            ultrasonic_vision_candidate_start = now
                            ultrasonic_vision_candidate_updates = (
                                1 if fresh_vision_update else 0
                            )
                            ultrasonic_vision_override_active = False
                        elif fresh_vision_update:
                            ultrasonic_vision_candidate_updates += 1

                        candidate_age = (
                            now - ultrasonic_vision_candidate_start
                            if ultrasonic_vision_candidate_start is not None
                            else 0.0
                        )
                        vision_override_score = score
                        vision_override_level = level
                        vision_override_details = details

                        if (
                            candidate_age >= ULTRASONIC_VISION_CONFIRM_S
                            and ultrasonic_vision_candidate_updates >= ULTRASONIC_VISION_MIN_UPDATES
                        ):
                            ultrasonic_vision_override_active = True
                            ultrasonic_vision_override_track = candidate
                            vision_override_track = candidate
                            vision_override_reason = (
                                f"stable vision {candidate.vision_smoothed_dist:.2f}m "
                                f"< ultrasonic {ultrasonic_m:.2f}m | {level} ({score:.2f})"
                            )
                    else:
                        # No sufficiently trustworthy visual override. Keep
                        # HC-SR04 authoritative and expose the reason in UI.
                        ultrasonic_vision_candidate_key = None
                        ultrasonic_vision_candidate_start = None
                        ultrasonic_vision_candidate_updates = 0
                        ultrasonic_vision_override_active = False
                        ultrasonic_vision_override_track = None

                        diagnostics = []
                        for tr in final_obstacles:
                            if tr.box is None:
                                continue
                            x1, _, x2, _ = tr.box
                            if abs(((x1 + x2) / 2.0) - (w / 2.0)) <= center_limit:
                                _score, _level, _details = vision_confidence(tr)
                                diagnostics.append((_score, _level, _details))
                        if diagnostics:
                            _, vision_override_level, vision_override_details = max(
                                diagnostics, key=lambda item: item[0]
                            )
                else:
                    ultrasonic_vision_candidate_key = None
                    ultrasonic_vision_candidate_start = None
                    ultrasonic_vision_candidate_updates = 0
                    ultrasonic_vision_override_active = False
                    ultrasonic_vision_override_track = None
            else:
                ultrasonic_vision_candidate_key = None
                ultrasonic_vision_candidate_start = None
                ultrasonic_vision_candidate_updates = 0
                ultrasonic_vision_override_active = False
                ultrasonic_vision_override_track = None

            # Unified obstacle selection.
            # A confirmed long-range vision override remains first priority.
            # Otherwise choose the most urgent fresh ultrasonic channel (front
            # or rear) by TTC. If neither ultrasonic channel is fresh, fall back
            # to the normal visual TTC selection.
            ultrasonic_candidates = [
                tr for tr in (ultrasonic_track, rear_ultrasonic_track)
                if tr is not None
            ]
            safety_candidates = list(ultrasonic_candidates)
            if vision_override_track is not None:
                safety_candidates.append(vision_override_track)

            if safety_candidates:
                # Let the most urgent fresh safety channel win. This prevents a
                # rear obstacle from being hidden simply because a long-range
                # front vision override also happens to be active.
                selected = min(safety_candidates, key=lambda t: t.ttc())
            else:
                selected = (
                    min(final_obstacles, key=lambda t: t.ttc())
                    if final_obstacles else None
                )

            # Direction is attached to the ultrasonic source that selected the
            # current safety obstacle. Rear is always BACK. Front uses the
            # matched visual bbox zone, or FRONT when no bbox matches.
            if selected is rear_ultrasonic_track:
                selected_direction = "back"
            elif selected is ultrasonic_track:
                selected_direction = ultrasonic_direction or "front"
            else:
                selected_direction = None

            for tr in final_obstacles:
                if tr.box is None:
                    continue

                is_unknown = getattr(tr, "source", "YOLO") == "MiDaS"
                color = (0, 0, 255) if tr is selected else ((0, 255, 255) if is_unknown else (150, 150, 150))

                cv2.rectangle(frame, tr.box[:2], tr.box[2:], color, 3)

            if selected is not None:
                if (
                    ultrasonic_vision_override_active
                    and vision_override_track is selected
                ):
                    dist = float(selected.vision_smoothed_dist)
                    closing_speed = float(selected.vision_closing_speed())
                else:
                    dist = selected.smoothed_dist
                    closing_speed = selected.closing_speed()
                cls_name = selected.cls_name
                selected_is_unknown = (
                    selected is not None
                    and getattr(selected, "source", "YOLO") == "MiDaS"
                )
            else:
                dist = MAX_RANGE
                closing_speed = 0.0
                cls_name = "none"
                selected_is_unknown = False

            # ---------------------------------------------------------
            # Existing GRU / proximity path — unchanged except that `dist`
            # is now the fused/aggregated metric estimate.
            # ---------------------------------------------------------
            yolo_anchor_dist = (
                current_yolo_anchor_distance(
                    selected,
                    w * h,
                )
                if selected is not None
                else MAX_RANGE
            )

            # ---------------------------------------------------------
            # Existing GRU / proximity path — unchanged
            # ---------------------------------------------------------
            # Preserve the real YOLO semantic label everywhere; only collapse
            # it at the boundary to the legacy GRU's six-class feature.
            gru_cls_name = (
                cls_name
                if cls_name == "curb"
                else GRU_CLASS_MAP.get(cls_name, "none")
            )
            cls_idx = FEATURE_CLASSES.index(
                gru_cls_name
            )

            feature_vec = [
                dist / MAX_RANGE,
                closing_speed,
                cls_idx / len(FEATURE_CLASSES),
                AGENT_SPEED_ASSUMED,
            ]

            feature_buffer.append(
                feature_vec
            )

            frame_counter += 1

            if (
                len(feature_buffer)
                >= GRU_SEQ_LEN
                and
                frame_counter
                % GRU_INFERENCE_EVERY_N_FRAMES
                == 0
            ):
                live_risk_target = predict_live_risk(
                    gru_model,
                    feature_buffer,
                )

            # Low-pass the neural risk display as well. The GRU itself is
            # unchanged; this only prevents one noisy input window from
            # making the displayed risk jump several buckets in one frame.
            live_risk = (
                RISK_EMA_ALPHA * live_risk_target
                + (1.0 - RISK_EMA_ALPHA) * live_risk
            )

            proximity_level, proximity_reason = (
                proximity_override(
                    selected,
                    h,
                    distance_override=dist,
                )
            )

            final_risk, final_bucket = (
                combine_risk(
                    live_risk,
                    proximity_level,
                )
            )

            # ---------------------------------------------------------
            # MiDaS unknown-obstacle safety layer
            # ---------------------------------------------------------
            selected_unknown_candidates = []
            for tr in final_obstacles:
                if getattr(tr, "source", "YOLO") == "MiDaS":
                    selected_unknown_candidates.append({
                        "zone": "FINAL",
                        "box": (
                            tr.box[0], tr.box[1],
                            tr.box[2] - tr.box[0],
                            tr.box[3] - tr.box[1],
                        ),
                        "score": 1.0,
                        "depth_level": 1.0,
                        "confirmed": True,
                    })

            (
                unknown_risk_floor,
                unknown_level,
                unknown_zone,
                unknown_reason,
            ) = unknown_obstacle_risk(selected_unknown_candidates, h)

            if unknown_risk_floor > final_risk:
                final_risk = unknown_risk_floor
                final_bucket = risk_bucket(final_risk)

            # If the selected obstacle is the MiDaS unknown obstacle, the
            # GRU/proximity path above has ALREADY consumed its distance,
            # closing speed and class proxy. Keep the safety floor as an
            # additional guard, but do not overwrite those measurements.
            # ---------------------------------------------------------
            # AUDIO / LLM NAVIGATION LAYER
            # ---------------------------------------------------------
            # CRITICAL hazards bypass the network and speak immediately.
            # Other risk levels use event-triggered Groq for concise contextual
            # guidance; the deterministic sentence is used if Groq is unavailable
            # or fails.
            # SpeechManager.request() already handles forced_direction.
            # _direction() does not accept that argument.
            speech_direction = (
                selected_direction
                if selected_direction
                else speech._direction(
                    selected, w, unknown_zone=unknown_zone
                )
            )
            current_ttc = (
                selected.vision_ttc()
                if (
                    selected is not None
                    and ultrasonic_vision_override_active
                    and vision_override_track is selected
                )
                else (selected.ttc() if selected is not None else TTC_SAFE_VALUE)
            )
            if final_bucket == "CRITICAL":
                speech.request(
                    final_bucket,
                    selected,
                    w,
                    unknown_zone=unknown_zone,
                    forced_direction=selected_direction,
                )
            elif selected is not None:
                groq_navigation.request(
                    final_bucket,
                    cls_name if cls_name != "none" else "obstacle",
                    dist,
                    speech_direction,
                    closing_speed,
                    current_ttc,
                )
            else:
                # No selected obstacle: end the current Groq event episode.
                groq_navigation.reset()
                speech.request(
                    final_bucket,
                    selected,
                    w,
                    unknown_zone=unknown_zone,
                    forced_direction=selected_direction,
                )

            selected_source = (
                "VISION OVERRIDE"
                if ultrasonic_vision_override_active
                and vision_override_track is selected
                else (
                    "HC-SR04 FRONT" if ultrasonic_track is selected
                    else (
                        "HC-SR04 REAR" if rear_ultrasonic_track is selected
                        else (
                            "MiDaS UNKNOWN" if selected_is_unknown
                            else ("YOLO" if selected is not None else "NONE")
                        )
                    )
                )
            )

            # ---------------------------------------------------------
            # CLEAN THREE-PANEL DISPLAY
            # ---------------------------------------------------------
            # No telemetry text is drawn over the camera/depth/mask images.
            # Each panel gets its own dedicated information strip underneath.

            def _draw_status_panel(width, height, lines, bg=(28, 28, 28), fg=(235, 235, 235), title=None):
                strip = np.full((height, width, 3), bg, dtype=np.uint8)
                y = 25
                if title:
                    cv2.putText(
                        strip, title, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        fg, 2, cv2.LINE_AA,
                    )
                    y += 28
                for text, color, scale, thickness in lines:
                    if y >= height - 8:
                        break
                    cv2.putText(
                        strip, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, thickness, cv2.LINE_AA,
                    )
                    y += 24
                return strip

            # Keep the displayed obstacle boxes, but put ALL descriptive text
            # below the corresponding image instead of over it.
            if selected is not None:
                if (
                    ultrasonic_vision_override_active
                    and vision_override_track is selected
                ):
                    dist = float(selected.vision_smoothed_dist)
                    closing_speed = float(selected.vision_closing_speed())
                else:
                    dist = selected.smoothed_dist
                    closing_speed = selected.closing_speed()
                cls_name = selected.cls_name
                selected_is_unknown = (
                    getattr(selected, "source", "YOLO") == "MiDaS"
                )
            else:
                dist = MAX_RANGE
                closing_speed = 0.0
                cls_name = "none"
                selected_is_unknown = False

            selected_source = (
                "VISION OVERRIDE"
                if ultrasonic_vision_override_active
                and vision_override_track is selected
                else (
                    "HC-SR04 FRONT" if ultrasonic_track is selected
                    else (
                        "HC-SR04 REAR" if rear_ultrasonic_track is selected
                        else (
                            "MiDaS UNKNOWN" if selected_is_unknown
                            else ("YOLO" if selected is not None else "NONE")
                        )
                    )
                )
            )
            selected_ttc = (
                selected.vision_ttc()
                if (
                    selected is not None
                    and ultrasonic_vision_override_active
                    and vision_override_track is selected
                )
                else (selected.ttc() if selected is not None else TTC_SAFE_VALUE)
            )

            # ---------------- Camera panel ----------------
            camera_lines = [
                (
                    f"SELECTED: {selected_source} {cls_name} | distance {dist:.2f} m",
                    (0, 90, 255), 0.52, 2,
                ),
                (
                    (
                        f"Ultrasonic: {ultrasonic_distance_cm:.1f} cm (smoothed)"
                        if ultrasonic_distance_cm is not None
                        else "Ultrasonic: -- cm (no recent reading)"
                    ),
                    (0, 255, 255), 0.50, 2,
                ),
                (
                    (
                        f"Rear ultrasonic: {rear_ultrasonic_distance_cm:.1f} cm (smoothed)"
                        if rear_ultrasonic_distance_cm is not None
                        else "Rear ultrasonic: -- cm (no recent reading)"
                    ),
                    (0, 255, 255), 0.50, 2,
                ),
                (
                    f"Direction: {selected_direction or 'visual'}",
                    (255, 255, 0), 0.48, 2,
                ),
                (
                    (
                        f"Vision override: ACTIVE | {vision_override_reason}"
                        if ultrasonic_vision_override_active
                        else (
                            f"Vision check: {ultrasonic_vision_candidate_updates}/"
                            f"{ULTRASONIC_VISION_MIN_UPDATES} updates"
                            if ultrasonic_distance_cm is not None
                            and ultrasonic_distance_cm / 100.0
                            >= ULTRASONIC_VISION_OVERRIDE_START_M
                            else "Vision override: inactive (HC-SR04 authoritative)"
                        )
                    ),
                    (255, 255, 0), 0.44, 1,
                ),
                (
                    f"Vision confidence: {vision_override_level} | {vision_override_details}",
                    (255, 255, 0), 0.44, 1,
                ),
                (
                    f"Closing: {closing_speed:+.2f} m/s | TTC: {selected_ttc:.1f} s",
                    (0, 90, 255), 0.50, 2,
                ),
                (
                    f"GRU: {live_risk:.3f} | RISK: {risk_bucket(live_risk)} | buffer {len(feature_buffer)}/{GRU_SEQ_LEN}",
                    (0, 90, 255), 0.50, 2,
                ),
                (
                    f"GRU input: d={dist:.2f} m  close={closing_speed:+.2f}  YOLO={cls_name}  GRU-class={gru_cls_name}",
                    (235, 235, 235), 0.46, 1,
                ),
                (
                    (
                        (
                            "Distance source: VISION OVERRIDE (long-range)"
                            if ultrasonic_vision_override_active
                            and vision_override_track is selected
                            else (
                                "Distance source: HC-SR04 FRONT (matched visual bbox)"
                                if ultrasonic_track is selected and front_ultrasonic_match is not None
                                else (
                                    "Distance source: HC-SR04 REAR"
                                    if rear_ultrasonic_track is selected
                                    else "Distance source: HC-SR04 FRONT (independent obstacle)"
                                )
                            )
                        )
                        if ultrasonic_track is selected
                        else "Distance source: YOLO + MiDaS"
                    ),
                    (255, 255, 0), 0.44, 1,
                ),
                (
                    f"Proximity: {proximity_level} | Final: {final_risk:.3f} {final_bucket}",
                    (0, 90, 255), 0.50, 2,
                ),
                (
                    f"Reason: {proximity_reason} | tracks={len(final_obstacles)}",
                    (235, 235, 235), 0.46, 1,
                ),
                (
                    f"FPS: {fps:.1f} | MiDaS grid: native 256x256 | async",
                    (235, 235, 235), 0.46, 1,
                ),
            ]

            # Unknown status is kept in the camera strip because it is a
            # navigation-system decision, while the raw mask itself remains
            # text-free.
            confirmed = [
                c for c in unknown_candidates
                if c.get("stable_confirmed", False)
            ]
            if confirmed:
                best_unknown = max(
                    confirmed,
                    key=lambda c: c.get("score", 0.0),
                )
                unknown_summary = (
                    f"Unknown obstacle: {best_unknown['zone']} "
                    f"{best_unknown.get('stable_hits', 0)}/{STABILITY_HITS_REQUIRED} "
                    f"| risk {unknown_level}"
                )
            elif unknown_candidates:
                best_unknown = unknown_candidates[0]
                unknown_summary = (
                    f"Unknown candidate: {best_unknown['zone']} "
                    f"{best_unknown.get('stable_hits', 0)}/{STABILITY_HITS_REQUIRED}"
                )
            else:
                unknown_summary = "Unknown obstacle: none"

            camera_lines.append(
                (unknown_summary, (0, 255, 255), 0.46, 1)
            )

            # ---------------- MiDaS panel ----------------
            metric_status = (
                f"Metric status: CALIBRATED ({len(midas_calibrator.samples)} samples)"
                if midas_calibrator.ready
                else f"Metric status: UNCALIBRATED ({len(midas_calibrator.samples)}/3 samples)"
            )
            yolo_anchor_text = (
                f"YOLO anchor: {yolo_anchor_dist:.2f} m | final: {dist:.2f} m"
                if selected is not None
                else "YOLO anchor: -- | final: --"
            )
            midas_corr = (
                float(getattr(selected, "last_midas_correction", 1.0))
                if selected is not None else 1.0
            )
            midas_lines = [
                ("MiDaS Small - RELATIVE DEPTH", (255, 255, 255), 0.58, 2),
                (metric_status, (0, 255, 255), 0.46, 1),
                (yolo_anchor_text, (255, 255, 0), 0.46, 1),
                (f"MiDaS correction factor: {midas_corr:.2f}", (255, 255, 0), 0.46, 1),
                (
                    f"YOLO confidence: {getattr(selected, 'last_yolo_confidence', 0.0):.2f} | "
                    f"vision score: {vision_override_score:.2f}",
                    (255, 255, 0), 0.44, 1,
                ),
                ("Depth values provide relative correction evidence only.", (235, 235, 235), 0.44, 1),
                ("Final bbox geometry: fused physical obstacles only.", (235, 235, 235), 0.44, 1),
            ]

            # ---------------- Mask panel ----------------
            mask_lines = [
                ("DEPTH PROTRUSION MASK", (255, 255, 255), 0.58, 2),
                (f"Confirmed candidates: {len(confirmed)}", (0, 255, 255), 0.46, 1),
                (f"Raw candidates: {len(raw_unknown_candidates)}", (235, 235, 235), 0.46, 1),
                (f"Stable requirement: {STABILITY_HITS_REQUIRED}/{STABILITY_WINDOW} frames", (235, 235, 235), 0.46, 1),
                ("White regions = MiDaS protrusion evidence.", (235, 235, 235), 0.44, 1),
                ("Only confirmed candidates enter navigation fusion.", (235, 235, 235), 0.44, 1),
            ]

            # Draw only bounding boxes on the image itself.
            for tr in final_obstacles:
                if tr.box is None:
                    continue
                is_unknown = getattr(tr, "source", "YOLO") == "MiDaS"
                color = (
                    (0, 0, 255) if tr is selected
                    else ((0, 255, 255) if is_unknown else (150, 150, 150))
                )
                cv2.rectangle(
                    frame,
                    tr.box[:2],
                    tr.box[2:],
                    color,
                    3,
                )

            if SHOW_ALL_DETECTIONS:
                for cls_name_raw, box_px, confidence in all_boxes:
                    cv2.rectangle(
                        frame,
                        box_px[:2],
                        box_px[2:],
                        (80, 80, 80),
                        1,
                    )

            # Build depth image and mask image without text overlays.
            if have_depth_result and latest_midas_depth is not None:
                try:
                    depth_vis = midas_visual(
                        latest_midas_depth,
                        w,
                        h,
                    )

                    # Same final physical obstacles, no labels.
                    for tr in final_obstacles:
                        if tr.box is None:
                            continue
                        is_unknown = getattr(tr, "source", "YOLO") == "MiDaS"
                        color = (
                            (0, 0, 255) if tr is selected
                            else ((0, 255, 255) if is_unknown else (150, 150, 150))
                        )
                        x1, y1, x2, y2 = map(int, tr.box)
                        cv2.rectangle(
                            depth_vis,
                            (x1, y1), (x2, y2),
                            color, 3,
                        )

                    if latest_unknown_mask is not None and latest_unknown_mask.size > 0:
                        mask_vis = cv2.cvtColor(
                            latest_unknown_mask,
                            cv2.COLOR_GRAY2BGR,
                        )
                    else:
                        mask_vis = np.zeros(
                            (h, w, 3),
                            dtype=np.uint8,
                        )

                except Exception as exc:
                    print(f"MiDaS display warning: {exc}")
                    depth_vis = np.zeros_like(frame)
                    mask_vis = np.zeros_like(frame)
            else:
                depth_vis = np.zeros_like(frame)
                mask_vis = np.zeros_like(frame)

            status_h = 215
            camera_status = _draw_status_panel(
                w, status_h, camera_lines,
                bg=(22, 22, 22),
                fg=(235, 235, 235),
            )
            midas_status = _draw_status_panel(
                w, status_h, midas_lines,
                bg=(22, 22, 22),
                fg=(235, 235, 235),
            )
            mask_status = _draw_status_panel(
                w, status_h, mask_lines,
                bg=(22, 22, 22),
                fg=(235, 235, 235),
            )

            camera_panel = np.vstack((frame, camera_status))
            midas_panel = np.vstack((depth_vis, midas_status))
            mask_panel = np.vstack((mask_vis, mask_status))

            combined = np.hstack(
                (
                    camera_panel,
                    midas_panel,
                    mask_panel,
                )
            )

            safe_imshow(
                "wearable-nav + LIVE GRU + MiDaS",
                combined,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if selected is None:
                    print("MiDaS calibration: no selected obstacle.")
                elif latest_midas_raw_depth is None or selected.box is None:
                    print("MiDaS calibration: no raw MiDaS depth available.")
                else:
                    try:
                        raw_value = raw_midas_depth_for_box(
                            latest_midas_raw_depth,
                            selected.box,
                            frame.shape,
                        )
                        entered = input(
                            "\nTRUE distance for the selected object (metres): "
                        ).strip()
                        true_dist = float(entered)
                        if midas_calibrator.add(raw_value, true_dist):
                            print(
                                f"Captured MiDaS calibration: raw={raw_value:.5f} "
                                f"-> {true_dist:.3f}m"
                            )
                        else:
                            print("Invalid MiDaS calibration point; skipped.")
                    except ValueError:
                        print("Invalid distance; skipped.")
                    except Exception as exc:
                        print(f"MiDaS calibration warning: {exc}")

            elif key == ord("x"):
                midas_calibrator.clear()
                print("MiDaS metric calibration cleared.")

            elif key == ord("q"):
                break

    finally:
        midas_worker.stop()
        reader.release()
        ultrasonic.stop()
        groq_navigation.stop()
        speech.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
