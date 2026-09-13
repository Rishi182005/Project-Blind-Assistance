from google import genai
import time

client = genai.Client()

def get_navigation_instruction(
    object_name,
    distance_m,
    direction,
    closing_speed,
    ttc,
    risk
):
    prompt = f"""
You are a wearable navigation assistant for a visually impaired user.

Give ONE short spoken instruction using ONLY the supplied facts.

Rules:
- Never invent an obstacle, distance, direction, movement, or safe escape route.
- Never tell the user to turn left, turn right, or step backward unless explicitly provided as a safe route.
- Always include distance and direction when available.
- If an object is approaching, state that it is approaching.
- CRITICAL: start with "Stop" and tell the user not to move forward.
- HIGH: start with "Stop" and give a cautious instruction.
- MEDIUM: start with "Caution" and tell the user to slow down.
- LOW: give brief awareness only.
- Maximum 18 words.

Object: {object_name}
Distance: {distance_m:.1f} m
Direction: {direction}
Closing speed: {closing_speed:.2f} m/s
TTC: {ttc}
Risk: {risk}
"""

    start = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    elapsed = time.perf_counter() - start

    return response.text.strip(), elapsed


if __name__ == "__main__":
    instruction, response_time = get_navigation_instruction(
        object_name="person",
        distance_m=1.2,
        direction="front-left",
        closing_speed=0.35,
        ttc=3.4,
        risk="HIGH"
    )

    print("\nGemini instruction:")
    print(instruction)

    print(f"\nResponse time: {response_time:.3f} seconds")