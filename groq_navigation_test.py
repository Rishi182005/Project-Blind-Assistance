from groq import Groq
import time

client = Groq()

prompt = """
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

Object: person
Distance: 3.0 m
Direction: ahead on the right
Closing speed: 0 m/s
TTC: unavailable
Risk: LOW
"""

start = time.perf_counter()

response = client.chat.completions.create(
    model="qwen/qwen3.6-27b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0,
    max_tokens=50,
    reasoning_effort="none"
)

elapsed = time.perf_counter() - start

print("\nGroq instruction:")
print(response.choices[0].message.content.strip())

print(f"\nResponse time: {elapsed:.3f} seconds")