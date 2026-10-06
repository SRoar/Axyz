# Axyz (Illumin)

**Independent paper-test taking for blind and low-vision students.**

Axyz guides a visually impaired student's pen to the right answer box on a handwritten exam, using audio, touch, and an overhead camera, so they can write their own answers without a human scribe.

Built at MHacks 2026. [Devpost](https://devpost.com/software/axyz)

## Why

Paper tests are still the default in most classrooms. For blind and low-vision students, that usually means a human scribe, which costs them independence and privacy and lets someone else decide what gets written down. Axyz gives students the same confidence a sighted student has: pick up a pen, find the box, write your own answer.

## How It Works

1. The student holds the pen still and hears the question read aloud.
2. As they move, left and right buzzers on the pen pulse and short spoken cues ("Move 2 inches down") steer the pen toward the answer box.
3. When the pen settles inside the box, a double tone confirms it's locked in.
4. While the student writes, the voice stays silent. Drifting past the margin triggers a harsh buzz.
5. Three taps on the pen means "I'm done." Gemini checks a photo of the box, then the system confirms "Answer recorded."

**Core rule:** the camera knows *where* the pen is, the accelerometer knows *what* the pen is doing, and the brain acts on the combination.

## Architecture

| Component | What it does | Tech |
|---|---|---|
| **Pen probe** | Classifies the pen as still, moving, writing, or lifted; detects 1/2/3 taps; drives left/right buzzers | Arduino 101, accelerometer, light sensor, buzzers (C++) |
| **Overhead camera** | Tracks the pen via markers and a four-corner homography to get pen-tip position in page coordinates; holds the last position briefly when the hand covers the pen | OpenCV, NumPy |
| **Brain** | State machine (idle, reading, navigating, writing, auditing) that never speaks while the student is writing | Python, Pydantic |
| **Voice** | Reads questions and navigation cues | ElevenLabs |
| **Auditor** | Checks the answer box for ink; falls back to a pixel-based check if the network is down or slow | Gemini 2.5 Flash (structured output) |
| **Live view** | Browser UI showing camera, voice waveform, buzzers, and pen state | HTML, CSS, JavaScript, GSAP, Server-Sent Events |

The camera maps image points to page points with a homography:

```
p_page = H · p_image
```

## Repository Structure

```
Axyz/
├── Depth_Comp/                 # Depth computation
├── data/                       # Data files
├── src/                        # Main source code
├── test_arduino_all_sensors/   # Arduino sketch to test all pen sensors
├── test_elevenlabs.py          # Tests the ElevenLabs connection
├── test_voice_loop.py          # Tests the voice loop
├── PLAN.md                     # Project plan
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
└── README.md
```

## Requirements

**Hardware**
- Arduino 101 with accelerometer, light sensor, and two buzzers (the pen probe)
- Overhead camera
- Printed exam page with corner markers
- Speakers or headphones

**Software**
- Python 3.x
- Arduino IDE
- API keys: [ElevenLabs](https://elevenlabs.io) and [Google Gemini](https://ai.google.dev)

## Setup

```bash
git clone https://github.com/SRoar/Axyz.git
cd Axyz

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then add your API keys
```

Flash the pen: open `test_arduino_all_sensors/` in the Arduino IDE, select the Arduino 101 and its port, upload, and check the Serial Monitor for sensor readings. The pen connects to the computer over serial (PySerial).

## Usage

Test the ElevenLabs voice:
```bash
python test_elevenlabs.py
```

Test the voice loop:
```bash
python test_voice_loop.py
```

Run the full system:
```bash
# TODO: replace with the real entry point
python src/main.py
```

## Design Notes

- **Silence is a feature.** The most important thing the system does is know when not to speak.
- **Graceful failure.** Network, camera, sensor, and voice can each fail; every component has a fallback or keyboard override.
- **Shared contract and fakes.** Each component has a fake driven by a scripted timeline, so hardware, vision, voice, and AI could be built and tested independently.

## Roadmap

- Detect the box edge directly with the pen's light sensor for a faster margin buzz
- Multi-page tests
- Handwriting transcription so answers can be reviewed digitally
- More languages

See [PLAN.md](PLAN.md) for more.

## Team

Aditya Kanginaya Madhuchandra, Yug Shah, Aditya Kumar, Shivaji Raj

## License

TODO: add a license.
