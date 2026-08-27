[README (2).md](https://github.com/user-attachments/files/31516845/README.2.md)
# 🤖✋ Sign Language Communication System

**A Low-Cost Arduino-Based Robotic Hand for Real-Time ASL Sign Language Generation with Integrated AI-Powered Camera Recognition**

A complete **bidirectional** assistive communication system that helps deaf/speech-impaired and hearing people communicate — in both directions — using a 3D-printed, Arduino-controlled robotic hand and an AI-powered Python desktop app.

> Authors: Abhimanyu Sangwan, K. Rajesh, Love Sharma — Dept. of ECE / AI of Things, School of Engineering & Technology, Manav Rachna International Institute of Research and Studies, Faridabad, India

---

## 📖 Overview

Over 466 million people worldwide are deaf or hard of hearing, and around 70 million rely on sign language as their primary means of communication. This project closes the gap between sign language and spoken/written English **from both directions**:

- **Deaf → Hearing:** A person signs the ASL alphabet in front of a laptop camera. The system decodes each letter, reconstructs the spelled-out word stream into a natural English sentence using an AI language model, and can optionally spell that sentence back out loud through the robotic hand for a hearing person to read.
- **Hearing → Deaf:** A hearing person types a sentence into the app. The robotic hand fingerspells it letter-by-letter in ASL for a deaf person to read.

The system costs **~USD 50** in components, runs entirely on consumer hardware (a laptop + Arduino Uno), and needs no cloud-mandatory services, proprietary hardware, or wearable sensors.



---

## ✨ Key Features

- 🖐️ **5-finger 3D-printed robotic hand** — Dyneema-tendon driven, actuated by 5 MG90S metal-gear micro servos
- 👀 **Real-time ASL recognition** — MediaPipe 21-point hand-landmark geometry, no ML training data required
- 🧠 **AI sentence reconstruction** — Claude API cleans up imperfect fingerspelling into natural English
- ⌨️ **Type → Hand** — type any sentence and watch the robotic hand fingerspell it back
- 📚 **Custom gesture library** — record or import your own gestures/phrases, matched via Dynamic Time Warping (DTW)
- 🎛️ **Full Tkinter GUI** — 4 tabs: Camera Decode, Type → Hand, Gesture Library, Gesture Match
- 🔌 **Simple serial protocol** — Arduino ⇄ laptop over USB at 9600 baud, one ASCII character per letter
- 💰 **Low cost, no specialist hardware** — Arduino Uno + breadboard + 5 servos + 3D-printed parts

---

## 🎥 Software Interface (Screenshots)

| Camera Decode — live landmark tracking | Type → Hand |
|---|---|


| Camera Decode — idle / camera not started | Compact window layout |
|---|---|


The GUI has four tabs — **Camera Decode**, **Type → Hand**, **Gesture Library**, and **Gesture Match** — plus Arduino COM-port selection, camera index/URL selection (including phone IP-camera support), a live confidence bar, hold timer, spelling buffer, and an AI-reconstructed sentence box.

---

## 🦾 Hardware in Action

| Live demo | Servo assembly inside palm | ASL alphabet reference |
|---|---|---|


📹 Full build demo: [LinkedIn post](https://www.linkedin.com/posts/rajesh-kotipalli-b82602288_robotics-arduino-servomotor-activity-7447674706284974080-w6u1)

---

## 🏗️ System Architecture

```
                 ┌───────────────────────────── Laptop (Python) ─────────────────────────────┐
                 │                                                                            │
   Webcam ───►   │  asl_recognizer.py  ──►  main_interface.py (Tkinter GUI, 4 tabs)            │
  (or phone IP   │  (MediaPipe landmark        │            │            │                    │
   camera)       │   geometry classifier)      │            │            │                    │
                 │                              ▼            ▼            ▼                    │
                 │                    sentence_builder.py  gesture_library.py                  │
                 │                    (Claude AI sentence   (record / DTW match /               │
                 │                     reconstruction)       JSON persistence)                  │
                 └───────────────────────────────────┬──────────────────────────────────────────┘
                                                       │ USB Serial @ 9600 baud
                                                       │ (single ASCII char per letter)
                                                       ▼
                 ┌────────────────────────── Arduino Uno (firmware) ─────────────────────────┐
                 │  26-letter gesture lookup table → smooth proportional 5-servo movement      │
                 │  Inputs: serial chars  +  26-button breadboard keypad (pull-down, debounced) │
                 └──────────────────────────────────────┬──────────────────────────────────────┘
                                                          │ PWM (5 channels)
                                                          ▼
                                     5 × MG90S micro servos → Dyneema tendons
                                          → 3D-printed robotic hand fingers
                    (powered by dedicated external 5V/3A supply, common GND with Arduino)
```

### Software modules

| Module | Role |
|---|---|
| `asl_recognizer.py` | Classifies the ASL letter from MediaPipe's 21-point hand skeleton using deterministic geometric rules (finger curl, thumb position, spread, etc.) — no training data needed. Includes a 12-frame rolling-majority confidence vote. |
| `main_interface.py` | The Tkinter GUI and control centre. Manages the serial link to Arduino, renders the camera feed, and hosts the four tabs. |
| `gesture_library.py` | Records/imports custom multi-frame gesture sequences, normalizes MediaPipe landmarks, and matches live gestures against the library using Dynamic Time Warping (DTW). |
| `sentence_builder.py` | Calls the Claude API to turn an imperfect letter stream (e.g. `HELO MY NAM IS AJESH`) into a natural sentence (`Hello, my name is Rajesh.`), with an offline fallback if the API is unavailable. |

---

## 🔧 Hardware Bill of Materials

| Component | Spec / Notes |
|---|---|
| Microcontroller | Arduino Uno (ATmega328P, 16 MHz, 5V logic) |
| Servos (×5) | MG90S metal-gear micro servo, 2.2 kg·cm stall torque @ 5V, 180° range |
| Hand chassis | 3D-printed PLA, 0.2 mm layer height, 30% gyroid infill, Dyneema-tendon fingers, ~142 g |
| Power supply | Dedicated external 5V / ≥3A supply, common GND with Arduino; 1000 µF capacitor across servo rail |
| Input keypad | 26 push-buttons on breadboard, pull-down config, 10 kΩ resistors, 20 ms software debounce |
| Camera | Any laptop webcam or phone camera over IP (e.g. DroidCam/IP Webcam URL) |
| Approx. total cost | ~USD 50 |

**Wiring summary**
- Servos → Arduino PWM pins 3, 5, 6, 9, 10 (thumb → little finger)
- Servo signal wires → Arduino digital pins; servo VCC/GND → external 5V supply
- External supply GND ↔ Arduino GND (common reference — required for clean PWM)
- Breadboard buttons → GPIO pins with 10 kΩ pull-down resistors to GND, other terminal to 5V

---

## 🚀 Getting Started

### 1. Hardware setup
1. 3D-print the hand (STL files in `hardware/`, PLA, 0.2 mm layers, 30% infill).
2. Mount the 5 MG90S servos in the palm and route Dyneema tendons through the finger guide channels.
3. Wire servos to Arduino pins 3, 5, 6, 9, 10; wire the external 5V/3A supply to the servo power rail with a common GND to the Arduino.
4. (Optional) Wire the 26-button breadboard keypad as a standalone fallback input.
5. Flash `firmware/robotic_hand.ino` to the Arduino Uno via Arduino IDE (v2.3+).

### 2. Software setup
```bash
# Clone the repository
git clone <your-repo-url>
cd sign-language-communication-system

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe pyserial anthropic

# Set your Claude API key for sentence reconstruction
export ANTHROPIC_API_KEY="your-api-key-here"    # Windows: set ANTHROPIC_API_KEY=your-api-key-here

# Launch the app
python main_interface.py
```

### 3. Using the app
1. Select the Arduino COM port and click **Connect**.
2. Select a camera index (or paste a phone IP-camera URL) and click **Test Camera**, then **Start Camera**.
3. **Camera Decode tab** — sign the ASL alphabet; hold each letter ~1.5 s to register it. Remove your hand for 2 s to end a word and trigger AI sentence reconstruction. Click **Send to Hand** to have the robotic hand spell the result back.
4. **Type → Hand tab** — type a sentence, choose a speed (Slow/Normal/Fast), and click **Send to Hand**.
5. **Gesture Library tab** — record or import custom gestures with a name and meaning.
6. **Gesture Match tab** — perform a custom gesture; the best DTW match is displayed and spelled out automatically.

---

## 📊 Results

| Metric | Result |
|---|---|
| Hardware response latency | 183 ms mean (SD 12 ms), max 220 ms — under the 300 ms human-perceptible threshold |
| Observer letter-recognition accuracy | 90.8% (SD 4.3%), n = 20 untrained participants |
| Mechanical reliability | 500 full-alphabet cycles (13,000 servo actuations), zero mechanical failures |
| Camera classifier accuracy (per-frame) | ~88% across all 26 letters |
| Camera classifier accuracy (at registration, after majority-vote filtering) | ~93% |
| Most confused letter pairs | M/N, R/U (also A/S, D/Z, U/V by geometric ambiguity) |

---

## ⚠️ Limitations

- Covers letter-by-letter fingerspelling, not whole-word ASL signs.
- Manual push-button input is slow for extended communication.
- The static geometric classifier does not adapt to individual signing styles.
- Camera recognition accuracy is sensitive to lighting conditions.

## 🔭 Future Work

- Smartphone BLE companion app to replace the push-button keyboard
- Voice-to-sign real-time translation via speech recognition
- Cable-driven differential mechanism for finer finger-joint control
- Indian Sign Language (ISL) support via an alternate angle lookup table
- Machine-vision pose verification with a secondary calibration camera

---

## 📁 Suggested Repository Structure

```
sign-language-communication-system/
├── firmware/
│   └── robotic_hand.ino          # Arduino firmware (Servo.h, gesture lookup table)
├── software/
│   ├── main_interface.py         # Tkinter GUI — entry point
│   ├── asl_recognizer.py         # MediaPipe geometric ASL classifier
│   ├── gesture_library.py        # Recording, DTW matching, JSON persistence
│   └── sentence_builder.py       # Claude API sentence reconstruction
├── hardware/
│   └── *.stl                     # 3D-printable hand parts (Fusion 360 source optional)
├── images/                       # Photos & screenshots used in this README
└── README.md
```

---

## 📚 References

1. World Health Organization (WHO). (2021). *Deafness and Hearing Loss.*
2. Sharma, P., & Verma, A. (2020). Smart glove for sign language recognition using flex sensors. *IJARCCE*, 9(3), 45–52.
3. Demir, M. U., et al. (2024). A novel MEMS and flex sensor-based hand gesture recognition and regenerating system using deep learning. *IEEE Sensors Journal*, 24(16), 26110–26123.
4. Matsuo, T., et al. (2003). A humanoid robotic hand performing sign language motions. *IEEE/RSJ IROS*, 1282–1287.
5. Gibet, S., et al. (1999). High-level specification and control of communication gestures: the GESSYCA system. *IEEE ICRA*, 726–731.
6. Nguyen, T. A. T., et al. (2021). An adaptive, affordable, open-source robotic hand for deaf and deaf-blind communication using tactile ASL. *IEEE EMBC*, 6045–6049.
7. Nguyen, T. A. T., et al. (2022). An adaptive, affordable, humanlike arm-hand system for deaf and deaf-blind ASL communication. *IEEE ROBIO*, 1012–1017.
8. Lach, L., et al. (2023). Signs of language: Embodied sign language fingerspelling acquisition from demonstrations for human-robot interaction. *IEEE RO-MAN*, 1178–1185.
9. Singh, R., & Patel, M. (2020). Servo motor controlled robotic hand using Arduino. *IRJET*, 7(6), 3301–3307.
10. Alabbad, D. A., et al. (2022). A robot-based Arabic sign language translating system. *IEEE CDMA*, 151–156.
11. Kumar, R. (2021). Design and development of robotic hand for sign language communication. *IJERT*, 8(5), 217–224.
12. MediaPipe Team, Google. (2023). *MediaPipe Hands.*
13. Arduino Project. (2024). *Arduino IDE v2.3 Documentation.*
14. Chandra, S., et al. (2022). Dynamic sign language translator. *IEEE ICCAR*, 345–349.

---

## 👥 Authors

- **Abhimanyu Sangwan** — Dept. of Electronics and Communication Engineering
- **K. Rajesh** — Dept. of Artificial Intelligence of Things
- **Love Sharma** — Dept. of Artificial Intelligence of Things

School of Engineering & Technology, Manav Rachna International Institute of Research and Studies, Faridabad, India

## 📄 License

Add your preferred license here (e.g. MIT, Apache 2.0).
