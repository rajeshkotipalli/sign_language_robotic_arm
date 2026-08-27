"""
main_interface.py
Complete Sign Language Communication System.

Mode 1 (Camera → Decode):
  - Camera sees ASL signs
  - Recognizes letters with improved 26-letter classifier
  - Buffers letters into words
  - Claude AI builds natural sentences from those words
  - Sentence displayed on screen (big, readable)
  - Optionally sent to robotic hand to spell back

Mode 2 (Type → Hand):
  - User types a word/sentence
  - Robotic hand spells it out in ASL letter by letter
"""

import tkinter as tk
from tkinter import ttk, font, scrolledtext
import threading
import serial
import serial.tools.list_ports
import time
import cv2
import mediapipe as mp

from asl_recognizer import classify_asl, classify_with_confidence
from sentence_builder import build_sentence, build_sentence_quick


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

HOLD_SECONDS      = 1.5     # seconds to hold a sign before it registers
WORD_GAP_SECONDS  = 2.0     # pause with no hand → end of current word
HISTORY_SIZE      = 12      # frames for confidence smoothing
MIN_CONFIDENCE    = 0.65    # min stability to register a letter


# ─────────────────────────────────────────────────────────────────────────────
#  Colours & fonts
# ─────────────────────────────────────────────────────────────────────────────

BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
ACCENT  = "#1f6feb"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
WHITE   = "#f0f6fc"
GRAY    = "#8b949e"
RED     = "#f85149"
CYAN    = "#79c0ff"


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────

class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Communication System")
        self.root.geometry("1100x800")
        self.root.configure(bg=BG)

        self.serial_conn    = None
        self.cam_running    = False
        self.cam_thread     = None

        # Recognition state
        self.letter_history   = []
        self.current_sign     = ""
        self.sign_start_time  = None
        self.last_registered  = ""
        self.last_hand_time   = time.time()

        # Buffers
        self.current_word    = []   # letters being spelled
        self.words_spelled   = []   # completed words
        self.sentence_text   = ""   # final sentence from AI

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        f_title  = font.Font(family="Helvetica", size=17, weight="bold")
        f_label  = font.Font(family="Helvetica", size=11)
        f_btn    = font.Font(family="Helvetica", size=11, weight="bold")
        f_mono   = font.Font(family="Courier",   size=14)
        f_big    = font.Font(family="Helvetica", size=32, weight="bold")
        f_letter = font.Font(family="Helvetica", size=64, weight="bold")

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#0d1f2d", height=58)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Sign Language Communication System",
                 font=f_title, fg=CYAN, bg="#0d1f2d").pack(pady=16)

        # ── Serial bar ────────────────────────────────────────────────────────
        bar = tk.Frame(self.root, bg=PANEL, pady=6)
        bar.pack(fill="x", padx=8, pady=(6, 0))

        tk.Label(bar, text="Arduino:", font=f_label, fg=GRAY, bg=PANEL).pack(side="left", padx=8)
        self.port_var   = tk.StringVar()
        self.port_combo = ttk.Combobox(bar, textvariable=self.port_var, width=14, font=f_label)
        self.port_combo.pack(side="left", padx=4)
        self._refresh_ports()

        tk.Button(bar, text="⟳", command=self._refresh_ports,
                  font=f_btn, bg=PANEL, fg=WHITE, relief="flat", padx=6
                  ).pack(side="left", padx=2)

        self.conn_btn = tk.Button(bar, text="Connect",
                  command=self._toggle_connection,
                  font=f_btn, bg="#006400", fg=WHITE, relief="flat", padx=12)
        self.conn_btn.pack(side="left", padx=8)

        self.conn_lbl = tk.Label(bar, text="● Disconnected",
                  font=f_label, fg=RED, bg=PANEL)
        self.conn_lbl.pack(side="left", padx=6)

        # ── Notebook ──────────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",      background=BG,    borderwidth=0)
        style.configure("TNotebook.Tab",  background=PANEL, foreground=GRAY,
                        padding=[22, 8],  font=("Helvetica", 11, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", WHITE)])

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # ══ TAB 1: Camera → Decode ════════════════════════════════════════════
        t1 = tk.Frame(nb, bg=BG)
        nb.add(t1, text="  📷  Camera → Decode ASL  ")

        # Top controls
        ctrl = tk.Frame(t1, bg=BG)
        ctrl.pack(fill="x", padx=16, pady=(12, 4))

        self.cam_start_btn = tk.Button(ctrl, text="▶  Start Camera",
                  command=self._start_camera,
                  font=f_btn, bg="#006400", fg=WHITE, relief="flat",
                  padx=18, pady=7)
        self.cam_start_btn.pack(side="left", padx=4)

        self.cam_stop_btn = tk.Button(ctrl, text="■  Stop",
                  command=self._stop_camera, state="disabled",
                  font=f_btn, bg="#8B0000", fg=WHITE, relief="flat",
                  padx=14, pady=7)
        self.cam_stop_btn.pack(side="left", padx=4)

        tk.Button(ctrl, text="🗑  Clear All",
                  command=self._clear_all,
                  font=f_btn, bg=PANEL, fg=WHITE, relief="flat",
                  padx=14, pady=7).pack(side="left", padx=4)

        tk.Button(ctrl, text="⌫  Delete word",
                  command=self._delete_last_word,
                  font=f_btn, bg=PANEL, fg=WHITE, relief="flat",
                  padx=14, pady=7).pack(side="left", padx=4)

        tk.Button(ctrl, text="🤖  Send sentence to hand",
                  command=self._send_sentence_to_hand,
                  font=f_btn, bg="#1f3a6e", fg=CYAN, relief="flat",
                  padx=14, pady=7).pack(side="right", padx=4)

        # Current letter display (big)
        letter_frame = tk.Frame(t1, bg=PANEL, height=130)
        letter_frame.pack(fill="x", padx=16, pady=(6, 4))
        letter_frame.pack_propagate(False)

        tk.Label(letter_frame, text="Current sign:",
                 font=f_label, fg=GRAY, bg=PANEL).place(x=12, y=10)

        self.letter_lbl = tk.Label(letter_frame, text="—",
                 font=f_letter, fg=GREEN, bg=PANEL)
        self.letter_lbl.place(x=20, y=30)

        tk.Label(letter_frame, text="Confidence:",
                 font=f_label, fg=GRAY, bg=PANEL).place(x=160, y=10)

        self.conf_bar_canvas = tk.Canvas(letter_frame, bg=PANEL,
                  width=300, height=20, highlightthickness=0)
        self.conf_bar_canvas.place(x=160, y=32)

        tk.Label(letter_frame, text="Hold timer:",
                 font=f_label, fg=GRAY, bg=PANEL).place(x=160, y=66)

        self.hold_bar_canvas = tk.Canvas(letter_frame, bg=PANEL,
                  width=300, height=20, highlightthickness=0)
        self.hold_bar_canvas.place(x=160, y=88)

        # Spelling buffer (letters → words)
        tk.Label(t1, text="Spelling buffer:",
                 font=f_label, fg=GRAY, bg=BG).pack(anchor="w", padx=16, pady=(4, 0))

        self.buffer_lbl = tk.Label(t1, text="",
                 font=f_mono, fg=YELLOW, bg=PANEL,
                 anchor="w", padx=12, pady=8, wraplength=1050)
        self.buffer_lbl.pack(fill="x", padx=16, pady=(2, 4))

        # Sentence output (large, prominent)
        tk.Label(t1, text="Decoded sentence:",
                 font=f_label, fg=GRAY, bg=BG).pack(anchor="w", padx=16, pady=(4, 0))

        self.sentence_lbl = tk.Label(t1, text="",
                 font=f_big, fg=WHITE, bg=PANEL,
                 anchor="w", padx=16, pady=14, wraplength=1050,
                 justify="left")
        self.sentence_lbl.pack(fill="x", padx=16, pady=(2, 6))

        # AI status
        self.ai_status_lbl = tk.Label(t1, text="",
                 font=f_label, fg=GRAY, bg=BG)
        self.ai_status_lbl.pack(anchor="w", padx=16)

        # Instructions
        tk.Label(t1,
                 text="Hold a sign steady for 1.5 s to register the letter.  "
                      "Remove hand for 2 s to finish the current word and trigger AI sentence building.",
                 font=("Helvetica", 9), fg=GRAY, bg=BG, justify="left"
                 ).pack(anchor="w", padx=16, pady=(6, 0))

        # ══ TAB 2: Type → Hand ════════════════════════════════════════════════
        t2 = tk.Frame(nb, bg=BG)
        nb.add(t2, text="  ⌨️  Type → Robotic Hand  ")

        tk.Label(t2,
                 text="Type a word or sentence. The robotic hand will spell it out letter by letter.",
                 font=f_label, fg=GRAY, bg=BG).pack(pady=(14, 4))

        inp = tk.Frame(t2, bg=BG)
        inp.pack(fill="x", padx=20, pady=6)

        tk.Label(inp, text="Enter text:", font=f_label, fg=CYAN, bg=BG).pack(anchor="w")

        self.type_input = tk.Text(inp, height=4, font=f_mono,
                  bg=PANEL, fg=WHITE, insertbackground=WHITE,
                  relief="flat", borderwidth=8, wrap="word")
        self.type_input.pack(fill="x", pady=6)
        self.type_input.bind("<Control-Return>", lambda e: self._send_typed_to_hand())

        speed_row = tk.Frame(t2, bg=BG)
        speed_row.pack(pady=8)
        tk.Label(speed_row, text="Speed:", font=f_label, fg=WHITE, bg=BG).pack(side="left", padx=(0, 10))
        self.speed_var = tk.IntVar(value=1200)
        for lbl, val in [("Slow", 2000), ("Normal", 1200), ("Fast", 700)]:
            tk.Radiobutton(speed_row, text=lbl, variable=self.speed_var, value=val,
                           font=f_label, fg=WHITE, bg=BG, selectcolor=PANEL,
                           activebackground=BG, activeforeground=WHITE
                           ).pack(side="left", padx=10)

        btns2 = tk.Frame(t2, bg=BG)
        btns2.pack(pady=10)

        tk.Button(btns2, text="🤖  Send to Hand",
                  command=self._send_typed_to_hand,
                  font=f_btn, bg="#006400", fg=WHITE, relief="flat",
                  padx=20, pady=10).pack(side="left", padx=8)

        tk.Button(btns2, text="✋  Open Hand",
                  command=lambda: self._serial_send(" "),
                  font=f_btn, bg=PANEL, fg=WHITE, relief="flat",
                  padx=14, pady=10).pack(side="left", padx=8)

        tk.Label(t2, text="Ctrl+Enter to send quickly",
                 font=("Helvetica", 9), fg=GRAY, bg=BG).pack()

        tk.Label(t2, text="Progress:", font=f_label, fg=CYAN, bg=BG).pack(anchor="w", padx=20, pady=(14, 2))

        self.prog_var = tk.StringVar(value="Idle")
        tk.Label(t2, textvariable=self.prog_var, font=f_mono,
                 fg=GREEN, bg=PANEL, anchor="w", padx=10, pady=7
                 ).pack(fill="x", padx=20)

        self.prog_bar = ttk.Progressbar(t2, mode="determinate")
        self.prog_bar.pack(fill="x", padx=20, pady=6)

        # ── Status bar ────────────────────────────────────────────────────────
        sb = tk.Frame(self.root, bg="#0a0a14", height=28)
        sb.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(sb, textvariable=self.status_var,
                 font=("Helvetica", 9), fg=GRAY, bg="#0a0a14"
                 ).pack(side="left", padx=10, pady=4)

    # ─────────────────────────────────────────────────────────────────────────
    #  Serial
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if ports:
            self.port_combo.current(0)

    def _toggle_connection(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.serial_conn = None
            self.conn_lbl.config(text="● Disconnected", fg=RED)
            self.conn_btn.config(text="Connect", bg="#006400")
            self._set_status("Disconnected")
        else:
            port = self.port_var.get()
            if not port:
                self._set_status("Select a port first")
                return
            try:
                self.serial_conn = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
                self.conn_lbl.config(text="● Connected", fg=GREEN)
                self.conn_btn.config(text="Disconnect", bg="#8B0000")
                self._set_status(f"Connected to {port}")
            except Exception as e:
                self._set_status(f"Connection error: {e}")

    def _serial_send(self, text):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(text.encode())

    # ─────────────────────────────────────────────────────────────────────────
    #  Camera thread
    # ─────────────────────────────────────────────────────────────────────────

    def _start_camera(self):
        if self.cam_running:
            return
        self.cam_running = True
        self.cam_start_btn.config(state="disabled")
        self.cam_stop_btn.config(state="normal")
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

    def _stop_camera(self):
        self.cam_running = False
        self.cam_start_btn.config(state="normal")
        self.cam_stop_btn.config(state="disabled")

    def _camera_loop(self):
        mp_hands   = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        cap        = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Per-loop state
        letter_history  = []
        hold_start      = None
        current_stable  = ""
        last_registered = ""
        last_hand_seen  = time.time()

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.80,
            min_tracking_confidence=0.75
        ) as hands:
            while self.cam_running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                frame  = cv2.flip(frame, 1)
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                detected    = "?"
                confidence  = 0.0
                hold_frac   = 0.0
                hand_visible = False

                if result.multi_hand_landmarks:
                    hand_visible    = True
                    last_hand_seen  = time.time()
                    lm = result.multi_hand_landmarks[0].landmark

                    mp_drawing.draw_landmarks(
                        frame,
                        result.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 220, 120), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2)
                    )

                    detected, confidence = classify_with_confidence(
                        lm, letter_history, HISTORY_SIZE
                    )

                    # Hold-to-register logic
                    if detected == current_stable and detected != "?" and confidence >= MIN_CONFIDENCE:
                        if hold_start is None:
                            hold_start = time.time()
                        elapsed   = time.time() - hold_start
                        hold_frac = min(elapsed / HOLD_SECONDS, 1.0)

                        if elapsed >= HOLD_SECONDS and detected != last_registered:
                            # Register the letter
                            last_registered = detected
                            hold_start      = None
                            self.root.after(0, self._register_letter, detected)
                    else:
                        current_stable  = detected
                        hold_start      = None
                        last_registered = ""

                else:
                    # No hand — check if we should end the current word
                    gap = time.time() - last_hand_seen
                    if gap >= WORD_GAP_SECONDS and self.current_word:
                        self.root.after(0, self._end_word)
                    letter_history.clear()
                    current_stable  = ""
                    hold_start      = None
                    last_registered = ""

                # ── CV overlay ────────────────────────────────────────────────
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 80), (10, 14, 20), -1)

                color = (0, 220, 120) if detected != "?" else (150, 150, 150)
                cv2.putText(frame, f"Sign: {detected}",
                            (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)

                # Confidence bar
                bar_w = int((w - 320) * confidence)
                cv2.rectangle(frame, (230, 20), (230 + bar_w, 36), (0, 140, 255), -1)
                cv2.putText(frame, f"conf {int(confidence*100)}%",
                            (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

                # Hold timer bar
                hold_bar_w = int((w - 320) * hold_frac)
                cv2.rectangle(frame, (230, 44), (230 + hold_bar_w, 56),
                              (0, 255, 120) if hold_frac < 1.0 else (0, 255, 255), -1)

                # Bottom: spelled buffer
                cv2.rectangle(frame, (0, h - 50), (w, h), (10, 14, 20), -1)
                buf_display = " ".join(
                    ["".join(self.words_spelled[i]) if i < len(self.words_spelled)
                     else "" for i in range(len(self.words_spelled))]
                    + ["".join(self.current_word)]
                )
                cv2.putText(frame, buf_display[-80:],
                            (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 210, 0), 2)

                # Update UI bars
                self.root.after(0, self._update_bars, detected, confidence, hold_frac)

                cv2.imshow("ASL Sign Decoder — press Q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.cam_running = False
        self.root.after(0, lambda: self.cam_start_btn.config(state="normal"))
        self.root.after(0, lambda: self.cam_stop_btn.config(state="disabled"))

    # ─────────────────────────────────────────────────────────────────────────
    #  Letter / word management
    # ─────────────────────────────────────────────────────────────────────────

    def _register_letter(self, letter):
        """Called on main thread when a letter is confirmed."""
        self.current_word.append(letter)
        self._update_buffer_display()
        self._set_status(f"Registered: {letter}")

    def _end_word(self):
        """Called when hand is removed for WORD_GAP_SECONDS — finalises the word."""
        if not self.current_word:
            return
        word = "".join(self.current_word)
        self.words_spelled.append(list(self.current_word))
        self.current_word.clear()
        self._update_buffer_display()
        self._set_status(f"Word complete: {word}")

        # Trigger AI sentence building in background
        self.ai_status_lbl.config(text="Building sentence with AI…", fg=YELLOW)
        threading.Thread(target=self._build_sentence_thread, daemon=True).start()

    def _build_sentence_thread(self):
        """Background thread: call Claude AI to build sentence."""
        words = ["".join(w) for w in self.words_spelled]
        sentence = build_sentence(words)
        self.sentence_text = sentence
        self.root.after(0, self._show_sentence, sentence)

    def _show_sentence(self, sentence):
        self.sentence_lbl.config(text=sentence)
        self.ai_status_lbl.config(text="✓ Sentence ready", fg=GREEN)

    def _update_buffer_display(self):
        parts = ["".join(w) for w in self.words_spelled] + ["".join(self.current_word)]
        display = "  |  ".join(p for p in parts if p)
        self.buffer_lbl.config(text=display or "—")

    def _update_bars(self, letter, confidence, hold_frac):
        """Update the letter label and progress bars in the UI."""
        color = GREEN if letter != "?" else GRAY
        self.letter_lbl.config(text=letter if letter != "?" else "—", fg=color)

        # Confidence bar
        self.conf_bar_canvas.delete("all")
        w = 300
        fill_w = int(w * confidence)
        bar_color = GREEN if confidence >= MIN_CONFIDENCE else YELLOW
        self.conf_bar_canvas.create_rectangle(0, 0, w, 20, fill=PANEL, outline="")
        self.conf_bar_canvas.create_rectangle(0, 0, fill_w, 20, fill=bar_color, outline="")
        self.conf_bar_canvas.create_text(fill_w + 4, 10,
            text=f"{int(confidence*100)}%", fill=WHITE,
            font=("Courier", 10), anchor="w")

        # Hold timer bar
        self.hold_bar_canvas.delete("all")
        hold_w = int(w * hold_frac)
        hold_color = CYAN if hold_frac < 1.0 else WHITE
        self.hold_bar_canvas.create_rectangle(0, 0, w, 20, fill=PANEL, outline="")
        self.hold_bar_canvas.create_rectangle(0, 0, hold_w, 20, fill=hold_color, outline="")

    def _clear_all(self):
        self.current_word.clear()
        self.words_spelled.clear()
        self.sentence_text = ""
        self.sentence_lbl.config(text="")
        self.buffer_lbl.config(text="—")
        self.ai_status_lbl.config(text="")
        self._set_status("Cleared")

    def _delete_last_word(self):
        if self.current_word:
            self.current_word.clear()
        elif self.words_spelled:
            self.words_spelled.pop()
        self._update_buffer_display()

    def _send_sentence_to_hand(self):
        text = self.sentence_text.strip()
        if not text:
            self._set_status("No sentence to send")
            return
        threading.Thread(target=self._transmit, args=(text,), daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Mode 2: Type → Hand
    # ─────────────────────────────────────────────────────────────────────────

    def _send_typed_to_hand(self):
        text = self.type_input.get("1.0", "end").strip().upper()
        if not text:
            self._set_status("Nothing to send")
            return
        if not self.serial_conn or not self.serial_conn.is_open:
            self._set_status("Not connected to Arduino")
            return
        threading.Thread(target=self._transmit, args=(text,), daemon=True).start()

    def _transmit(self, text):
        """Send text to Arduino letter by letter."""
        chars = [c for c in text if c.isalpha() or c == " "]
        total = len(chars)
        delay = self.speed_var.get() / 1000.0

        self.root.after(0, lambda: self.prog_bar.config(maximum=max(total, 1), value=0))

        for i, ch in enumerate(chars):
            self.root.after(0, lambda c=ch, idx=i: (
                self.prog_var.set(f"Sending: '{c}'  ({idx+1}/{total})"),
                self.prog_bar.config(value=idx + 1)
            ))
            self._serial_send(ch)
            time.sleep(delay)

        self.root.after(0, lambda: (
            self.prog_var.set(f"✅  Done — sent {total} characters"),
            self.prog_bar.config(value=total)
        ))
        self._set_status(f"Sent: {text[:40]}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
