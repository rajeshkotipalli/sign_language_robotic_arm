"""
main_interface.py  —  Sign Language Communication System  v2  (FIXED)
=============================================================
Tab 1: Camera → Decode ASL (letter by letter, AI sentence)
Tab 2: Type → Robotic Hand (spell words to hand)
Tab 3: Gesture Library (record/upload reference sequences + meanings)
Tab 4: Gesture Match  (live camera matches against library → shows meaning → hand spells it)

FIXES:
- Camera now renders INSIDE the tkinter window (no separate OpenCV window needed)
- Removed duplicate import of classify_with_confidence inside loop
- Added camera index selector (try 0,1,2 if camera not found)
- All errors shown in GUI status bar instead of crashing silently
- Tab 4 camera also renders inside GUI
"""

import tkinter as tk
from tkinter import ttk, font, filedialog, messagebox
import threading
import serial
import serial.tools.list_ports
import time
import cv2
import mediapipe as mp
import os
from PIL import Image, ImageTk   # pip install Pillow

from asl_recognizer   import classify_asl, classify_with_confidence
from sentence_builder import build_sentence
from gesture_library  import (
    load_library, save_library, add_entry, delete_entry,
    record_from_camera, extract_from_video,
    find_best_match, landmarks_to_list, normalise_frame,
    sequence_to_numpy
)

# ── Colours ───────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
ACCENT = "#1f6feb"
GREEN  = "#3fb950"
YELLOW = "#d29922"
WHITE  = "#f0f6fc"
GRAY   = "#8b949e"
RED    = "#f85149"
CYAN   = "#79c0ff"
PURPLE = "#bc8cff"
ORANGE = "#ffa657"

# ── Matching ──────────────────────────────────────────────────────────────────
MATCH_THRESHOLD   = 0.18
MATCH_WINDOW_SEC  = 4.0
HOLD_SECONDS      = 1.5
WORD_GAP_SECONDS  = 2.0
HISTORY_SIZE      = 12
MIN_CONFIDENCE    = 0.65

# ── Camera display size inside GUI ───────────────────────────────────────────
CAM_W = 640
CAM_H = 360

# ─────────────────────────────────────────────────────────────────────────────
class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Communication System v2")
        self.root.geometry("1200x900")
        self.root.configure(bg=BG)

        # Serial
        self.serial_conn = None

        # Camera index (user can change if cam not found)
        self.cam_index = 1

        # Camera / decode state (Tab 1)
        self.cam1_running   = False
        self.letter_history = []
        self.current_word   = []
        self.words_spelled  = []
        self.sentence_text  = ""
        self._t1_photo      = None   # keep reference to avoid GC

        # Gesture library
        self.library = load_library()

        # Match state (Tab 4)
        self.cam4_running      = False
        self.match_window      = []
        self.match_window_start = None
        self.last_match_result = None
        self._t4_photo         = None

        self._build_ui()

    # =========================================================================
    #  UI
    # =========================================================================

    def _build_ui(self):
        fT  = font.Font(family="Helvetica", size=17, weight="bold")
        fL  = font.Font(family="Helvetica", size=11)
        fB  = font.Font(family="Helvetica", size=11, weight="bold")
        fM  = font.Font(family="Courier",   size=13)
        fBig= font.Font(family="Helvetica", size=24, weight="bold")
        fLet= font.Font(family="Helvetica", size=60, weight="bold")

        # Header
        hdr = tk.Frame(self.root, bg="#0a1628", height=55)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Sign Language Communication System",
                 font=fT, fg=CYAN, bg="#0a1628").pack(pady=14)

        # Serial bar
        bar = tk.Frame(self.root, bg=PANEL, pady=5)
        bar.pack(fill="x", padx=6, pady=(5, 0))
        tk.Label(bar, text="Arduino:", font=fL, fg=GRAY, bg=PANEL).pack(side="left", padx=8)
        self.port_var   = tk.StringVar()
        self.port_combo = ttk.Combobox(bar, textvariable=self.port_var, width=12, font=fL)
        self.port_combo.pack(side="left", padx=3)
        self._refresh_ports()
        tk.Button(bar, text="⟳", command=self._refresh_ports,
                  font=fB, bg=PANEL, fg=WHITE, relief="flat", padx=5).pack(side="left", padx=2)
        self.conn_btn = tk.Button(bar, text="Connect", command=self._toggle_conn,
                  font=fB, bg="#006400", fg=WHITE, relief="flat", padx=10)
        self.conn_btn.pack(side="left", padx=6)
        self.conn_lbl = tk.Label(bar, text="● Disconnected", font=fL, fg=RED, bg=PANEL)
        self.conn_lbl.pack(side="left", padx=6)

        # Camera index selector
        tk.Label(bar, text="  Cam:", font=fL, fg=GRAY, bg=PANEL).pack(side="left", padx=(20,2))
        self.cam_var = tk.IntVar(value=1)
        for idx in [0, 1, 2]:
            tk.Radiobutton(bar, text=str(idx), variable=self.cam_var, value=idx,
                           font=fL, fg=WHITE, bg=PANEL, selectcolor=BG,
                           activebackground=PANEL, activeforeground=WHITE,
                           command=lambda: setattr(self, 'cam_index', self.cam_var.get())
                           ).pack(side="left", padx=3)

        # Notebook
        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure("TNotebook",     background=BG,  borderwidth=0)
        sty.configure("TNotebook.Tab", background=PANEL, foreground=GRAY,
                       padding=[18, 8], font=("Helvetica", 11, "bold"))
        sty.map("TNotebook.Tab",
                background=[("selected", ACCENT)],
                foreground=[("selected", WHITE)])

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        self._build_tab1(nb, fL, fB, fM, fBig, fLet)
        self._build_tab2(nb, fL, fB, fM)
        self._build_tab3(nb, fL, fB, fM)
        self._build_tab4(nb, fL, fB, fM, fBig)

        # Status bar
        sb = tk.Frame(self.root, bg="#08080f", height=26)
        sb.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(sb, textvariable=self.status_var,
                 font=("Helvetica", 9), fg=GRAY, bg="#08080f"
                 ).pack(side="left", padx=10, pady=3)

    # ── Tab 1: Camera → Decode ────────────────────────────────────────────────
    def _build_tab1(self, nb, fL, fB, fM, fBig, fLet):
        t = tk.Frame(nb, bg=BG); nb.add(t, text="  📷  Camera Decode  ")

        # Left: camera feed
        left = tk.Frame(t, bg=BG)
        left.pack(side="left", fill="y", padx=(10,4), pady=8)

        ctrl = tk.Frame(left, bg=BG); ctrl.pack(fill="x", pady=(0,6))
        self.t1_start = tk.Button(ctrl, text="▶ Start Camera", command=self._t1_start,
                  font=fB, bg="#006400", fg=WHITE, relief="flat", padx=12, pady=5)
        self.t1_start.pack(side="left", padx=2)
        self.t1_stop = tk.Button(ctrl, text="■ Stop", command=self._t1_stop, state="disabled",
                  font=fB, bg="#8B0000", fg=WHITE, relief="flat", padx=8, pady=5)
        self.t1_stop.pack(side="left", padx=2)
        tk.Button(ctrl, text="🗑 Clear", command=self._t1_clear,
                  font=fB, bg=PANEL, fg=WHITE, relief="flat", padx=8, pady=5).pack(side="left", padx=2)
        tk.Button(ctrl, text="⌫ Del word", command=self._t1_del_word,
                  font=fB, bg=PANEL, fg=WHITE, relief="flat", padx=8, pady=5).pack(side="left", padx=2)

        # Camera canvas — renders frame here
        self.t1_cam_canvas = tk.Canvas(left, width=CAM_W, height=CAM_H,
                                        bg="#050810", highlightthickness=1,
                                        highlightbackground=BORDER)
        self.t1_cam_canvas.pack()
        self.t1_cam_canvas.create_text(CAM_W//2, CAM_H//2,
                                        text="Camera feed will appear here\nPress ▶ Start Camera",
                                        fill=GRAY, font=("Helvetica", 13), justify="center")

        # Right: info panel
        right = tk.Frame(t, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(4,10), pady=8)

        # Letter display
        lf = tk.Frame(right, bg=PANEL); lf.pack(fill="x", pady=(0,6))
        tk.Label(lf, text="Current sign:", font=fL, fg=GRAY, bg=PANEL).pack(anchor="w", padx=10, pady=(6,0))
        self.t1_letter_lbl = tk.Label(lf, text="—", font=fLet, fg=GREEN, bg=PANEL)
        self.t1_letter_lbl.pack(anchor="w", padx=14)

        tk.Label(right, text="Confidence:", font=fL, fg=GRAY, bg=BG).pack(anchor="w", padx=4)
        self.t1_conf_cv = tk.Canvas(right, bg=BG, width=350, height=18, highlightthickness=0)
        self.t1_conf_cv.pack(anchor="w", padx=4, pady=(0,4))

        tk.Label(right, text="Hold timer:", font=fL, fg=GRAY, bg=BG).pack(anchor="w", padx=4)
        self.t1_hold_cv = tk.Canvas(right, bg=BG, width=350, height=18, highlightthickness=0)
        self.t1_hold_cv.pack(anchor="w", padx=4, pady=(0,8))

        tk.Label(right, text="Spelling buffer:", font=fL, fg=GRAY, bg=BG).pack(anchor="w", padx=4)
        self.t1_buf_lbl = tk.Label(right, text="—", font=fM, fg=YELLOW, bg=PANEL,
                  anchor="w", padx=10, pady=6, wraplength=480)
        self.t1_buf_lbl.pack(fill="x", padx=4, pady=(2,8))

        tk.Label(right, text="AI sentence:", font=fL, fg=GRAY, bg=BG).pack(anchor="w", padx=4)
        self.t1_sentence_lbl = tk.Label(right, text="", font=fBig, fg=WHITE, bg=PANEL,
                  anchor="w", padx=10, pady=10, wraplength=480, justify="left")
        self.t1_sentence_lbl.pack(fill="x", padx=4, pady=(2,4))
        self.t1_ai_lbl = tk.Label(right, text="", font=fL, fg=GRAY, bg=BG)
        self.t1_ai_lbl.pack(anchor="w", padx=4)

        tk.Button(right, text="🤖 Send to Hand", command=self._t1_send_to_hand,
                  font=fB, bg="#1f3a6e", fg=CYAN, relief="flat", padx=12, pady=6
                  ).pack(anchor="w", padx=4, pady=10)

        tk.Label(right, text="Hold a sign 1.5s to register · Remove hand 2s to finish word",
                 font=("Helvetica", 9), fg=GRAY, bg=BG).pack(anchor="w", padx=4)

    # ── Tab 2: Type → Hand ────────────────────────────────────────────────────
    def _build_tab2(self, nb, fL, fB, fM):
        t = tk.Frame(nb, bg=BG); nb.add(t, text="  ⌨️  Type → Hand  ")
        tk.Label(t, text="Type a sentence. The hand will spell it out letter by letter.",
                 font=fL, fg=GRAY, bg=BG).pack(pady=(14, 4))
        inp = tk.Frame(t, bg=BG); inp.pack(fill="x", padx=20, pady=6)
        tk.Label(inp, text="Enter text:", font=fL, fg=CYAN, bg=BG).pack(anchor="w")
        self.t2_input = tk.Text(inp, height=4, font=fM,
                  bg=PANEL, fg=WHITE, insertbackground=WHITE,
                  relief="flat", borderwidth=8, wrap="word")
        self.t2_input.pack(fill="x", pady=6)
        self.t2_input.bind("<Control-Return>", lambda e: self._t2_send())
        sr = tk.Frame(t, bg=BG); sr.pack(pady=8)
        tk.Label(sr, text="Speed:", font=fL, fg=WHITE, bg=BG).pack(side="left", padx=(0, 10))
        self.speed_var = tk.IntVar(value=1200)
        for lbl, val in [("Slow", 2000), ("Normal", 1200), ("Fast", 700)]:
            tk.Radiobutton(sr, text=lbl, variable=self.speed_var, value=val,
                           font=fL, fg=WHITE, bg=BG, selectcolor=PANEL,
                           activebackground=BG, activeforeground=WHITE).pack(side="left", padx=10)
        br = tk.Frame(t, bg=BG); br.pack(pady=8)
        tk.Button(br, text="🤖 Send to Hand", command=self._t2_send,
                  font=fB, bg="#006400", fg=WHITE, relief="flat", padx=18, pady=8).pack(side="left", padx=6)
        tk.Button(br, text="✋ Open Hand", command=lambda: self._serial_send(" "),
                  font=fB, bg=PANEL, fg=WHITE, relief="flat", padx=12, pady=8).pack(side="left", padx=6)
        tk.Label(t, text="Ctrl+Enter to send", font=("Helvetica", 9), fg=GRAY, bg=BG).pack()
        tk.Label(t, text="Progress:", font=fL, fg=CYAN, bg=BG).pack(anchor="w", padx=20, pady=(14, 2))
        self.t2_prog_var = tk.StringVar(value="Idle")
        tk.Label(t, textvariable=self.t2_prog_var, font=fM,
                 fg=GREEN, bg=PANEL, anchor="w", padx=10, pady=6).pack(fill="x", padx=20)
        self.t2_prog_bar = ttk.Progressbar(t, mode="determinate")
        self.t2_prog_bar.pack(fill="x", padx=20, pady=6)

    # ── Tab 3: Gesture Library ────────────────────────────────────────────────
    def _build_tab3(self, nb, fL, fB, fM):
        t = tk.Frame(nb, bg=BG); nb.add(t, text="  📚  Gesture Library  ")
        left = tk.Frame(t, bg=PANEL, width=280)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        tk.Label(left, text="Saved Gestures", font=fB, fg=CYAN, bg=PANEL).pack(pady=(10, 4))
        self.lib_listbox = tk.Listbox(left, bg="#0d1117", fg=WHITE,
                  selectbackground=ACCENT, font=fM,
                  relief="flat", borderwidth=0, activestyle="none")
        self.lib_listbox.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.lib_listbox.bind("<<ListboxSelect>>", self._lib_on_select)
        tk.Button(left, text="🗑 Delete selected", command=self._lib_delete,
                  font=fB, bg="#8B0000", fg=WHITE, relief="flat", pady=6
                  ).pack(fill="x", padx=8, pady=(0, 10))
        right = tk.Frame(t, bg=BG); right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        tk.Label(right, text="Gesture name (short label):", font=fL, fg=GRAY, bg=BG).pack(anchor="w", pady=(0, 2))
        self.lib_name_var = tk.StringVar()
        tk.Entry(right, textvariable=self.lib_name_var, font=fM,
                 bg=PANEL, fg=WHITE, insertbackground=WHITE,
                 relief="flat", borderwidth=6).pack(fill="x", pady=(0, 10))
        tk.Label(right, text="Meaning (what this gesture means):", font=fL, fg=GRAY, bg=BG).pack(anchor="w", pady=(0, 2))
        self.lib_meaning_txt = tk.Text(right, height=3, font=fM,
                  bg=PANEL, fg=WHITE, insertbackground=WHITE,
                  relief="flat", borderwidth=6, wrap="word")
        self.lib_meaning_txt.pack(fill="x", pady=(0, 12))
        tk.Label(right, text="Recording duration (seconds):", font=fL, fg=GRAY, bg=BG).pack(anchor="w")
        self.lib_dur_var = tk.IntVar(value=5)
        dur_row = tk.Frame(right, bg=BG); dur_row.pack(fill="x", pady=(2, 12))
        for d in [3, 5, 8, 10]:
            tk.Radiobutton(dur_row, text=f"{d}s", variable=self.lib_dur_var, value=d,
                           font=fL, fg=WHITE, bg=BG, selectcolor=PANEL,
                           activebackground=BG, activeforeground=WHITE).pack(side="left", padx=8)
        abr = tk.Frame(right, bg=BG); abr.pack(fill="x", pady=6)
        tk.Button(abr, text="🎥  Record Live (camera)",
                  command=self._lib_record_live,
                  font=fB, bg="#006400", fg=WHITE, relief="flat",
                  padx=16, pady=9).pack(side="left", padx=(0, 8))
        tk.Button(abr, text="📁  Import Video File",
                  command=self._lib_import_video,
                  font=fB, bg="#1f3a6e", fg=CYAN, relief="flat",
                  padx=16, pady=9).pack(side="left")
        self.lib_status_lbl = tk.Label(right, text="", font=fL, fg=GREEN, bg=BG, wraplength=750)
        self.lib_status_lbl.pack(anchor="w", pady=(10, 0))
        self.lib_detail_lbl = tk.Label(right, text="", font=fM, fg=YELLOW, bg=PANEL,
                  anchor="w", padx=10, pady=8, wraplength=750)
        self.lib_detail_lbl.pack(fill="x", pady=(4, 0))
        self._lib_refresh_list()

    # ── Tab 4: Gesture Match ──────────────────────────────────────────────────
    def _build_tab4(self, nb, fL, fB, fM, fBig):
        t = tk.Frame(nb, bg=BG); nb.add(t, text="  🔍  Gesture Match  ")

        # Left: camera
        left = tk.Frame(t, bg=BG)
        left.pack(side="left", fill="y", padx=(10,4), pady=8)

        ctrl = tk.Frame(left, bg=BG); ctrl.pack(fill="x", pady=(0,6))
        self.t4_start = tk.Button(ctrl, text="▶ Start Matching", command=self._t4_start,
                  font=fB, bg="#006400", fg=WHITE, relief="flat", padx=12, pady=5)
        self.t4_start.pack(side="left", padx=2)
        self.t4_stop = tk.Button(ctrl, text="■ Stop", command=self._t4_stop, state="disabled",
                  font=fB, bg="#8B0000", fg=WHITE, relief="flat", padx=8, pady=5)
        self.t4_stop.pack(side="left", padx=2)
        tk.Button(ctrl, text="🗑 Clear result", command=self._t4_clear,
                  font=fB, bg=PANEL, fg=WHITE, relief="flat", padx=8, pady=5).pack(side="left", padx=2)

        self.t4_cam_canvas = tk.Canvas(left, width=CAM_W, height=CAM_H,
                                        bg="#050810", highlightthickness=1,
                                        highlightbackground=BORDER)
        self.t4_cam_canvas.pack()
        self.t4_cam_canvas.create_text(CAM_W//2, CAM_H//2,
                                        text="Camera feed will appear here\nPress ▶ Start Matching",
                                        fill=GRAY, font=("Helvetica", 13), justify="center")

        # Window config
        wr = tk.Frame(left, bg=BG); wr.pack(fill="x", pady=(6,0))
        tk.Label(wr, text="Capture window:", font=fL, fg=GRAY, bg=BG).pack(side="left")
        self.t4_window_var = tk.IntVar(value=4)
        for d in [3, 4, 5, 8]:
            tk.Radiobutton(wr, text=f"{d}s", variable=self.t4_window_var, value=d,
                           font=fL, fg=WHITE, bg=BG, selectcolor=PANEL,
                           activebackground=BG, activeforeground=WHITE).pack(side="left", padx=6)

        # Right: results
        right = tk.Frame(t, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(4,10), pady=8)

        # Status panel
        sp = tk.Frame(right, bg=PANEL); sp.pack(fill="x", pady=(0,8))
        tk.Label(sp, text="Status:", font=fL, fg=GRAY, bg=PANEL).pack(anchor="w", padx=10, pady=(6,0))
        self.t4_status_lbl = tk.Label(sp, text="Idle — start matching to begin",
                  font=fB, fg=GRAY, bg=PANEL)
        self.t4_status_lbl.pack(anchor="w", padx=10)
        tk.Label(sp, text="Window timer:", font=fL, fg=GRAY, bg=PANEL).pack(anchor="w", padx=10, pady=(4,0))
        self.t4_timer_cv = tk.Canvas(sp, bg=PANEL, width=380, height=18, highlightthickness=0)
        self.t4_timer_cv.pack(anchor="w", padx=10)
        self.t4_frames_lbl = tk.Label(sp, text="", font=fL, fg=GRAY, bg=PANEL)
        self.t4_frames_lbl.pack(anchor="w", padx=10, pady=(2,8))

        tk.Label(right, text="Matched gesture:", font=fL, fg=GRAY, bg=BG).pack(anchor="w", padx=4, pady=(6,0))
        self.t4_match_name = tk.Label(right, text="—", font=fB, fg=ORANGE, bg=PANEL,
                  anchor="w", padx=12, pady=6)
        self.t4_match_name.pack(fill="x", padx=4, pady=(2,2))

        tk.Label(right, text="Meaning (answer):", font=fL, fg=GRAY, bg=BG).pack(anchor="w", padx=4)
        self.t4_meaning_lbl = tk.Label(right, text="", font=fBig, fg=WHITE, bg=PANEL,
                  anchor="w", padx=14, pady=14, wraplength=480, justify="left")
        self.t4_meaning_lbl.pack(fill="x", padx=4, pady=(2,6))

        tk.Button(right, text="🤖 Send answer to Hand", command=self._t4_send_to_hand,
                  font=fB, bg="#1f3a6e", fg=CYAN, relief="flat", padx=14, pady=7
                  ).pack(anchor="w", padx=4, pady=6)

        tk.Label(right,
                 text="The robotic hand will automatically spell the meaning when a match is found.",
                 font=("Helvetica", 9), fg=GRAY, bg=BG).pack(anchor="w", padx=4)

    # =========================================================================
    #  Serial helpers
    # =========================================================================

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if ports: self.port_combo.current(0)

    def _toggle_conn(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close(); self.serial_conn = None
            self.conn_lbl.config(text="● Disconnected", fg=RED)
            self.conn_btn.config(text="Connect", bg="#006400")
            self._status("Disconnected")
        else:
            port = self.port_var.get()
            if not port: self._status("Select a port first"); return
            try:
                self.serial_conn = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
                self.conn_lbl.config(text="● Connected", fg=GREEN)
                self.conn_btn.config(text="Disconnect", bg="#8B0000")
                self._status(f"Connected to {port}")
            except Exception as e:
                self._status(f"Error: {e}")
                messagebox.showerror("Serial Error", str(e))

    def _serial_send(self, text):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(text.encode())

    def _transmit(self, text):
        chars = [c for c in text if c.isalpha() or c == " "]
        total = len(chars)
        delay = self.speed_var.get() / 1000.0
        self.root.after(0, lambda: self.t2_prog_bar.config(maximum=max(total, 1), value=0))
        for i, ch in enumerate(chars):
            self.root.after(0, lambda c=ch, idx=i: (
                self.t2_prog_var.set(f"Sending: '{c}'  ({idx+1}/{total})"),
                self.t2_prog_bar.config(value=idx+1)
            ))
            self._serial_send(ch)
            time.sleep(delay)
        self.root.after(0, lambda: (
            self.t2_prog_var.set(f"✅ Done — {total} characters sent"),
            self.t2_prog_bar.config(value=total)
        ))

    def _transmit_meaning(self, text):
        chars = [c for c in text.upper() if c.isalpha() or c == " "]
        delay = self.speed_var.get() / 1000.0 if hasattr(self, 'speed_var') else 1.2
        for ch in chars:
            self._serial_send(ch)
            time.sleep(delay)

    # =========================================================================
    #  Camera helpers  — renders frame into tkinter Canvas
    # =========================================================================

    def _frame_to_tkimage(self, frame, w=CAM_W, h=CAM_H):
        """Convert OpenCV BGR frame → ImageTk.PhotoImage sized to (w, h)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (w, h))
        img = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(image=img)

    def _open_camera(self):
        """Try to open camera at selected index. Returns cap or None."""
        idx = self.cam_var.get()
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            # try next indices automatically
            for alt in [0, 1, 2]:
                cap = cv2.VideoCapture(alt)
                if cap.isOpened():
                    self._status(f"Camera {alt} opened (index {idx} failed)")
                    break
            else:
                self._status("ERROR: No camera found. Check Cam index buttons.")
                return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    # =========================================================================
    #  Tab 1: Camera Decode
    # =========================================================================

    def _t1_start(self):
        if self.cam1_running: return
        self.cam1_running = True
        self.t1_start.config(state="disabled"); self.t1_stop.config(state="normal")
        self._status("Starting camera…")
        threading.Thread(target=self._t1_loop, daemon=True).start()

    def _t1_stop(self):
        self.cam1_running = False
        self.t1_start.config(state="normal"); self.t1_stop.config(state="disabled")
        self._status("Camera stopped")

    def _t1_loop(self):
        mp_hands   = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        cap = self._open_camera()
        if cap is None:
            self.cam1_running = False
            self.root.after(0, lambda: (self.t1_start.config(state="normal"),
                                        self.t1_stop.config(state="disabled")))
            return

        letter_history  = []
        hold_start      = None
        current_stable  = ""
        last_registered = ""
        last_hand_seen  = time.time()

        with mp_hands.Hands(max_num_hands=1,
                             min_detection_confidence=0.80,
                             min_tracking_confidence=0.75) as hands:
            while self.cam1_running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    self._status("Camera read failed — check connection")
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                detected   = "?"
                confidence = 0.0
                hold_frac  = 0.0

                if result.multi_hand_landmarks:
                    last_hand_seen = time.time()
                    lm = result.multi_hand_landmarks[0].landmark
                    mp_drawing.draw_landmarks(
                        frame, result.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 220, 120), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2))

                    detected, confidence = classify_with_confidence(lm, letter_history, HISTORY_SIZE)

                    if (detected == current_stable and
                            detected != "?" and
                            confidence >= MIN_CONFIDENCE):
                        if hold_start is None:
                            hold_start = time.time()
                        elapsed   = time.time() - hold_start
                        hold_frac = min(elapsed / HOLD_SECONDS, 1.0)
                        if elapsed >= HOLD_SECONDS and detected != last_registered:
                            last_registered = detected
                            hold_start      = None
                            self.root.after(0, self._t1_register, detected)
                    else:
                        current_stable  = detected
                        hold_start      = None
                        last_registered = ""
                else:
                    if (time.time() - last_hand_seen >= WORD_GAP_SECONDS and
                            self.current_word):
                        self.root.after(0, self._t1_end_word)
                    letter_history.clear()
                    current_stable  = ""
                    hold_start      = None
                    last_registered = ""

                # Draw overlay text on frame
                h_f, w_f = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w_f, 50), (10, 14, 20), -1)
                sign_text = detected if detected != "?" else "-"
                conf_text = f"{int(confidence*100)}%"
                cv2.putText(frame, f"Sign: {sign_text}  Conf: {conf_text}",
                            (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 220, 120) if detected != "?" else (150, 150, 150), 2)

                # ── Render frame inside tkinter canvas ──
                photo = self._frame_to_tkimage(frame)
                def _update_cam1(p=photo):
                    self._t1_photo = p   # prevent garbage collection
                    self.t1_cam_canvas.delete("all")
                    self.t1_cam_canvas.create_image(0, 0, anchor="nw", image=p)
                self.root.after(0, _update_cam1)

                # Update bars
                self.root.after(0, self._t1_update_bars, detected, confidence, hold_frac)

                # Small sleep to avoid flooding the GUI event queue
                time.sleep(0.03)

        cap.release()
        self.cam1_running = False
        self.root.after(0, lambda: (
            self.t1_start.config(state="normal"),
            self.t1_stop.config(state="disabled"),
            self._status("Camera stopped")
        ))

    def _t1_register(self, letter):
        self.current_word.append(letter)
        self._t1_update_buf()
        self._status(f"Registered: {letter}")

    def _t1_end_word(self):
        if not self.current_word: return
        word = "".join(self.current_word)
        self.words_spelled.append(list(self.current_word))
        self.current_word.clear()
        self._t1_update_buf()
        self._status(f"Word done: {word}")
        self.t1_ai_lbl.config(text="Building sentence…", fg=YELLOW)
        threading.Thread(target=self._t1_ai_thread, daemon=True).start()

    def _t1_ai_thread(self):
        words = ["".join(w) for w in self.words_spelled]
        sent  = build_sentence(words)
        self.sentence_text = sent
        self.root.after(0, lambda: (
            self.t1_sentence_lbl.config(text=sent),
            self.t1_ai_lbl.config(text="✓ Sentence ready", fg=GREEN)
        ))

    def _t1_update_buf(self):
        parts = ["".join(w) for w in self.words_spelled] + ["".join(self.current_word)]
        self.t1_buf_lbl.config(text="  |  ".join(p for p in parts if p) or "—")

    def _t1_update_bars(self, letter, conf, hold):
        self.t1_letter_lbl.config(
            text=letter if letter != "?" else "—",
            fg=GREEN if letter != "?" else GRAY)
        for cv_widget, val, col in [
            (self.t1_conf_cv, conf, CYAN),
            (self.t1_hold_cv, hold, GREEN)
        ]:
            cv_widget.delete("all")
            w = 350
            fw = int(w * val)
            cv_widget.create_rectangle(0, 0, w, 18, fill=PANEL, outline="")
            cv_widget.create_rectangle(0, 0, fw, 18, fill=col, outline="")

    def _t1_clear(self):
        self.current_word.clear()
        self.words_spelled.clear()
        self.sentence_text = ""
        self.t1_sentence_lbl.config(text="")
        self.t1_buf_lbl.config(text="—")
        self.t1_ai_lbl.config(text="")
        self._status("Cleared")

    def _t1_del_word(self):
        if self.current_word:
            self.current_word.clear()
        elif self.words_spelled:
            self.words_spelled.pop()
        self._t1_update_buf()

    def _t1_send_to_hand(self):
        text = self.sentence_text.strip()
        if not text:
            self._status("No sentence yet"); return
        if not self.serial_conn or not self.serial_conn.is_open:
            self._status("Not connected to Arduino"); return
        threading.Thread(target=self._transmit, args=(text,), daemon=True).start()

    # =========================================================================
    #  Tab 2: Type → Hand
    # =========================================================================

    def _t2_send(self):
        text = self.t2_input.get("1.0", "end").strip().upper()
        if not text:
            self._status("Nothing to send"); return
        if not self.serial_conn or not self.serial_conn.is_open:
            messagebox.showwarning("Not connected", "Connect to Arduino first.")
            return
        threading.Thread(target=self._transmit, args=(text,), daemon=True).start()

    # =========================================================================
    #  Tab 3: Gesture Library
    # =========================================================================

    def _lib_refresh_list(self):
        self.lib_listbox.delete(0, "end")
        for e in self.library:
            self.lib_listbox.insert("end", f"  {e['name']}")

    def _lib_on_select(self, evt):
        sel = self.lib_listbox.curselection()
        if not sel: return
        idx = sel[0]; e = self.library[idx]
        self.lib_detail_lbl.config(
            text=f"Name: {e['name']}   |   Frames: {len(e['sequence'])}   |   Meaning: {e['meaning']}")

    def _lib_delete(self):
        sel = self.lib_listbox.curselection()
        if not sel: return
        name = self.library[sel[0]]["name"]
        if messagebox.askyesno("Delete", f"Delete '{name}'?"):
            self.library = delete_entry(self.library, name)
            self._lib_refresh_list()
            self.lib_detail_lbl.config(text="")
            self._status(f"Deleted: {name}")

    def _lib_record_live(self):
        name    = self.lib_name_var.get().strip()
        meaning = self.lib_meaning_txt.get("1.0", "end").strip()
        if not name or not meaning:
            messagebox.showwarning("Missing info", "Enter both a name and a meaning first.")
            return
        dur = self.lib_dur_var.get()
        self.lib_status_lbl.config(text=f"Opening camera… recording for {dur}s after countdown.", fg=YELLOW)
        self.root.update()
        def _run():
            seq = record_from_camera(duration_seconds=dur, countdown=3)
            if seq:
                self.library = add_entry(self.library, name, meaning, seq)
                self.root.after(0, self._lib_refresh_list)
                self.root.after(0, lambda: self.lib_status_lbl.config(
                    text=f"✓ Saved '{name}' — {len(seq)} frames captured.", fg=GREEN))
                self.root.after(0, lambda: self._status(f"Library: added '{name}'"))
            else:
                self.root.after(0, lambda: self.lib_status_lbl.config(
                    text="Recording cancelled or no hand detected.", fg=RED))
        threading.Thread(target=_run, daemon=True).start()

    def _lib_import_video(self):
        name    = self.lib_name_var.get().strip()
        meaning = self.lib_meaning_txt.get("1.0", "end").strip()
        if not name or not meaning:
            messagebox.showwarning("Missing info", "Enter both a name and a meaning first.")
            return
        path = filedialog.askopenfilename(
            title="Select gesture video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")])
        if not path: return
        self.lib_status_lbl.config(text="Extracting landmarks from video…", fg=YELLOW)
        self.root.update()
        def _run():
            try:
                seq = extract_from_video(path)
                if seq:
                    self.library = add_entry(self.library, name, meaning, seq)
                    self.root.after(0, self._lib_refresh_list)
                    self.root.after(0, lambda: self.lib_status_lbl.config(
                        text=f"✓ Saved '{name}' from video — {len(seq)} frames.", fg=GREEN))
                else:
                    self.root.after(0, lambda: self.lib_status_lbl.config(
                        text="No hand detected in video.", fg=RED))
            except Exception as ex:
                self.root.after(0, lambda: self.lib_status_lbl.config(text=f"Error: {ex}", fg=RED))
        threading.Thread(target=_run, daemon=True).start()

    # =========================================================================
    #  Tab 4: Gesture Match
    # =========================================================================

    def _t4_start(self):
        if self.cam4_running: return
        if not self.library:
            messagebox.showinfo("No gestures", "Add gestures in the Library tab first.")
            return
        self.cam4_running = True
        self.t4_start.config(state="disabled"); self.t4_stop.config(state="normal")
        self.t4_status_lbl.config(text="Watching… show your gesture sequence now.", fg=CYAN)
        threading.Thread(target=self._t4_loop, daemon=True).start()

    def _t4_stop(self):
        self.cam4_running = False
        self.t4_start.config(state="normal"); self.t4_stop.config(state="disabled")
        self.t4_status_lbl.config(text="Stopped", fg=GRAY)

    def _t4_loop(self):
        mp_hands   = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        cap = self._open_camera()
        if cap is None:
            self.cam4_running = False
            self.root.after(0, lambda: (self.t4_start.config(state="normal"),
                                        self.t4_stop.config(state="disabled")))
            return

        window_sec    = self.t4_window_var.get()
        window_frames = []
        window_start  = time.time()
        match_cooldown = 0.0

        with mp_hands.Hands(max_num_hands=1,
                             min_detection_confidence=0.78,
                             min_tracking_confidence=0.75) as hands:
            while self.cam4_running and cap.isOpened():
                ok, frame = cap.read()
                if not ok: break
                frame  = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                now    = time.time()

                if result.multi_hand_landmarks:
                    lm_list = landmarks_to_list(result.multi_hand_landmarks[0])
                    window_frames.append(lm_list)
                    mp_drawing.draw_landmarks(
                        frame, result.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(120, 80, 255), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2))

                elapsed   = now - window_start
                time_left = max(0.0, window_sec - elapsed)
                prog_frac = min(elapsed / window_sec, 1.0)

                self.root.after(0, self._t4_update_timer, prog_frac, len(window_frames), time_left)

                if elapsed >= window_sec and now > match_cooldown:
                    match_cooldown = now + 2.0
                    captured       = list(window_frames)
                    window_frames  = []
                    window_start   = now
                    def _match_thread(seq=captured):
                        match = find_best_match(seq, self.library, MATCH_THRESHOLD)
                        self.root.after(0, self._t4_show_result, match)
                    threading.Thread(target=_match_thread, daemon=True).start()

                # Overlay text
                h_f, w_f = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w_f, 50), (10, 10, 20), -1)
                cv2.putText(frame, f"Window: {time_left:.1f}s  |  Frames: {len(window_frames)}",
                            (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 80, 255), 2)

                # ── Render frame inside tkinter canvas ──
                photo = self._frame_to_tkimage(frame)
                def _update_cam4(p=photo):
                    self._t4_photo = p
                    self.t4_cam_canvas.delete("all")
                    self.t4_cam_canvas.create_image(0, 0, anchor="nw", image=p)
                self.root.after(0, _update_cam4)

                time.sleep(0.03)

        cap.release()
        self.cam4_running = False
        self.root.after(0, lambda: (
            self.t4_start.config(state="normal"),
            self.t4_stop.config(state="disabled"),
            self.t4_status_lbl.config(text="Stopped", fg=GRAY)
        ))

    def _t4_update_timer(self, frac, frames, time_left):
        self.t4_timer_cv.delete("all")
        w = 380; fw = int(w * frac)
        self.t4_timer_cv.create_rectangle(0, 0, w, 18, fill=PANEL, outline="")
        col = PURPLE if frac < 0.8 else ORANGE
        self.t4_timer_cv.create_rectangle(0, 0, fw, 18, fill=col, outline="")
        self.t4_frames_lbl.config(
            text=f"{frames} landmark frames captured  |  {time_left:.1f}s remaining")

    def _t4_show_result(self, match):
        if match:
            self.last_match_result = match
            self.t4_match_name.config(text=f"  {match['name']}", fg=ORANGE)
            self.t4_meaning_lbl.config(text=match["meaning"], fg=WHITE)
            self.t4_status_lbl.config(
                text=f"✓ Match: '{match['name']}'  →  sending to hand…", fg=GREEN)
            self._status(f"Matched: {match['name']}")
            threading.Thread(target=self._transmit_meaning,
                             args=(match["meaning"],), daemon=True).start()
        else:
            self.t4_status_lbl.config(text="No match found — try again.", fg=YELLOW)
            self._status("No match")

    def _t4_clear(self):
        self.last_match_result = None
        self.t4_match_name.config(text="—")
        self.t4_meaning_lbl.config(text="")
        self.t4_status_lbl.config(text="Cleared", fg=GRAY)

    def _t4_send_to_hand(self):
        if not self.last_match_result:
            self._status("No match result to send"); return
        meaning = self.last_match_result["meaning"]
        threading.Thread(target=self._transmit_meaning, args=(meaning,), daemon=True).start()
        self._status(f"Sending: {meaning[:40]}")

    # =========================================================================
    #  Shared helpers
    # =========================================================================

    def _status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
