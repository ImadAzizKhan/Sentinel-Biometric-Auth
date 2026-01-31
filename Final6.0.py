import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import time
import csv
import joblib
import numpy as np
import threading
import sys
import pyttsx3
import random
import keyboard
from PIL import Image, ImageTk, ImageDraw
from datetime import datetime

# ==========================================
#   CONFIG & PATHS
# ==========================================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_DATASET = os.path.join(BASE_DIR, "dataset", "faces")
KEY_DATASET = os.path.join(BASE_DIR, "dataset", "keystrokes")
MODEL_DIR = os.path.join(BASE_DIR, "models")
INTRUDER_DIR = os.path.join(BASE_DIR, "dataset", "intruders")
REPORT_FILE = os.path.join(BASE_DIR, "exam_report.txt")

# --- CUSTOMIZE HERE ---
STUDENT_NAMES = "Designed by: Muhammad Imad Aziz Khan, Airad Khan, Jassahib Singh"
ADMIN_PASSWORD = "admin"
FUSION_THRESHOLD = 0.55 

# --- MODERN MATERIAL PALETTE ---
COL_BG = "#f4f4f9"        # Light Grey
COL_PANEL = "#ffffff"     # White
COL_ACCENT = "#74b9ff"    # Sky Blue
COL_BTN_MAIN = "#0984e3"  # Azure
COL_BTN_SUCC = "#00b894"  # Mint
COL_BTN_WARN = "#d63031"  # Pomegranate
COL_TEXT = "#2d3436"      # Dark Grey

DEFAULT_W_FACE = 0.6
DEFAULT_W_KEY = 0.4

FONT_MAIN = ("Verdana", 10)
FONT_HEADER = ("Verdana", 20, "bold")
FONT_SUB = ("Verdana", 12)

SYSTEM_PHRASES = [
    "the quick brown fox", "blue sky green grass", "super fast red car",
    "open the secure door", "security clearance five", "hello world system",
    "keep it secret safe", "the stars are barking", "dogs are shining",
    "the night has fallen"
]

for d in [FACE_DATASET, KEY_DATASET, MODEL_DIR, INTRUDER_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
#        CUSTOM UI ELEMENTS (Curved)
# ==========================================
class RoundedButton(tk.Canvas):
    def __init__(self, parent, width, height, corner_radius, padding, color, text, text_color="white", command=None):
        tk.Canvas.__init__(self, parent, borderwidth=0, 
            relief="flat", highlightthickness=0, bg=parent["bg"])
        self.command = command
        self.color = color
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.text = text
        self.text_color = text_color

        if corner_radius > 0.5 * width: corner_radius = 0.5 * width
        if corner_radius > 0.5 * height: corner_radius = 0.5 * height

        self.rad = 2 * corner_radius
        self.def_image = self.round_rect(width, height, self.rad, color)
        
        self.configure(width=width, height=height)
        self.id = self.create_image(0, 0, image=self.def_image, anchor='nw')
        self.text_id = self.create_text(width/2, height/2, text=self.text, fill=self.text_color, font=("Verdana", 11, "bold"))
        
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def round_rect(self, width, height, rad, fill):
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((0, 0, width, height), radius=rad/2, fill=fill)
        return ImageTk.PhotoImage(image)

    def _on_press(self, event):
        self.move(self.text_id, 1, 1)

    def _on_release(self, event):
        self.move(self.text_id, -1, -1)
        if self.command:
            self.command()

    def _on_enter(self, event):
        self.configure(cursor="hand2")
        
    def _on_leave(self, event):
        self.configure(cursor="arrow")

# ==========================================
#        GLOBAL SHARED STATE
# ==========================================
GLOBAL_STATS = {
    "live_attempts": 0, "live_success": 0, "live_denied": 0,
    "sim_attempts": 0, "last_far": 0.0, "last_frr": 0.0
}

# ==========================================
#            GLOBAL HELPERS
# ==========================================
user_mapping_file = os.path.join(MODEL_DIR, "user_mapping.pkl")
phrase_mapping_file = os.path.join(MODEL_DIR, "user_phrases.pkl")
variance_mapping_file = os.path.join(MODEL_DIR, "user_variance.pkl")

user_mapping = {}
user_phrases = {}
user_variances = {}

def load_global_data():
    global user_mapping, user_phrases, user_variances
    try:
        if os.path.exists(user_mapping_file): user_mapping = joblib.load(user_mapping_file)
        if os.path.exists(phrase_mapping_file): user_phrases = joblib.load(phrase_mapping_file)
        if os.path.exists(variance_mapping_file): user_variances = joblib.load(variance_mapping_file)
    except: pass

def save_mappings():
    try:
        joblib.dump(user_mapping, user_mapping_file)
        joblib.dump(user_phrases, phrase_mapping_file)
        joblib.dump(user_variances, variance_mapping_file)
    except: pass

def get_face_detector():
    if getattr(sys, 'frozen', False): 
        local_path = os.path.join(os.path.dirname(sys.executable), "haarcascade_frontalface_default.xml")
        if os.path.exists(local_path): return cv2.CascadeClassifier(local_path)
    else: 
        base_path = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
        if os.path.exists(local_path): return cv2.CascadeClassifier(local_path)
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def speak(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except: pass
    threading.Thread(target=_speak).start()

def center_window(win, width=None, height=None):
    try:
        win.update_idletasks()
        if width is None: width = win.winfo_width()
        if height is None: height = win.winfo_height()
        x = (win.winfo_screenwidth() // 2) - (width // 2)
        y = (win.winfo_screenheight() // 2) - (height // 2)
        win.geometry(f'{width}x{height}+{x}+{y}')
    except: pass

# =========================================================
#       ADVANCED BIOMETRIC MATH
# =========================================================

def assess_image_quality(gray_img):
    mean_brightness = np.mean(gray_img)
    final_q = 1.0
    if mean_brightness < 40 or mean_brightness > 220: final_q = 0.4
    laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    if laplacian_var < 50: final_q = min(final_q, 0.3)
    status = "OK" if final_q > 0.5 else "POOR"
    return final_q, status

def adaptive_fusion(face_score, key_score, face_quality):
    w_face = DEFAULT_W_FACE
    w_key = DEFAULT_W_KEY
    if face_quality < 0.6: 
        w_face = 0.2
        w_key = 0.8
    final = (face_score * w_face) + (key_score * w_key)
    return final

def calculate_mahalanobis_score(input_vec, profile_mean, profile_std):
    if not input_vec or not profile_mean: return 0.0
    min_len = min(len(input_vec), len(profile_mean))
    if min_len == 0: return 0.0
    
    u = np.array(input_vec[:min_len])
    v = np.array(profile_mean[:min_len])
    
    # Handle Standard Deviation
    if profile_std and len(profile_std) >= min_len:
        s = np.array(profile_std[:min_len])
        # Floor the variance to prevent division by zero or extreme sensitivity
        s[s < 0.05] = 0.05 
    else: 
        s = np.ones_like(u)

    # Calculate Distance
    diff = np.abs(u - v)
    weighted_diff = diff / s
    dist = np.mean(weighted_diff)
    
    # --- BALANCED DECAY ---
    score = 1.0 / (1.0 + (dist * 0.7))
    
    print(f"[DEBUG] Key Distance: {dist:.2f} | Score: {score:.2f}")
    return score

# ==========================================
#            UI POPUPS
# ==========================================
def ask_admin_password_large(parent):
    dialog = tk.Toplevel(parent)
    dialog.title("Authentication")
    dialog.configure(bg=COL_BG)
    center_window(dialog, 600, 350)
    dialog.transient(parent)
    dialog.grab_set()
    result = {"password": None}
    tk.Label(dialog, text="🔒 SECURITY CLEARANCE", font=FONT_HEADER, bg=COL_BG, fg=COL_BTN_MAIN).pack(pady=30)
    entry = tk.Entry(dialog, font=FONT_SUB, show="●", width=20, justify='center', bg="white")
    entry.pack(pady=20)
    entry.focus_set()
    def on_submit(event=None):
        result["password"] = entry.get()
        dialog.destroy()
    entry.bind("<Return>", on_submit)
    tk.Button(dialog, text="AUTHENTICATE", font=FONT_MAIN, bg=COL_BTN_MAIN, fg="white", width=20, command=on_submit).pack(pady=20)
    parent.wait_window(dialog)
    return result["password"]

def ask_username_large(parent):
    dialog = tk.Toplevel(parent)
    dialog.title("New Subject")
    dialog.configure(bg=COL_BG)
    center_window(dialog, 600, 350)
    dialog.transient(parent)
    dialog.grab_set()
    result = {"name": None}
    tk.Label(dialog, text="👤 NEW IDENTITY", font=FONT_HEADER, bg=COL_BG, fg=COL_BTN_MAIN).pack(pady=30)
    entry = tk.Entry(dialog, font=FONT_SUB, width=20, justify='center', bg="white")
    entry.pack(pady=20)
    entry.focus_set()
    def on_submit(event=None):
        val = entry.get().strip()
        if val:
            result["name"] = val
            dialog.destroy()
    entry.bind("<Return>", on_submit)
    tk.Button(dialog, text="PROCEED ➜", font=FONT_MAIN, bg=COL_BTN_SUCC, fg="white", width=20, command=on_submit).pack(pady=20)
    parent.wait_window(dialog)
    return result["name"]

def ask_phrase_mode(parent):
    dialog = tk.Toplevel(parent)
    dialog.title("Protocol Selection")
    dialog.configure(bg=COL_BG)
    center_window(dialog, 700, 400)
    dialog.transient(parent)
    dialog.grab_set()
    result = {"phrase": None}
    tk.Label(dialog, text="⌨ PHRASE SELECTION", font=FONT_HEADER, bg=COL_BG, fg=COL_BTN_MAIN).pack(pady=30)
    def set_random():
        result["phrase"] = random.choice(SYSTEM_PHRASES)
        dialog.destroy()
    def set_custom():
        custom_p = ask_custom_phrase_input(dialog)
        if custom_p:
            result["phrase"] = custom_p
            dialog.destroy()
    btn_frame = tk.Frame(dialog, bg=COL_BG)
    btn_frame.pack(pady=30)
    tk.Button(btn_frame, text="🎲 RANDOMIZE", font=FONT_MAIN, bg=COL_BTN_MAIN, fg="white", width=15, command=set_random).pack(side=tk.LEFT, padx=20)
    tk.Button(btn_frame, text="✍ CUSTOM", font=FONT_MAIN, bg=COL_BTN_SUCC, fg="white", width=15, command=set_custom).pack(side=tk.LEFT, padx=20)
    parent.wait_window(dialog)
    return result["phrase"]

def ask_custom_phrase_input(parent):
    dialog = tk.Toplevel(parent)
    dialog.title("Custom Input")
    dialog.configure(bg=COL_BG)
    center_window(dialog, 600, 350)
    dialog.grab_set()
    result = {"text": None}
    tk.Label(dialog, text="ENTER PHRASE (Min 15 chars)", font=("Verdana", 16, "bold"), bg=COL_BG, fg=COL_TEXT).pack(pady=30)
    entry = tk.Entry(dialog, font=FONT_SUB, width=30, bg="white")
    entry.pack(pady=10)
    entry.focus_set()
    def on_submit(event=None):
        t = entry.get().strip().lower()
        if len(t) < 15: return
        result["text"] = t
        dialog.destroy()
    entry.bind("<Return>", on_submit)
    tk.Button(dialog, text="CONFIRM", font=FONT_MAIN, bg=COL_BTN_MAIN, fg="white", command=on_submit).pack(pady=20)
    parent.wait_window(dialog)
    return result["text"]

def ask_user_selection(parent, users):
    dialog = tk.Toplevel(parent)
    dialog.title("Identify Yourself")
    dialog.configure(bg=COL_BG)
    center_window(dialog, 600, 300)
    dialog.transient(parent)
    dialog.grab_set()
    result = {"user": None}
    tk.Label(dialog, text="WHO ARE YOU?", font=FONT_HEADER, bg=COL_BG, fg=COL_BTN_MAIN).pack(pady=20)
    
    combo = ttk.Combobox(dialog, values=users, font=FONT_SUB, state="readonly", width=25)
    combo.pack(pady=20)
    if users: combo.current(0)
    
    def on_submit():
        result["user"] = combo.get()
        dialog.destroy()
        
    tk.Button(dialog, text="VERIFY ME ➜", font=FONT_MAIN, bg=COL_BTN_SUCC, fg="white", width=20, command=on_submit).pack(pady=20)
    parent.wait_window(dialog)
    return result["user"]

# ==========================================
#            PART 1: LIVE SYSTEM
# ==========================================
class LiveSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Biometric Defense System (LIVE)")
        try: self.root.state('zoomed')
        except: self.root.attributes('-fullscreen', True)
        self.root.configure(bg=COL_BG)
        self.container = tk.Frame(root, bg=COL_BG)
        self.container.pack(fill="both", expand=True)
        load_global_data()
        self.show_login_screen()

        # LIVE HUD
        self.hud_frame = tk.Frame(self.root, bg="#111", height=40)
        self.hud_frame.place(relx=0, rely=0, relwidth=1)
        self.hud_label = tk.Label(self.hud_frame, text="Initializing...", font=("Consolas", 11), bg="#111", fg=COL_ACCENT)
        self.hud_label.pack(pady=10)
        self.update_hud()

        # DEBUG LOG for Live System
        self.debug_frame = tk.Frame(self.root, bg="black", height=100)
        self.debug_frame.pack(side="bottom", fill="x")
        self.txt_log = tk.Text(self.debug_frame, bg="black", fg="#00ff00", font=("Consolas", 9), height=5)
        self.txt_log.pack(fill="both")

    def log(self, msg):
        print(f"[LIVE LOG] {msg}")
        try:
            if self.txt_log.winfo_exists():
                self.txt_log.insert("end", f"> {msg}\n")
                self.txt_log.see("end")
        except: pass

    def update_hud(self):
        if not self.hud_label.winfo_exists(): return
        text = (f" 🔴 LIVE: {GLOBAL_STATS['live_attempts']} (✅{GLOBAL_STATS['live_success']} / ⛔{GLOBAL_STATS['live_denied']})    |    "
                f" 📊 STATS: FAR {GLOBAL_STATS['last_far']:.1f}% / FRR {GLOBAL_STATS['last_frr']:.1f}%")
        self.hud_label.config(text=text)
        self.root.after(1000, self.update_hud)

    def show_login_screen(self):
        for w in self.container.winfo_children(): w.destroy()
        f = tk.Frame(self.container, bg=COL_BG)
        f.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(f, text="SENTINEL BIO-SUITE", font=FONT_HEADER, bg=COL_BG, fg=COL_TEXT).pack(pady=10)
        tk.Label(f, text="ADVANCED MULTIMODAL AUTHENTICATION", font=FONT_SUB, bg=COL_BG, fg=COL_ACCENT).pack(pady=(0, 40))
        
        RoundedButton(f, width=280, height=50, corner_radius=25, padding=0, color=COL_BTN_MAIN, text="🔓 AUTHENTICATE", command=self.start_auth).pack(pady=10)
        RoundedButton(f, width=280, height=50, corner_radius=25, padding=0, color=COL_BTN_SUCC, text="👤 ENROLL USER", command=self.start_enroll).pack(pady=10)
        RoundedButton(f, width=280, height=50, corner_radius=25, padding=0, color="#636e72", text="🛠 ADMIN PANEL", command=self.open_admin).pack(pady=10)
        RoundedButton(f, width=280, height=50, corner_radius=25, padding=0, color=COL_BTN_WARN, text="🚪 EXIT SYSTEM", command=self.go_home).pack(pady=30)

    def go_home(self):
        for w in self.root.winfo_children(): w.destroy()
        Launcher(self.root)

    def check_duplicate_face(self):
        try:
            if not os.path.exists(os.path.join(MODEL_DIR, "face_model.yml")): return None
            face_model = cv2.face.LBPHFaceRecognizer_create()
            face_model.read(os.path.join(MODEL_DIR, "face_model.yml"))
            inv_map = {v:k for k,v in user_mapping.items()}
            cam = cv2.VideoCapture(0)
            detected_user = None
            face_cascade = get_face_detector()
            for _ in range(20):
                ret, frame = cam.read()
                if not ret: continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                    label, conf = face_model.predict(roi)
                    if conf < 60:
                        detected_user = inv_map.get(label)
                        break
                if detected_user: break
            cam.release()
            return detected_user
        except: return None

    def capture_keystroke_sequence(self, phrase):
        popup = tk.Toplevel(self.root)
        popup.title("Keystroke Dynamics")
        popup.configure(bg=COL_BG)
        try: popup.state('zoomed')
        except: popup.attributes('-fullscreen', True)
        container = tk.Frame(popup, bg=COL_BG)
        container.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(container, text="TYPE THE FOLLOWING PHRASE:", font=FONT_SUB, bg=COL_BG, fg=COL_TEXT).pack(pady=10)
        tk.Label(container, text=phrase, font=("Consolas", 28, "bold"), bg=COL_BG, fg=COL_ACCENT).pack(pady=20)
        lbl = tk.Label(container, text="Start typing...", font=("Consolas", 20), fg=COL_TEXT, bg=COL_BG)
        lbl.pack(pady=20)
        popup.update()
        while keyboard.is_pressed('enter'): pass
        current_text = ""
        press_times, timings = {}, []
        while True:
            event = keyboard.read_event()
            if event.event_type == "down":
                if event.name == 'enter': break
                elif event.name == 'backspace':
                    current_text = current_text[:-1]
                    lbl.config(text=current_text)
                    popup.update()
                elif len(event.name) == 1 or event.name == 'space':
                    char = " " if event.name == 'space' else event.name
                    current_text += char
                    lbl.config(text=current_text)
                    popup.update()
                press_times[event.name] = time.time()
            elif event.event_type == "up":
                if event.name in press_times:
                    timings.append(time.time() - press_times[event.name])
        popup.destroy()
        return timings

    def run_enrollment_logic(self, user_id, phrase):
        try:
            self.log("Starting Enrollment...")
            speak("System check initiated.")
            duplicate = self.check_duplicate_face()
            if duplicate:
                messagebox.showerror("Conflict", f"User '{duplicate}' already exists.")
                return
            
            save_path = os.path.join(FACE_DATASET, user_id)
            os.makedirs(save_path, exist_ok=True)
            user_phrases[user_id] = phrase
            save_mappings()
            
            speak("Face enrollment starting.")
            cam = cv2.VideoCapture(0)
            face_cascade = get_face_detector()
            
            poses = [("LOOK CENTER", 20), ("MOVE CLOSER", 15), ("MOVE BACK", 15)]
            total_photos = 0
            
            for instruction, count in poses:
                speak(f"{instruction}")
                time.sleep(1) 
                c = 0
                while c < count:
                    ret, frame = cam.read()
                    if not ret: continue
                    frame = cv2.flip(frame, 1) 
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) > 0:
                        (x,y,w,h) = faces[0]
                        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                        total_photos += 1
                        c += 1
                        cv2.imwrite(os.path.join(save_path, f"{total_photos}.jpg"), face)
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    
                    cv2.putText(frame, f"{instruction}: {c}/{count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Enrollment", frame)
                    cv2.waitKey(1)
            
            cam.release()
            cv2.destroyAllWindows()
            
            samples = []
            for i in range(3):
                speak(f"Keystroke sample {i+1} of 3.")
                t = self.capture_keystroke_sequence(phrase)
                if t: samples.append(t)
            
            if not samples: return
            
            with open(os.path.join(KEY_DATASET, f"{user_id}.csv"), "w", newline="") as f:
                csv.writer(f).writerows(samples)
            
            max_len = max(len(s) for s in samples)
            padded_samples = [s + [0]*(max_len-len(s)) for s in samples]
            user_variances[user_id] = np.std(padded_samples, axis=0).tolist()
            save_mappings()

            # --- FIX: CORRECT ID ASSIGNMENT ---
            faces, labels = [], []
            if user_mapping:
                curr_idx = max(user_mapping.values()) + 1
            else:
                curr_idx = 0
                
            all_users = os.listdir(FACE_DATASET)
            
            for u in all_users:
                if u not in user_mapping:
                    user_mapping[u] = curr_idx
                    curr_idx += 1
            
            for u in all_users:
                if u in user_mapping:
                    uid = user_mapping[u]
                    p = os.path.join(FACE_DATASET, u)
                    for img_name in os.listdir(p):
                        try:
                            img = cv2.imread(os.path.join(p, img_name), cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                faces.append(cv2.resize(img, (200, 200)))
                                labels.append(uid)
                        except: pass
            
            if faces:
                fm = cv2.face.LBPHFaceRecognizer_create()
                fm.train(faces, np.array(labels))
                fm.save(os.path.join(MODEL_DIR, "face_model.yml"))
                save_mappings()
            
            messagebox.showinfo("Enrollment", "User successfully registered.")
            speak("Enrollment complete.")
        except Exception as e:
            self.log(f"Enroll Error: {e}")
            messagebox.showerror("Error", str(e))

            
    def run_auth_logic(self):
        try:
            self.log("Authenticating...")
            if not os.path.exists(os.path.join(MODEL_DIR, "face_model.yml")): 
                messagebox.showerror("Error", "System is empty.")
                return
            
            # --- STEP 1: ASK WHO YOU ARE (1:1 Verification) ---
            known_users = list(user_mapping.keys())
            claimed_user = ask_user_selection(self.root, known_users)
            if not claimed_user: return
            
            face_model = cv2.face.LBPHFaceRecognizer_create()
            face_model.read(os.path.join(MODEL_DIR, "face_model.yml"))
            # Get ID for verification
            claimed_id = user_mapping.get(claimed_user)

            speak(f"Verifying {claimed_user}. Please look at the camera.")
            cam = cv2.VideoCapture(0)
            
            # --- STEP 2: LIVENESS CHECK ---
            task = "MOVE CLOSER TO CAMERA"
            speak(task)
            challenge_passed = False
            start_time = time.time()
            initial_width = 0
            
            while time.time() - start_time < 3:
                ret, frame = cam.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = get_face_detector().detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (x,y,w,h) = faces[0]
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
                    initial_width = w
                    cv2.putText(frame, "HOLD STILL...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("LIVENESS", frame)
                cv2.waitKey(1)
            
            if initial_width == 0:
                cam.release(); cv2.destroyAllWindows()
                self.log("Liveness Fail: No face found initially.")
                return

            start_time = time.time()
            while time.time() - start_time < 5: 
                ret, frame = cam.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = get_face_detector().detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    (x,y,w,h) = faces[0]
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    if w > initial_width * 1.15:
                        challenge_passed = True
                        break
                cv2.putText(frame, f"{task}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("LIVENESS", frame)
                cv2.waitKey(1)
            cv2.destroyWindow("LIVENESS")
            
            if not challenge_passed:
                speak("Liveness Check Failed.")
                cam.release()
                self.log("Liveness Failed: Did not move closer.")
                return
            self.log("Liveness Passed.")

            # --- STEP 3: FACE VERIFICATION (1:1) ---
            face_conf = 100.0
            face_quality_score = 0.0
            captured_face_img = None
            is_face_verified = False
            
            for _ in range(20):
                ret, frame = cam.read()
                if not ret: continue
                frame = cv2.flip(frame, 1) 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = get_face_detector().detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (x,y,w_face,h_face) = faces[0]
                    face_img = cv2.resize(gray[y:y+h_face, x:x+w_face], (200, 200))
                    face_quality_score, status = assess_image_quality(face_img)
                    label, conf = face_model.predict(face_img)
                    captured_face_img = face_img
                    
                    # 1:1 CHECK: Does the label match the ID we are looking for?
                    if label == claimed_id:
                        face_conf = conf
                        is_face_verified = True
                    else:
                        face_conf = 100.0 # Force bad score if ID mismatch
                        
                    break
                cv2.imshow("Scanning", frame)
                cv2.waitKey(1)
            cam.release()
            cv2.destroyAllWindows()

            norm_face_score = max(0, (100 - face_conf) / 100)
            if not is_face_verified: norm_face_score = 0.0 # Double check
            
            self.log(f"Face Verification ({claimed_user}): {norm_face_score:.2f}")
            
            # --- STEP 4: KEYSTROKE VERIFICATION ---
            phrase = user_phrases.get(claimed_user, "unknown")
            speak(f"Verified face. Please type pass-phrase.")
            timings = self.capture_keystroke_sequence(phrase)
            if not timings: return
            
            profile_mean = []
            key_score = 0.0
            try:
                p_path = os.path.join(KEY_DATASET, f"{claimed_user}.csv")
                with open(p_path, 'r') as f:
                    rows = [list(map(float, r)) for r in csv.reader(f) if r]
                min_len = min(len(r) for r in rows)
                trimmed_rows = [r[:min_len] for r in rows]
                profile_mean = np.mean(trimmed_rows, axis=0).tolist()
                
                profile_std = user_variances.get(claimed_user, [])
                key_score = calculate_mahalanobis_score(timings, profile_mean, profile_std)
                self.log(f"Keystroke Score: {key_score:.2f}")
            except: pass

            # --- STEP 5: FUSION ---
            final = adaptive_fusion(norm_face_score, key_score, face_quality_score)
            self.log(f"Final Fusion Score: {final:.2f}")

            GLOBAL_STATS["live_attempts"] += 1
            if final >= FUSION_THRESHOLD:
                GLOBAL_STATS["live_success"] += 1
                speak("Access Granted.")
                self.show_dashboard(claimed_user, norm_face_score, key_score, final, profile_mean, timings, "Passed", face_quality_score)
                # Auto-update template if high score
                if final > 0.90 and captured_face_img is not None:
                    save_path = os.path.join(FACE_DATASET, claimed_user)
                    count = len(os.listdir(save_path)) + 1
                    cv2.imwrite(os.path.join(save_path, f"auto_update_{count}.jpg"), captured_face_img)
            else:
                GLOBAL_STATS["live_denied"] += 1
                speak("Access Denied.")
                
                # --- STEP 6: INTRUDER TRAP ---
                if captured_face_img is not None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    intruder_path = os.path.join(INTRUDER_DIR, f"{claimed_user}_ATTACK_{ts}.jpg")
                    cv2.imwrite(intruder_path, captured_face_img)
                    self.log(f"Intruder snapshot saved: {intruder_path}")
                
                self.show_intruder_screen(f"Low Score: {final:.2f}", norm_face_score, key_score, final, None)
        except Exception as e: 
            print(e)
            messagebox.showerror("Error", str(e))

    def draw_graph(self, parent, profile, sample):
        c = tk.Canvas(parent, width=500, height=200, bg=COL_BG, highlightthickness=0)
        c.pack(pady=20)
        if not profile or not sample: return
        w, h = 500, 200
        limit = min(len(profile), len(sample))
        if limit == 0: return
        bar_w = w / limit
        try: max_val = max(max(profile[:limit]), max(sample[:limit])) + 0.1
        except: max_val = 1.0
        c.create_text(50, 20, text="Profile (Blue)", fill=COL_ACCENT, font=("Arial", 10))
        c.create_text(150, 20, text="Sample (Red)", fill=COL_BTN_WARN, font=("Arial", 10))
        for i in range(limit):
            x = i * bar_w
            h_p = (profile[i] / max_val) * (h - 40)
            c.create_rectangle(x+2, h, x+bar_w/2-2, h-h_p, fill=COL_ACCENT, outline="")
            h_s = (sample[i] / max_val) * (h - 40)
            c.create_rectangle(x+bar_w/2+2, h, x+bar_w-2, h-h_s, fill=COL_BTN_WARN, outline="")

    def show_dashboard(self, user, fs, ks, final, profile, sample, liveness, quality):
        for w in self.container.winfo_children(): w.destroy()
        f = tk.Frame(self.container, bg=COL_BG)
        f.pack(fill='both', expand=True)
        tk.Label(f, text="ACCESS GRANTED", font=("Verdana", 36, "bold"), bg=COL_BG, fg=COL_BTN_SUCC).pack(pady=20)
        tk.Label(f, text=f"Welcome, {user}", font=FONT_HEADER, bg=COL_BG, fg=COL_TEXT).pack()
        
        stats = tk.Frame(f, bg=COL_BG)
        stats.pack(pady=20)
        tk.Label(stats, text=f"Face: {fs:.2f}", font=FONT_SUB, fg=COL_ACCENT, bg=COL_BG).grid(row=0, column=0, padx=20)
        tk.Label(stats, text=f"Key: {ks:.2f}", font=FONT_SUB, fg=COL_BTN_WARN, bg=COL_BG).grid(row=0, column=1, padx=20)
        tk.Label(stats, text=f"FUSION: {final:.2f}", font=FONT_HEADER, fg="#fdcb6e", bg=COL_BG).grid(row=0, column=2, padx=20)
        
        tk.Label(f, text=f"Liveness: {liveness} | Quality: {quality:.2f}", font=("Consolas", 12), bg=COL_BG, fg="#b2bec3").pack()
        self.draw_graph(f, profile, sample)
        RoundedButton(f, width=200, height=45, corner_radius=20, padding=0, color=COL_BTN_WARN, text="🚪 LOGOUT", command=self.logout).pack(pady=30)

    def show_intruder_screen(self, reason, fs, ks, final, img_path):
        for w in self.container.winfo_children(): w.destroy()
        self.container.configure(bg=COL_BTN_WARN)
        f = tk.Frame(self.container, bg=COL_BTN_WARN)
        f.pack(fill='both', expand=True)
        tk.Label(f, text="ACCESS DENIED", font=("Verdana", 36, "bold"), bg=COL_BTN_WARN, fg="white").pack(pady=20)
        tk.Label(f, text=reason, font=FONT_HEADER, bg=COL_BTN_WARN, fg="#fab1a0").pack()
        score_text = f"FUSION SCORE: {final:.2f} (Required: {FUSION_THRESHOLD})"
        tk.Label(f, text=score_text, font=("Consolas", 18, "bold"), bg=COL_BTN_WARN, fg="yellow").pack(pady=20)
        
        RoundedButton(f, width=200, height=45, corner_radius=20, padding=0, color="white", text="RETRY", text_color=COL_BTN_WARN, command=self.logout).pack(pady=30)

    def open_admin(self):
        pwd = ask_admin_password_large(self.root)
        if pwd == ADMIN_PASSWORD: AdminPanel(self.root)
        else: messagebox.showerror("Error", "Wrong Password")

    def logout(self):
        self.container.configure(bg=COL_BG)
        self.show_login_screen()

    def start_auth(self): threading.Thread(target=self.run_auth_logic).start()
    def start_enroll(self):
        pwd = ask_admin_password_large(self.root)
        if pwd != ADMIN_PASSWORD: return
        uid = ask_username_large(self.root)
        if uid: 
            sel = ask_phrase_mode(self.root)
            if sel: threading.Thread(target=self.run_enrollment_logic, args=(uid, sel)).start()

# ==========================================
#            PART 2: SIMULATION SYSTEM
# ==========================================
class SimSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Biometric Sandbox (SIMULATION)")
        try: self.root.state('zoomed') 
        except: self.root.attributes('-fullscreen', True)
        self.root.configure(bg=COL_BG)
        self.users = []
        try: 
            if os.path.exists(FACE_DATASET):
                self.users = sorted([d for d in os.listdir(FACE_DATASET) if os.path.isdir(os.path.join(FACE_DATASET, d))])
        except: pass
        self.face_model = cv2.face.LBPHFaceRecognizer_create()
        self.test_data = {}; self.ref_images = {}; self.user_key_means = {}; self.user_key_stds = {}
        self.is_loaded = False
        self.create_widgets()
        self.root.update() 
        threading.Thread(target=self.prepare_data, daemon=True).start()

    def create_widgets(self):
        header = tk.Frame(self.root, bg="#111", height=50)
        header.pack(side=tk.TOP, fill="x")
        self.hud_label = tk.Label(header, text="INIT...", font=("Consolas", 10), bg="#111", fg=COL_ACCENT)
        self.hud_label.pack(side=tk.LEFT, padx=20)
        self.update_hud()
        
        tk.Button(header, text="❌ EXIT", bg=COL_BTN_WARN, fg="white", font=("Arial", 10, "bold"), command=self.go_home).pack(side="right", padx=10)
        
        ctrl = tk.Frame(self.root, bg=COL_PANEL, height=80); ctrl.pack(side=tk.TOP, fill="x")
        c_frame = tk.Frame(ctrl, bg=COL_PANEL); c_frame.pack(pady=15)
        
        tk.Label(c_frame, text="FACE:", font=("Verdana", 10, "bold"), bg=COL_PANEL, fg="black").pack(side="left", padx=5)
        self.combo_face = ttk.Combobox(c_frame, state="readonly", width=12); self.combo_face.pack(side="left", padx=5)
        
        tk.Label(c_frame, text="KEY:", font=("Verdana", 10, "bold"), bg=COL_PANEL, fg="black").pack(side="left", padx=5)
        self.combo_key = ttk.Combobox(c_frame, state="readonly", width=12); self.combo_key.pack(side="left", padx=5)
        
        tk.Label(c_frame, text="➜ ATTACKS ➜", font=("Verdana", 10), bg=COL_PANEL, fg="#aaa").pack(side="left", padx=5)
        
        tk.Label(c_frame, text="TARGET:", font=("Verdana", 10, "bold"), bg=COL_PANEL, fg=COL_ACCENT).pack(side="left", padx=5)
        self.combo_target = ttk.Combobox(c_frame, state="readonly", width=12); self.combo_target.pack(side="left", padx=5)
        
        self.btn_run_frame = tk.Frame(c_frame, bg=COL_PANEL)
        self.btn_run_frame.pack(side="left", padx=20)
        self.btn_run = RoundedButton(self.btn_run_frame, width=180, height=40, corner_radius=20, padding=0, 
                                     color=COL_BTN_SUCC, text="▶ RUN SIM", command=self.run_manual_test)
        self.btn_run.pack()
        
        main_body = tk.Frame(self.root, bg=COL_BG); main_body.pack(side=tk.TOP, fill="both", expand=True, padx=20, pady=10)
        
        # Left (Face)
        left_pane = tk.Frame(main_body, bg=COL_PANEL, width=500, bd=1, relief="solid"); left_pane.pack(side="left", fill="both", expand=True, padx=5)
        tk.Label(left_pane, text=f"FACIAL ANALYSIS", font=("Verdana", 12, "bold"), bg=COL_PANEL, fg="black").pack(fill="x", pady=5)
        face_con = tk.Frame(left_pane, bg=COL_PANEL); face_con.pack(expand=True)
        self.lbl_img_input = tk.Label(face_con, bg="black", width=180, height=180); self.lbl_img_input.pack(side="left", padx=10)
        self.lbl_img_ref = tk.Label(face_con, bg="black", width=180, height=180); self.lbl_img_ref.pack(side="left", padx=10)
        self.lbl_face_status = tk.Label(left_pane, text="Score: 0.00", font=("Verdana", 20, "bold"), bg=COL_PANEL, fg="#555"); self.lbl_face_status.pack(pady=20)
        
        # Right (Keys)
        right_pane = tk.Frame(main_body, bg=COL_PANEL, width=500, bd=1, relief="solid"); right_pane.pack(side="right", fill="both", expand=True, padx=5)
        tk.Label(right_pane, text=f"KEYSTROKE DYNAMICS", font=("Verdana", 12, "bold"), bg=COL_PANEL, fg="black").pack(fill="x", pady=5)
        self.graph_canvas = tk.Canvas(right_pane, bg=COL_BG, height=200, highlightthickness=0); self.graph_canvas.pack(fill="x", padx=10, pady=20)
        self.lbl_key_status = tk.Label(right_pane, text="Score: 0.00", font=("Verdana", 20, "bold"), bg=COL_PANEL, fg="#555"); self.lbl_key_status.pack(pady=20)
        
        bottom_panel = tk.Frame(self.root, bg=COL_BG, height=150); bottom_panel.pack(side="bottom", fill="x", padx=20, pady=10)
        bottom_panel.pack_propagate(False)
        self.txt_log = tk.Text(bottom_panel, bg="black", fg="#00ff00", font=("Consolas", 10)); self.txt_log.pack(fill="both", expand=True)

    def update_hud(self):
        if not self.hud_label.winfo_exists(): return
        # Calculate stats
        t_sim = GLOBAL_STATS["sim_attempts"]
        t_live = GLOBAL_STATS["live_attempts"]
        far = GLOBAL_STATS["last_far"]
        frr = GLOBAL_STATS["last_frr"]
        
        text = (f" 📡 SIMULATIONS: {t_sim} | LIVE ATTACKS: {t_live}    |    "
                f" 📊 METRICS: FAR {far:.1f}% / FRR {frr:.1f}%")
        self.hud_label.config(text=text)
        self.root.after(1000, self.update_hud)

    def go_home(self):
        for w in self.root.winfo_children(): w.destroy()
        Launcher(self.root)

    def log(self, msg):
        if not self.root.winfo_exists(): return
        self.root.after(0, lambda: self._log_internal(msg))

    def _log_internal(self, msg):
        try:
            self.txt_log.insert("end", f"> {msg}\n"); self.txt_log.see("end")
        except: pass

    def prepare_data(self):
        try:
            self.log("Loading datasets...")
            X_faces, y_faces = [], []
            X_keys, y_keys = [], [] # Fixed Initialization
            
            self.id_map = {u: i for i, u in enumerate(self.users)}; self.inv_map = {i: u for u, i in self.id_map.items()}
            
            if not self.users: 
                self.log("ERROR: No dataset found.")
                return

            for user in self.users:
                uid = self.id_map[user]
                self.test_data[user] = {'faces':[], 'keys':[]}; self.ref_images[user] = None; train_keys_temp = []
                f_path = os.path.join(FACE_DATASET, user)
                if os.path.exists(f_path):
                    imgs = [f for f in sorted(os.listdir(f_path)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    for i, img_name in enumerate(imgs):
                        try:
                            path = os.path.join(f_path, img_name)
                            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                            if img is None: continue
                            img = cv2.resize(img, (200, 200))
                            if i < 5: 
                                X_faces.append(img); y_faces.append(uid)
                                if self.ref_images[user] is None: self.ref_images[user] = path
                            else: self.test_data[user]['faces'].append(path)
                        except: pass
                    if not self.test_data[user]['faces'] and self.ref_images[user]:
                        self.test_data[user]['faces'].append(self.ref_images[user])
                
                k_path = os.path.join(KEY_DATASET, f"{user}.csv")
                if os.path.exists(k_path):
                    try:
                        with open(k_path, 'r') as f:
                            reader = csv.reader(f); rows = [list(map(float, r)) for r in list(reader) if r]
                        if rows:
                            split = len(rows) // 2
                            for i, row in enumerate(rows):
                                if i < split: train_keys_temp.append(row)
                                else: self.test_data[user]['keys'].append(row)
                            if not self.test_data[user]['keys'] and train_keys_temp:
                                 self.test_data[user]['keys'].append(train_keys_temp[0])
                    except: pass
                
                if train_keys_temp:
                    max_l = max(len(x) for x in train_keys_temp)
                    padded = [x + [0]*(max_l - len(x)) for x in train_keys_temp]
                    self.user_key_means[user] = np.mean(padded, axis=0).tolist()
                    self.user_key_stds[user] = np.std(padded, axis=0).tolist()
                else: 
                    self.user_key_means[user] = []
                    self.user_key_stds[user] = []
            
            if not X_faces:
                self.log("ERROR: No face data loaded."); return

            self.face_model.train(X_faces, np.array(y_faces))
            self.log("SYSTEM READY. Select users and click Run."); self.is_loaded = True
            valid = [u for u in self.users if self.ref_images.get(u)]
            self.root.after(0, lambda: self.setup_dropdowns(valid))
        except Exception as e: self.log(f"Error: {e}")

    def setup_dropdowns(self, users):
        try:
            self.combo_face['values'] = users; self.combo_key['values'] = users; self.combo_target['values'] = users
            if users: self.combo_face.current(0); self.combo_key.current(0); self.combo_target.current(0)
        except: pass

    def display_img(self, label, path):
        try:
            img = Image.open(path).resize((180, 180)); imgtk = ImageTk.PhotoImage(image=img)
            label.configure(image=imgtk); label.image = imgtk
        except: pass

    def draw_graph(self, input_vec, target_vec):
        c = self.graph_canvas; c.delete("all"); w = c.winfo_width(); h = c.winfo_height()
        limit = min(len(input_vec), len(target_vec)); 
        if limit == 0: return
        bar_w = w / limit
        try: max_val = max(max(input_vec[:limit]), max(target_vec[:limit])) + 0.1
        except: max_val = 1.0
        for i in range(limit):
            x = i * bar_w
            h_t = (target_vec[i] / max_val) * (h - 20)
            c.create_rectangle(x+2, h, x+bar_w/2-2, h-h_t, fill=COL_ACCENT, outline="")
            h_i = (input_vec[i] / max_val) * (h - 20); 
            c.create_rectangle(x+bar_w/2+2, h, x+bar_w-2, h-h_i, fill=COL_BTN_WARN, outline="")

    def run_manual_test(self):
        if not self.is_loaded: return
        face_src = self.combo_face.get(); key_src = self.combo_key.get(); target = self.combo_target.get()
        if not face_src or not key_src or not target: return
        threading.Thread(target=self.execute_logic, args=(face_src, key_src, target), daemon=True).start()

    def execute_logic(self, face_src, key_src, target):
        try:
            self.log("-" * 40); self.log(f"Simulating: {face_src} (Face) + {key_src} (Key) -> {target}")
            input_face = random.choice(self.test_data[face_src]['faces']) if self.test_data[face_src]['faces'] else None
            ref_face = self.ref_images[target]
            self.root.after(0, lambda: self.display_img(self.lbl_img_input, input_face))
            self.root.after(0, lambda: self.display_img(self.lbl_img_ref, ref_face))
            
            input_keys = random.choice(self.test_data[key_src]['keys']) if self.test_data[key_src]['keys'] else None
            target_mean = self.user_key_means[target]
            target_std = self.user_key_stds[target]
            
            g_len = min(len(input_keys), len(target_mean)); self.root.after(0, lambda: self.draw_graph(input_keys[:g_len], target_mean[:g_len]))
            
            img = cv2.imread(input_face, cv2.IMREAD_GRAYSCALE); img = cv2.resize(img, (200, 200))
            label, conf = self.face_model.predict(img)
            face_score = max(0, (100 - conf) / 100) if label == self.id_map[target] else 0.0
            
            f_col = COL_BTN_SUCC if face_score > 0.6 else COL_BTN_WARN
            self.root.after(0, lambda: self.lbl_face_status.config(text=f"{face_score:.2f}", fg=f_col))
            
            key_score = calculate_mahalanobis_score(input_keys, target_mean, target_std)
            k_col = COL_BTN_SUCC if key_score > 0.5 else COL_BTN_WARN
            self.root.after(0, lambda: self.lbl_key_status.config(text=f"{key_score:.2f}", fg=k_col))
            
            final_score = adaptive_fusion(face_score, key_score, 1.0) 
            self.log(f" > Scores: Face={face_score:.2f}, Key={key_score:.2f}, Final={final_score:.2f}")
            
            # --- BIOMETRIC MENAGERIE LOGIC ---
            animal = "Sheep (Normal)"
            if (face_src == target) and (key_src == target):
                if final_score < 0.4: animal = "GOAT (Hard to match)"
            else:
                if final_score > 0.6: animal = "WOLF (Good impersonator)"
            
            self.log(f" > Classification: {animal}")
            # ---------------------------------

            decision = "GRANTED" if final_score >= FUSION_THRESHOLD else "DENIED"
            self.log(f" > Access {decision}")
            
            # Update Stats
            GLOBAL_STATS["sim_attempts"] += 1
            if (face_src == target) and (key_src == target):
                if final_score < FUSION_THRESHOLD: GLOBAL_STATS["last_frr"] += 1
            else:
                if final_score >= FUSION_THRESHOLD: GLOBAL_STATS["last_far"] += 1
                
        except Exception as e: self.log(f"Error: {e}")

# ==========================================
#            PART 3: ADMIN PANEL
# ==========================================
class AdminPanel:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Admin Panel")
        self.win.configure(bg=COL_BG)
        center_window(self.win, 900, 600)
        notebook = ttk.Notebook(self.win)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self.tab_users = tk.Frame(notebook, bg=COL_BG); notebook.add(self.tab_users, text="Users")
        self.tab_intruders = tk.Frame(notebook, bg=COL_BG); notebook.add(self.tab_intruders, text="Intruders")
        self.tab_eval = tk.Frame(notebook, bg=COL_BG); notebook.add(self.tab_eval, text="Evaluation (DET)")
        self.build_users_tab(); self.build_intruders_tab(); self.build_eval_tab()

    def build_users_tab(self):
        left_f = tk.Frame(self.tab_users, width=200, bg=COL_PANEL); left_f.pack(side=tk.LEFT, fill='y', padx=10, pady=10)
        right_f = tk.Frame(self.tab_users, bg=COL_BG); right_f.pack(side=tk.RIGHT, fill='both', expand=True, padx=10, pady=10)
        self.ulist = tk.Listbox(left_f, font=FONT_MAIN, bg=COL_BG, fg="black"); self.ulist.pack(fill='both', expand=True)
        self.ulist.bind('<<ListboxSelect>>', self.show_user_details)
        for user in user_mapping.keys(): self.ulist.insert(tk.END, user)
        self.u_face_lbl = tk.Label(right_f, bg="black", width=200, height=200); self.u_face_lbl.pack(pady=5)
        self.u_graph_canvas = tk.Canvas(right_f, height=150, bg=COL_BG, highlightthickness=0); self.u_graph_canvas.pack(fill='x', padx=20)

    def show_user_details(self, event):
        sel = self.ulist.curselection()
        if not sel: return
        user = self.ulist.get(sel[0])
        face_path = os.path.join(FACE_DATASET, user, "1.jpg") 
        if os.path.exists(face_path):
            try:
                img = Image.open(face_path).resize((200, 200)); ph = ImageTk.PhotoImage(img)
                self.u_face_lbl.config(image=ph, width=200, height=200); self.u_face_lbl.image = ph
            except: pass
        key_path = os.path.join(KEY_DATASET, f"{user}.csv"); profile_mean = []
        if os.path.exists(key_path):
            try:
                with open(key_path, 'r') as f:
                    rows = [list(map(float, r)) for r in csv.reader(f) if r]
                max_len = max(len(r) for r in rows); padded = [r + [0]*(max_len - len(r)) for r in rows]
                profile_mean = np.mean(padded, axis=0).tolist()
            except: pass
        self.draw_admin_graph(self.u_graph_canvas, profile_mean)

    def draw_admin_graph(self, canvas, data):
        canvas.delete("all"); w = canvas.winfo_width(); h = 150
        if not data: return
        if w < 50: w = 400 
        bar_w = w / len(data); max_val = max(data) + 0.1
        for i, val in enumerate(data):
            x = i * bar_w; bar_h = (val / max_val) * (h - 20)
            canvas.create_rectangle(x+2, h, x+bar_w-2, h-bar_h, fill=COL_ACCENT, outline="")

    def build_intruders_tab(self):
        frame = tk.Frame(self.tab_intruders, bg=COL_BG); frame.pack(fill='both', expand=True)
        self.ilist = tk.Listbox(frame, width=30, bg=COL_PANEL, fg="black"); self.ilist.pack(side=tk.LEFT, fill='y', padx=10, pady=10)
        self.ilist.bind('<<ListboxSelect>>', self.show_img)
        self.ilabel = tk.Label(frame, bg="black"); self.ilabel.pack(side=tk.RIGHT, fill='both', expand=True, padx=10, pady=10)
        if os.path.exists(INTRUDER_DIR):
            for f in os.listdir(INTRUDER_DIR): self.ilist.insert(tk.END, f)

    def show_img(self, event):
        sel = self.ilist.curselection()
        if not sel: return
        path = os.path.join(INTRUDER_DIR, self.ilist.get(sel[0]))
        try:
            img = Image.open(path).resize((400, 300)); ph = ImageTk.PhotoImage(img)
            self.ilabel.config(image=ph); self.ilabel.image = ph
        except: pass

    def build_eval_tab(self):
        monitor_frame = tk.Frame(self.tab_eval, bg=COL_BG)
        monitor_frame.pack(fill='x', padx=10, pady=10)
        self.monitor_label = tk.Label(monitor_frame, text="Initializing...", font=("Consolas", 14), bg=COL_BG, fg=COL_ACCENT)
        self.monitor_label.pack()
        
        btn_frame = tk.Frame(self.tab_eval, bg=COL_BG)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="GENERATE DET CURVE", bg=COL_BTN_SUCC, fg="white", font=("Verdana", 10), command=self.run_test).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="EXPORT REPORT", bg=COL_BTN_MAIN, fg="white", font=("Verdana", 10), command=self.export_report).pack(side=tk.LEFT, padx=10)
        
        self.det_canvas = tk.Canvas(self.tab_eval, width=600, height=350, bg="white")
        self.det_canvas.pack(pady=10)
        
        self.elog = tk.Text(self.tab_eval, height=8, bg=COL_PANEL, fg="black"); self.elog.pack(fill='x', padx=20, pady=5)

    def export_report(self):
        try:
            with open(REPORT_FILE, "w") as f:
                f.write(f"SENTINEL BIOMETRIC REPORT\n")
                f.write(f"=========================\n")
                f.write(f"Total Simulations: {len(self.gen_scores) + len(self.imp_scores)}\n")
            messagebox.showinfo("Export", f"Report saved to {REPORT_FILE}")
        except: pass

    def run_test(self):
        self.elog.delete(1.0, tk.END)
        self.elog.insert(tk.END, "Running DET Analysis (0.0 to 1.0 thresholds)... please wait...\n")
        self.win.update()
        threading.Thread(target=self._run_det_analysis, daemon=True).start()

    def _run_det_analysis(self):
        try:
            if not os.path.exists(os.path.join(MODEL_DIR, "face_model.yml")): return
            face_model = cv2.face.LBPHFaceRecognizer_create()
            face_model.read(os.path.join(MODEL_DIR, "face_model.yml"))
            
            self.gen_scores = []
            self.imp_scores = []
            users = list(user_mapping.keys())
            key_profiles, key_stds = {}, {}
            
            for u in users:
                try:
                    with open(os.path.join(KEY_DATASET, f"{u}.csv"), 'r') as f:
                        rows = [list(map(float, r)) for r in csv.reader(f) if r]
                    min_len = min(len(r) for r in rows)
                    trimmed = [r[:min_len] for r in rows]
                    key_profiles[u] = np.mean(trimmed, axis=0).tolist()
                    key_stds[u] = np.std(trimmed, axis=0).tolist()
                except: key_profiles[u] = []; key_stds[u] = []

            for target in users:
                f_path = os.path.join(FACE_DATASET, target)
                if os.path.exists(f_path):
                    imgs = [f for f in os.listdir(f_path) if f.endswith('jpg')]
                    # Use fewer images for speed
                    for img_name in imgs[:5]:
                        img = cv2.imread(os.path.join(f_path, img_name), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (200, 200))
                        label, conf = face_model.predict(img)
                        f_score = max(0, (100-conf)/100) if label == user_mapping[target] else 0.0
                        k_score = 1.0 
                        final = (f_score * DEFAULT_W_FACE) + (k_score * DEFAULT_W_KEY)
                        self.gen_scores.append(final)

                for attacker in users:
                    if target == attacker: continue
                    f_path_atk = os.path.join(FACE_DATASET, attacker)
                    if not os.path.exists(f_path_atk): continue
                    imgs = [f for f in os.listdir(f_path_atk) if f.endswith('jpg')]
                    attacker_keys = key_profiles.get(attacker, [])
                    target_profile = key_profiles.get(target, [])
                    target_std = key_stds.get(target, [])
                    for img_name in imgs[:2]:
                        img = cv2.imread(os.path.join(f_path_atk, img_name), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (200, 200))
                        label, conf = face_model.predict(img)
                        f_score = max(0, (100-conf)/100) if label == user_mapping[target] else 0.0
                        k_score = calculate_mahalanobis_score(attacker_keys, target_profile, target_std)
                        final = (f_score * DEFAULT_W_FACE) + (k_score * DEFAULT_W_KEY)
                        self.imp_scores.append(final)

            thresholds = np.arange(0.0, 1.05, 0.05)
            det_points = []
            for t in thresholds:
                far = (sum(1 for s in self.imp_scores if s >= t) / len(self.imp_scores)) * 100 if self.imp_scores else 0
                frr = (sum(1 for s in self.gen_scores if s < t) / len(self.gen_scores)) * 100 if self.gen_scores else 0
                det_points.append((far, frr))
                
            def draw_det():
                # --- FIX: Check if widget exists ---
                if not self.det_canvas.winfo_exists(): return
                self.det_canvas.delete("all")
                w, h = 600, 350
                self.det_canvas.create_line(50, h-50, w-50, h-50, width=2)
                self.det_canvas.create_line(50, h-50, 50, 50, width=2)
                self.det_canvas.create_text(w-50, h-30, text="FAR (%)")
                self.det_canvas.create_text(30, 50, text="FRR (%)")
                prev_x, prev_y = None, None
                for far, frr in det_points:
                    x = 50 + (far * (w-100)/100)
                    y = (h-50) - (frr * (h-100)/100)
                    self.det_canvas.create_oval(x-3, y-3, x+3, y+3, fill="red")
                    if prev_x is not None:
                        self.det_canvas.create_line(prev_x, prev_y, x, y, fill="blue", width=2)
                    prev_x, prev_y = x, y
                if self.elog.winfo_exists():
                    self.elog.insert(tk.END, f"Graph Generated. Points: {len(det_points)}\n")
            
            self.win.after(0, draw_det)
            
        except Exception as e:
            print(f"Eval Error: {e}")

# ==========================================
#            PART 4: LAUNCHER
# ==========================================
class Launcher:
    def __init__(self, root):
        self.root = root
        self.root.title("SENTINEL LAUNCHER")
        center_window(self.root, 500, 550)
        self.root.configure(bg=COL_BG)
        
        tk.Label(self.root, text="SENTINEL MASTER SUITE", font=FONT_HEADER, bg=COL_BG, fg=COL_ACCENT).pack(pady=30)
        
        # Updated Names
        tk.Label(self.root, text=STUDENT_NAMES, font=("Verdana", 10), bg=COL_BG, fg="#888", wraplength=400).pack(pady=5)
        
        # --- CURVED BUTTONS ---
        RoundedButton(self.root, width=300, height=60, corner_radius=30, padding=10, color=COL_BTN_SUCC, text="🚀 LIVE SYSTEM", command=self.launch_live).pack(pady=20)
        RoundedButton(self.root, width=300, height=60, corner_radius=30, padding=10, color=COL_ACCENT, text="🎓 SIMULATION", command=self.launch_sim).pack(pady=20)
        RoundedButton(self.root, width=150, height=50, corner_radius=25, padding=10, color=COL_BTN_WARN, text="EXIT", command=self.root.destroy).pack(pady=30)

    def launch_live(self):
        for w in self.root.winfo_children(): w.destroy()
        LiveSystem(self.root)

    def launch_sim(self):
        for w in self.root.winfo_children(): w.destroy()
        SimSystem(self.root)

if __name__ == "__main__":
    root = tk.Tk()
    app = Launcher(root)
    root.mainloop()
