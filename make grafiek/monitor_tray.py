import multiprocessing
import time
import pull
import convert
import sys
import os
import json
import threading
from datetime import datetime
from pathlib import Path
import ctypes

def already_running():
    mutex = ctypes.windll.kernel32.CreateMutexW(
        None,
        False,
        "Global\\ScriptMonitorSingleton"
    )
    return ctypes.windll.kernel32.GetLastError() == 183



# extern
import pystray
from PIL import Image, ImageDraw

# Windows helpers
try:
    import win32gui
    import win32con
except Exception:
    win32gui = None
    win32con = None

# --- paden / bestanden ---
SCRIPT_DIR = Path(__file__).parent
PULL_SCRIPT = SCRIPT_DIR / "pull.py"
GOED_SCRIPT = SCRIPT_DIR / "convert.py"

SUCCESS_LOG = SCRIPT_DIR / "success_log.txt"
ERROR_LOG = SCRIPT_DIR / "error_log.txt"
CONTROL_FILE = SCRIPT_DIR / "tray_control.json"
LAST_OUTPUTS = SCRIPT_DIR / "last_outputs.json"

# --- globale status ---
error_count = 0
last_status = "Idle"
last_success_time = None
stop_flag = False
refresh_event = threading.Event()

# --- logging ---
def now_str():
    return datetime.now().isoformat(sep=" ", timespec="seconds")

def log_success(msg):
    with open(SUCCESS_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{now_str()}] {msg}\n")

def log_error(msg):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{now_str()}] {msg}\n\n")

# --- control file ---
def read_control():
    if CONTROL_FILE.exists():
        try:
            return json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log_error("Fout bij lezen control file: " + str(e))
    return {}

def write_control(data):
    try:
        CONTROL_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        log_error("Fout bij schrijven control file: " + str(e))

def toggle_auto_open(icon, item):
    cfg = read_control()
    cfg["auto_open"] = not bool(cfg.get("auto_open", False))
    write_control(cfg)

# --- tray icon ---
def create_icon(color):
    img = Image.new("RGB", (64, 64), color=color)
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 56, 56), fill=(255, 255, 255))
    return img

ICON_GREEN = create_icon((0, 200, 0))
ICON_RED = create_icon((200, 0, 0))

# --- subprocess ---
def run_pull():
    global error_count
    try:
        pull.run_once()
        log_success("pull.py OK")
        return True
    except Exception as e:
        error_count += 1
        log_error(f"pull.py fout:\n{e}")
        return False


def run_convert():
    global error_count
    try:
        convert.main()
        log_success("convert.py OK")
        return True
    except Exception as e:
        error_count += 1
        log_error(f"convert.py fout:\n{e}")
        return False



# --- bestand openen ---
def open_html(path_str):
    path = Path(path_str)
    if not path.exists():
        log_error("HTML bestaat niet: " + path_str)
        return
    try:
        os.startfile(str(path))  # opent in standaard browser
    except Exception as e:
        log_error(f"Kon HTML niet openen {path.name}: {e}")

# --- auto-open ---
def open_last_outputs_if_allowed():
    cfg = read_control()
    if not cfg.get("auto_open", False):
        return
    if not LAST_OUTPUTS.exists():
        return

    try:
        data = json.loads(LAST_OUTPUTS.read_text(encoding="utf-8"))
        for fp in data.get("html", []):
            open_html(fp)
    except Exception as e:
        log_error("Fout bij openen HTML outputs: " + str(e))

# --- menu teksten ---
def status_text(item):
    return f"Status: {last_status}"

def errors_text(item):
    return f"Errors: {error_count}"

def minutes_text(item):
    if last_success_time is None:
        return "Minuten sinds laatste update: N/A"
    return f"Minuten sinds laatste update: {int((time.time() - last_success_time) / 60)}"

def auto_open_text(item):
    return "Auto open: ON" if read_control().get("auto_open", False) else "Auto open: OFF"

# --- menu acties ---
def open_last_html(icon, item):
    if not LAST_OUTPUTS.exists():
        log_error("Geen last_outputs.json gevonden.")
        return
    try:
        data = json.loads(LAST_OUTPUTS.read_text(encoding="utf-8"))
        for fp in data.get("html", []):
            open_html(fp)
    except Exception as e:
        log_error("Fout bij openen HTML: " + str(e))

def open_success_log(icon, item):
    os.startfile(str(SUCCESS_LOG))

def open_error_log(icon, item):
    os.startfile(str(ERROR_LOG))

def refresh_now(icon, item):
    refresh_event.set()

# --- hoofdloop ---
def loop_scripts(icon):
    global last_status, last_success_time

    while not stop_flag:
        last_status = "Running pull.py"
        ok1 = run_pull()

        ok2 = False
        if ok1:
            last_status = "Running convert.py"
            ok2 = run_convert()

        if ok1 and ok2:
            last_status = "Last run OK - waiting 1 hour"
            last_success_time = time.time()
            icon.icon = ICON_GREEN
            open_last_outputs_if_allowed()

            refresh_event.clear()
            refresh_event.wait(timeout=3600)
        else:
            last_status = "Error - retrying"
            icon.icon = ICON_RED
            time.sleep(3)

# --- tray setup ---
def setup_tray():
    icon = pystray.Icon("ScriptMonitor", ICON_GREEN, "Script Monitor")

    def on_quit(icon, item):
        global stop_flag
        stop_flag = True
        refresh_event.set()
        icon.stop()

    icon.menu = pystray.Menu(
        pystray.MenuItem(status_text, None, enabled=False),
        pystray.MenuItem(errors_text, None, enabled=False),
        pystray.MenuItem(minutes_text, None, enabled=False),
        pystray.MenuItem("Refresh Now", refresh_now),
        pystray.MenuItem(auto_open_text, toggle_auto_open),
        pystray.MenuItem("Open last HTML", open_last_html),
        pystray.MenuItem("Open success log", open_success_log),
        pystray.MenuItem("Open error log", open_error_log),
        pystray.MenuItem("Quit", on_quit)
    )

    threading.Thread(target=loop_scripts, args=(icon,), daemon=True).start()
    icon.run()




if __name__ == "__main__":
    multiprocessing.freeze_support()

    if already_running():
        sys.exit(0)   # HARD STOP → geen tray, geen threads, niks

    if not CONTROL_FILE.exists():
        write_control({"auto_open": False})

    setup_tray()



