# smartcontrol_ai.py  — Deel 1/4
# ----------------------------
# Overzicht:
# - Imports en veilige import-check
# - Globale SETTINGS / load/save
# - Hulpfuncties
# - Modbus IO wrapper (read/write helpers)
#
# LET OP: dit is deel 1 van 4. Plak elk deel in dezelfde bestandsvolgorde.

import os
import sys
import io
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog, messagebox
from datetime import datetime, time as dtime, date
import csv

# -------------------- VEILIGE IMPORTS CHECK --------------------
# we detecteren ontbrekende packages zodat de gebruiker waarschuwingen krijgt
def _safe_imports():
    missing = []
    try:
        import pandas as _
    except Exception:
        missing.append("pandas")
    try:
        import numpy as _
    except Exception:
        missing.append("numpy")
    try:
        from pymodbus.client.tcp import ModbusTcpClient as _
    except Exception:
        missing.append("pymodbus")
    try:
        import requests as _
    except Exception:
        missing.append("requests")
    return missing

MISSING = _safe_imports()

# importeren (kan excepties werpen als modules ontbreken; main toont waarschuwing)
import pandas as pd  # type: ignore
import numpy as np   # type: ignore
from pymodbus.client.tcp import ModbusTcpClient  # type: ignore
import requests  # type: ignore

# -------------------- PADEN & DEFAULT SETTINGS --------------------
# alle instelbare waarden zitten in SETTINGS; later worden ze in variabelen geladen
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(sys.argv[0]), "data")

def ensure_dir(path: str):
    """Maak map aan als die nog niet bestaat; negeer fouten."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

# conversie-helpers
def _to_bool(x, default=False):
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    s = str(x).strip().lower()
    if s in ("1","true","yes","y","on"): return True
    if s in ("0","false","no","n","off"): return False
    return default

def _to_tuple_hours(x, default=(0,7)):
    try:
        s = str(x).strip()
        if "-" in s:
            a,b = s.split("-",1)
            return (int(a), int(b))
        if "," in s:
            a,b = s.split(",",1)
            return (int(a), int(b))
    except Exception:
        pass
    return default

# DEFAULT SETTINGS (hier kun je adressen en parameters wijzigen)
SETTINGS = {
    # communicatie
    "PLC_IP": "192.168.1.181",
    "MODBUS_PORT": 503,
    "INTERVAL_SEC": 5,

    # regelspecificaties
    "AGRESSIVITEIT": 1.0,
    "AUTOMATISCH_SCHAKELEN": True,
    "KOEL_ONDERSHOOT": 5.0,
    "WARM_OVERSHOOT": 3.0,
    "EXTRA_LUCHT_DUUR_SEC": 60,
    "TRAIN_SETPOINT": 20.0,

    # logging/debug
    "INFO": True,
    "DEBUG": False,

    # nachtkoeling/weervoorspelling
    "WEER_NACHT_KOELING": True,
    "HITTEGRENS": 27.0,
    "NACHTUREN": "0-7",
    "WEER_COORD_LAT": 51.6732,
    "WEER_COORD_LON": 5.6268,
    "WEERVOORSPELLING_CHECK_INTERVAL": 100,
    "VOORUITKIJK_DAGEN": 10,

    # paden
    "DATA_DIR": DEFAULT_DATA_DIR,
    "EXCEL_PAD": os.path.join(DEFAULT_DATA_DIR, "Temperature_2025.xlsx"),
    "MODEL_BESTAND": os.path.join(DEFAULT_DATA_DIR, "smartcontrol_model.json"),
    "RUNTIME_LOG": os.path.join(DEFAULT_DATA_DIR, "smartcontrol_runtime.csv"),
    "TRAINING_DATA_FILE": os.path.join(DEFAULT_DATA_DIR, "training_data_used_for_model.csv"),
    "SETTINGS_JSON": os.path.join(DEFAULT_DATA_DIR, "settings.json"),
    "LOG_FILE": os.path.join(DEFAULT_DATA_DIR, "smartcontrol_log.csv"),

    # Modbus adressering (standaardadressen; pas aan indien nodig)
    "ADDR_instuur_lucht": 26,
    "ADDR_recycle_lucht": 11,
    "ADDR_buitenlucht": 31,
    "ADDR_fitness_temp": 6,
    "ADDR_setpoint": 2,
    "ADDR_output_klep": 0,
    "ADDR_hand_of_auto": 8273,
    "ADDR_koel_of_warm": 8256,
    "ADDR_extra_lucht": 8279,
    # --- NIEUW: CO2 sensor Modbus-register (raw registernummer) ---
    "ADDR_co2_sensor": 533,

    # autostart + NN
    "AUTO_START_DELAY_SEC": 10,
    "USE_NN": True,
    "INSTUUR_AFTER_EXTRA": 8279
}

def load_settings():
    """Laad settings.json indien aanwezig en zet paden absolutebestand. Default-waarden blijven behouden."""
    ensure_dir(SETTINGS["DATA_DIR"])
    settings_path = SETTINGS["SETTINGS_JSON"]
    if not os.path.isabs(settings_path):
        settings_path = os.path.join(SETTINGS["DATA_DIR"], os.path.basename(settings_path))

    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                user = json.load(f)
            for k,v in user.items():
                SETTINGS[k] = v
        except Exception:
            pass
    else:
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(SETTINGS, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # normaliseer enkele paden naar absolute paden in de DATA_DIR
    dd = SETTINGS["DATA_DIR"]
    SETTINGS["EXCEL_PAD"] = SETTINGS["EXCEL_PAD"] if os.path.isabs(SETTINGS["EXCEL_PAD"]) else os.path.join(dd, SETTINGS["EXCEL_PAD"])
    SETTINGS["MODEL_BESTAND"] = SETTINGS["MODEL_BESTAND"] if os.path.isabs(SETTINGS["MODEL_BESTAND"]) else os.path.join(dd, SETTINGS["MODEL_BESTAND"])
    SETTINGS["RUNTIME_LOG"] = SETTINGS["RUNTIME_LOG"] if os.path.isabs(SETTINGS["RUNTIME_LOG"]) else os.path.join(dd, SETTINGS["RUNTIME_LOG"])
    SETTINGS["TRAINING_DATA_FILE"] = SETTINGS["TRAINING_DATA_FILE"] if os.path.isabs(SETTINGS["TRAINING_DATA_FILE"]) else os.path.join(dd, SETTINGS["TRAINING_DATA_FILE"])
    SETTINGS["SETTINGS_JSON"] = settings_path
    SETTINGS["LOG_FILE"] = SETTINGS["LOG_FILE"] if os.path.isabs(SETTINGS["LOG_FILE"]) else os.path.join(dd, os.path.basename(SETTINGS["LOG_FILE"]))
    ensure_dir(SETTINGS["DATA_DIR"])

def save_settings():
    """Sla huidige SETTINGS op naar settings.json."""
    ensure_dir(SETTINGS["DATA_DIR"])
    path = SETTINGS["SETTINGS_JSON"]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SETTINGS, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

# laad settings direct bij import
load_settings()

# -------------------- GLOBALS (gemakkelijk te gebruiken variabelen) --------------------
PLC_IP = SETTINGS["PLC_IP"]
MODBUS_PORT = int(SETTINGS["MODBUS_PORT"])
INTERVAL_SEC = int(SETTINGS["INTERVAL_SEC"])
AGRESSIVITEIT = float(SETTINGS["AGRESSIVITEIT"])
AUTOMATISCH_SCHAKELEN = _to_bool(SETTINGS["AUTOMATISCH_SCHAKELEN"], True)
KOEL_ONDERSHOOT = float(SETTINGS["KOEL_ONDERSHOOT"])
WARM_OVERSHOOT = float(SETTINGS["WARM_OVERSHOOT"])
EXTRA_LUCHT_DUUR_SEC = int(SETTINGS["EXTRA_LUCHT_DUUR_SEC"])
TRAIN_SETPOINT = float(SETTINGS["TRAIN_SETPOINT"])
INFO = _to_bool(SETTINGS["INFO"], True)
DEBUG = _to_bool(SETTINGS["DEBUG"], False)
WEER_NACHT_KOELING = _to_bool(SETTINGS["WEER_NACHT_KOELING"], True)
HITTEGRENS = float(SETTINGS["HITTEGRENS"])
NACHTUREN = _to_tuple_hours(SETTINGS["NACHTUREN"], (0,7))
WEER_COORDINATEN = {"lat": float(SETTINGS["WEER_COORD_LAT"]), "lon": float(SETTINGS["WEER_COORD_LON"])}
WEERVOORSPELLING_CHECK_INTERVAL = int(SETTINGS["WEERVOORSPELLING_CHECK_INTERVAL"])
VOORUITKIJK_DAGEN = int(SETTINGS["VOORUITKIJK_DAGEN"])
DATA_DIR = SETTINGS["DATA_DIR"]
EXCEL_PAD = SETTINGS["EXCEL_PAD"]
MODEL_BESTAND = SETTINGS["MODEL_BESTAND"]
RUNTIME_LOG = SETTINGS["RUNTIME_LOG"]
TRAINING_DATA_FILE = SETTINGS["TRAINING_DATA_FILE"]
SETTINGS_JSON = SETTINGS["SETTINGS_JSON"]
USE_NN = _to_bool(SETTINGS.get("USE_NN", True), True)
INSTUUR_AFTER_EXTRA = int(SETTINGS.get("INSTUUR_AFTER_EXTRA", 8278))
LOG_FILE = SETTINGS.get("LOG_FILE", os.path.join(DATA_DIR, "smartcontrol_log.csv"))

# ADDR mapping: leesbare keys naar raw register-adressen
ADDR = {
    "instuur_lucht": SETTINGS["ADDR_instuur_lucht"],
    "recycle_lucht": int(SETTINGS["ADDR_recycle_lucht"]),
    "buitenlucht": int(SETTINGS["ADDR_buitenlucht"]),
    "fitness_temp": int(SETTINGS["ADDR_fitness_temp"]),
    "setpoint": int(SETTINGS["ADDR_setpoint"]),
    "output_klep": int(SETTINGS["ADDR_output_klep"]),
    "hand_of_auto": int(SETTINGS["ADDR_hand_of_auto"]),
    "koel_of_warm": int(SETTINGS["ADDR_koel_of_warm"]),
    "extra_lucht": int(SETTINGS["ADDR_extra_lucht"]),
    # CO2 sensor adres in ADDR mapping
    "co2_sensor": int(SETTINGS["ADDR_co2_sensor"]),
}

AUTO_START_DELAY_SEC = int(SETTINGS["AUTO_START_DELAY_SEC"])
ensure_dir(DATA_DIR)

# -------------------- LOGGING --------------------
LOG_LOCK = threading.Lock()
def write_log(level: str, message: str):
    """
    Schrijf een CSV-logregel: timestamp, level, message.
    Splits multi-line messages in aparte lijnen.
    """
    try:
        ensure_dir(os.path.dirname(LOG_FILE))
        timestamp = datetime.now().isoformat()
        lines = str(message).splitlines()
        with LOG_LOCK:
            file_exists = os.path.exists(LOG_FILE)
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                if not file_exists:
                    writer.writerow(["timestamp", "level", "message"])
                for ln in lines:
                    writer.writerow([timestamp, level.upper(), ln])
    except Exception:
        try:
            print(f"[LOG FAIL] {level}: {message}")
        except:
            pass

# -------------------- HULP (nacht/datum/addr helper) --------------------
def is_nacht():
    """Controleer of het huidige tijdstip binnen NACHTUREN valt."""
    nu = datetime.now().time()
    start, eind = NACHTUREN
    return dtime(start) <= nu <= dtime(eind)

def _get_addr(key):
    """Veilige lookup van ADDR dictionary; retourneert None bij '#...' of invalid."""
    v = ADDR.get(key, None)
    if isinstance(v, str) and v.startswith("#"):
        return None
    try:
        return int(v)
    except Exception:
        return None

# -------------------- MODBUS IO WRAPPER --------------------
class IO:
    """
    Eenvoudige wrapper om Modbus TCP-commando's overzichtelijker te gebruiken.
    Bevat helpers voor: read_coil, read_di (discrete input), read_word (met scaling),
    read_word_raw (hele register), write_klep (register met 0-100), write_coil, write_register.
    """
    def __init__(self, ip, port):
        self.client = None
        self.connected = False
        try:
            self.client = ModbusTcpClient(ip, port=port)
            self.connected = bool(self.client.connect())
        except Exception:
            self.client = None
            self.connected = False

    def close(self):
        if self.client:
            try: self.client.close()
            except: pass

    def read_coil(self, addr, default=0):
        """Lees een coil (writeable coil) als 0/1."""
        if addr is None or not self.client: return default
        try:
            r = self.client.read_coils(address=addr, count=1)
            return int(r.bits[0]) if (r and not r.isError()) else default
        except: return default

    def read_di(self, addr, default=0):
        """Lees een discrete input (read-only) als 0/1."""
        if addr is None or not self.client: return default
        try:
            r = self.client.read_discrete_inputs(address=addr, count=1)
            return int(r.bits[0]) if (r and not r.isError()) else default
        except: return default

    def read_word(self, addr, scale=0.1, default=0.0):
        """
        Lees een holding register en schaal de waarde (bijv scale=0.1 voor 1 decimaal).
        Retourneert float.
        """
        if addr is None or not self.client:
            return default
        try:
            r = self.client.read_holding_registers(address=addr, count=1)
            if r and not r.isError():
                raw = r.registers[0]
                if raw > 32767:
                    raw = raw - 65536
                return raw * scale
            return default
        except:
            return default

    def read_word_raw(self, addr, default=0):
        """Lees raw registerwaarde (hele integer), handig voor CO₂-ppm of setpoint raw."""
        if addr is None or not self.client: return default
        try:
            r = self.client.read_holding_registers(address=addr, count=1)
            if r and not r.isError():
                return int(r.registers[0])
            return default
        except: return default

    def write_klep(self, addr, waarde):
        """
        Schrijf kleppositie (0-100) naar gegeven register.
        Als EXTRA_OVERRIDE actief is, negeren we writes < 100 om extra-lucht prioriteit te respecteren.
        """
        global EXTRA_OVERRIDE
        if addr is None:
            if DEBUG:
                write_log("DEBUG", "[IO] write_klep: geen adres (None)")
            return False
        try:
            waarde = int(max(0, min(100, int(round(waarde)))))
        except Exception:
            waarde = 0
        if EXTRA_OVERRIDE and waarde != 100:
            if DEBUG:
                write_log("DEBUG", f"[IO] Override actief — negeer write_klep({waarde})")
            return True
        if not self.client:
            if DEBUG:
                write_log("DEBUG", "[IO] Geen Modbus client beschikbaar bij write_klep")
            return False
        try:
            rr = self.client.write_register(address=addr, value=waarde)
            ok = (rr is not None and not getattr(rr, "isError", lambda: False)())
            if DEBUG:
                write_log("DEBUG", f"[IO] write_klep({addr} <- {waarde}) => {'OK' if ok else 'MISLUKT'}")
            return ok
        except Exception as e:
            if DEBUG:
                write_log("ERROR", f"[IO] Exception bij write_klep: {e}")
            return False

    def write_coil(self, addr, value=True):
        """Schrijf coil (True/False)."""
        if addr is None or not self.client:
            if DEBUG:
                write_log("DEBUG", "[IO] write_coil: geen adres of client")
            return False
        try:
            rr = self.client.write_coil(address=addr, value=bool(value))
            ok = (rr is not None and not getattr(rr, "isError", lambda: False)())
            if DEBUG:
                write_log("DEBUG", f"[IO] write_coil({addr} <- {value}) => {'OK' if ok else 'MISLUKT'}")
            return ok
        except Exception as e:
            if DEBUG:
                write_log("ERROR", f"[IO] Exception bij write_coil: {e}")
            return False

    def write_register(self, addr, value):
        """Schrijf raw register value (hele getal)."""
        if addr is None or not self.client:
            if DEBUG:
                write_log("DEBUG", "[IO] write_register: geen client of adres")
            return False
        try:
            rr = self.client.write_register(address=addr, value=int(value))
            ok = (rr is not None and not getattr(rr, "isError", lambda: False)())
            if DEBUG:
                write_log("DEBUG", f"[IO] write_register({addr} <- {value}) => {'OK' if ok else 'MISLUKT'}")
            return ok
        except Exception as e:
            if DEBUG:
                write_log("ERROR", f"[IO] Exception bij write_register: {e}")
            return False
# smartcontrol_ai.py  — Deel 2/4
# ----------------------------
# Overzicht:
# - Zelflerend model (eenvoudig NN of fallback linear)
# - Functies om trainingsdata te laden en model op te slaan
# - Analytische hulpmethoden voor mengtemperatuur en clamp-logica

# -------------------- LERENDE REGELAAR --------------------
class SelfLearningController:
    """
    Eenvoudige implementatie van een 'klein' neuronetwork:
    - Laadt/slaat model als JSON
    - Voorspelt klep percentage gegeven features
    - Kan trainen op Excel + runtime logs
    """
    def __init__(self, model_path=MODEL_BESTAND):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.last_fit = None
        # verwachte input features (in deze volgorde bij predict)
        self.input_features = ["buiten", "recycle", "fitness", "setpoint", "warm_flag"]
        self._load_model()

    def _load_model(self):
        """Laad model uit JSON als aanwezig, anders geen model."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if obj.get("type") == "nn":
                    self.model = {
                        "W1": np.array(obj["W1"], dtype=float),
                        "b1": np.array(obj["b1"], dtype=float),
                        "W2": np.array(obj["W2"], dtype=float),
                        "b2": float(obj["b2"])
                    }
                    self.scaler = {
                        "mean": np.array(obj.get("scaler_mean", []), dtype=float) if obj.get("scaler_mean") else None,
                        "std": np.array(obj.get("scaler_std", []), dtype=float) if obj.get("scaler_std") else None
                    }
                    self.last_fit = obj.get("last_fit")
                elif obj.get("type") == "linear":
                    self.model = {"b": np.array(obj.get("b"), dtype=float)}
                    self.scaler = None
                    self.last_fit = obj.get("last_fit")
                else:
                    self.model = None
        except Exception:
            self.model = None

    def _save_model(self):
        """Sla intern model op als JSON zodat training persistent is."""
        ensure_dir(os.path.dirname(self.model_path))
        try:
            if self.model and "W1" in self.model:
                out = {
                    "type": "nn",
                    "W1": self.model["W1"].tolist(),
                    "b1": self.model["b1"].tolist(),
                    "W2": self.model["W2"].tolist(),
                    "b2": float(self.model["b2"]),
                    "scaler_mean": (self.scaler["mean"].tolist() if self.scaler and self.scaler.get("mean") is not None else None),
                    "scaler_std": (self.scaler["std"].tolist() if self.scaler and self.scaler.get("std") is not None else None),
                    "last_fit": datetime.now().isoformat()
                }
            elif self.model and "b" in self.model:
                out = {"type": "linear", "b": self.model["b"].tolist(), "last_fit": datetime.now().isoformat()}
            else:
                out = {"type": "none", "last_fit": datetime.now().isoformat()}
            with open(self.model_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            self.last_fit = out.get("last_fit")
        except Exception:
            pass

    def _feature_row(self, buiten, recycle, fitness, setpoint, warm_flag=0):
        """Bouw feature vector (numpy array) voor predict/fitting."""
        return np.array([buiten, recycle, fitness, setpoint, 1 if warm_flag else 0], dtype=float)

    def predict(self, buiten, recycle, fitness, setpoint, fallback_klep, warm_flag=0):
        """
        Voorspel klep%:
        - Als model NN aanwezig: gebruik NN (+scaler indien aanwezig)
        - Als linear aanwezig: gebruik linear
        - Anders: fallback_klep
        """
        if self.model is None:
            return fallback_klep
        if "W1" in self.model:
            # NN predict
            x = self._feature_row(buiten, recycle, fitness, setpoint, warm_flag).astype(float)
            if self.scaler and self.scaler.get("mean") is not None:
                mean = self.scaler["mean"]
                std = self.scaler["std"]
                std_adj = np.where(std == 0, 1.0, std)
                x = (x - mean) / std_adj
            W1 = self.model["W1"]; b1 = self.model["b1"]; W2 = self.model["W2"]; b2 = self.model["b2"]
            z1 = x.dot(W1.T) + b1
            a1 = np.tanh(z1)
            y = float(a1.dot(W2.T) + b2)
            return max(0, min(100, y))
        elif "b" in self.model:
            # oude linear vorm (compat)
            x = np.concatenate(([1.0], self._feature_row(buiten, recycle, fitness, setpoint, warm_flag)[:4]))
            try:
                return max(0, min(100, float(np.dot(self.model["b"], x))))
            except Exception:
                return fallback_klep
        else:
            return fallback_klep

    def fit_from_dataframe(self, df, train_setpoint: float):
        """
        Train NN of fallback linear op samengevoegde dataframe.
        Verwacht kolommen voor buiten, recycle, fitness, klep (target).
        """
        if df is None or len(df) == 0 or np is None:
            if DEBUG: write_log("DEBUG", "[LEARN] Geen dataframe beschikbaar om te trainen.")
            return False
        cols = {c.lower(): c for c in df.columns}
        def pick_contains(fragment):
            for k, v in cols.items():
                if fragment in k:
                    return v
            return None
        col_buiten  = pick_contains("buiten temperatuur") or pick_contains("buiten")
        col_recycle = pick_contains("recycle temperatuur") or pick_contains("recycle")
        col_fit     = pick_contains("temperatuur fitness") or pick_contains("fitness")
        col_klep    = pick_contains("buitenlucht klep") or pick_contains("klep") or pick_contains("buitenlucht")
        needed = [col_buiten, col_recycle, col_fit, col_klep]
        if any(c is None for c in needed):
            if DEBUG: write_log("DEBUG", f"[LEARN] Vereiste kolommen niet gevonden in data voor trainen: {needed}")
            return False
        sub = df[[col_buiten, col_recycle, col_fit, col_klep]].dropna()
        if len(sub) < 20:
            if DEBUG: write_log("DEBUG", f"[LEARN] Te weinig rijen voor trainen (gebruikt {len(sub)}; >=20 vereist).")
            return False
        const_sp = float(train_setpoint)
        # X = buiten, recycle, fitness, setpoint, warm_flag (warm_flag onbekend in historische data -> 0)
        X_raw = np.column_stack([
            sub[col_buiten].astype(float).values,
            sub[col_recycle].astype(float).values,
            sub[col_fit].astype(float).values,
            np.full(len(sub), const_sp, dtype=float),
            np.zeros(len(sub), dtype=float)  # warm_flag placeholder
        ])
        y = sub[col_klep].astype(float).values

        # standaardiseer features
        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0)
        std_adj = np.where(std == 0, 1.0, std)
        X = (X_raw - mean) / std_adj

        # NN: 1 verborgen laag
        n_in = X.shape[1]
        hidden = max(6, min(32, int(n_in*4)))
        # initialisatie
        rng = np.random.RandomState(42)
        W1 = rng.normal(0, 0.1, size=(hidden, n_in))
        b1 = np.zeros(hidden, dtype=float)
        W2 = rng.normal(0, 0.1, size=(1, hidden))
        b2 = 0.0

        epochs = 300
        lr = 0.01
        n = len(X)
        try:
            for epoch in range(epochs):
                Z1 = X.dot(W1.T) + b1
                A1 = np.tanh(Z1)
                Ypred = A1.dot(W2.T).reshape(-1) + b2
                error = Ypred - y
                loss = (error**2).mean()

                # backprop
                dY = (2.0/n) * error
                dW2 = (dY.reshape(1, -1).dot(A1)).reshape(W2.shape)
                db2 = dY.sum()
                dA1 = np.outer(dY, W2.reshape(-1))
                dZ1 = dA1 * (1.0 - np.tanh(Z1)**2)
                dW1 = dZ1.T.dot(X)
                db1 = dZ1.sum(axis=0)

                W1 -= lr * dW1
                b1 -= lr * db1
                W2 -= lr * dW2
                b2 -= lr * db2

                if epoch % 50 == 0 and DEBUG:
                    write_log("DEBUG", f"[LEARN] Epoch {epoch}/{epochs}, loss={loss:.4f}")
                if loss < 0.1:
                    if DEBUG:
                        write_log("DEBUG", f"[LEARN] Vroegtijdig gestopt op epoch {epoch} met loss {loss:.4f}")
                    break

            # sla model en scaler op
            self.model = {"W1": W1, "b1": b1, "W2": W2.flatten(), "b2": float(b2)}
            self.scaler = {"mean": mean, "std": std_adj}
            self._save_model()
            if DEBUG:
                write_log("INFO", "[LEARN] NN training voltooid. Model opgeslagen.")
            return True
        except Exception as e:
            if DEBUG:
                write_log("ERROR", f"[LEARN] NN training faalde: {e}")
            # fallback linear least squares
            try:
                X_lin = np.column_stack([np.ones(len(sub)), X_raw[:,0], X_raw[:,1], X_raw[:,2], X_raw[:,3]])
                beta, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
                self.model = {"b": beta}
                self.scaler = None
                self._save_model()
                if DEBUG:
                    write_log("INFO", "[LEARN] Fallback lineaire fit voltooid.")
                return True
            except Exception as e2:
                if DEBUG:
                    write_log("ERROR", f"[LEARN] Fallback lineaire fit faalde: {e2}")
                return False

    def fit_from_excel_and_runtime(self, excel_path=EXCEL_PAD, runtime_csv=RUNTIME_LOG, train_setpoint: float = TRAIN_SETPOINT):
        """
        Probeer trainingdata te laden vanuit Excel (meerdere sheets) en runtime CSV en concateneer.
        Schrijf gecombineerd trainingsbestand weg ter inspectie.
        """
        frames = []
        try:
            if excel_path and os.path.exists(excel_path) and pd is not None:
                all_sheets = pd.read_excel(excel_path, sheet_name=None)
                if isinstance(all_sheets, dict):
                    for _, sdf in all_sheets.items():
                        if sdf is not None and len(sdf) > 0:
                            frames.append(sdf)
                elif all_sheets is not None and len(all_sheets) > 0:
                    frames.append(all_sheets)
        except Exception:
            try:
                xdf = pd.read_excel(excel_path)
                if xdf is not None and len(xdf) > 0:
                    frames.append(xdf)
            except Exception:
                pass
        try:
            if runtime_csv and os.path.exists(runtime_csv):
                cdf = pd.read_csv(runtime_csv)
                if cdf is not None and len(cdf) > 0:
                    frames.append(cdf)
        except Exception:
            pass

        ensure_dir(os.path.dirname(TRAINING_DATA_FILE))
        if frames:
            try:
                df = pd.concat(frames, ignore_index=True, sort=False)
                df.to_csv(TRAINING_DATA_FILE, index=False)
            except Exception:
                try:
                    with open(TRAINING_DATA_FILE, "w", encoding="utf-8") as f:
                        f.write("note\nkon_df_concat_niet_maken\n")
                except Exception:
                    pass
        else:
            try:
                with open(TRAINING_DATA_FILE, "w", encoding="utf-8") as f:
                    f.write("timestamp,buiten,recycle,fitness,setpoint,klep,koelstand\n")
            except Exception:
                pass
            return False

        return self.fit_from_dataframe(df, train_setpoint=train_setpoint)

    def last_fit_date(self):
        """Retourneer datum (YYYY-MM-DD) waarop model voor het laatst is gefit."""
        try:
            if not self.last_fit:
                return None
            dt = datetime.fromisoformat(self.last_fit)
            return dt.date()
        except Exception:
            return None

# -------------------- ANALYTISCHE / HULP FUNCTIES --------------------
def bereken_mengtemp(buiten, recycle, klep_pct):
    """Bereken mengtemperatuur op basis van buiten en recycle en klep%."""
    f = max(0.0, min(1.0, klep_pct / 100.0))
    return buiten * f + recycle * (1 - f)

def analytische_klep(buiten, recycle, target):
    """
    Los analytisch op welke klep% nodig is om target mengtemp te bereiken.
    Als buiten==recycle: return 0 om divide-by-zero te vermijden.
    """
    denom = (buiten - recycle)
    if abs(denom) < 1e-6:
        return 0.0
    f = (target - recycle) / denom
    return max(0.0, min(100.0, f * 100.0))

def kies_target(fitness, setpoint, koelstand):
    """
    Bepaal target mengtemp afhankelijk van verwarmen/koelen.
    - Koelen (koelstand==0): target iets onder fitness m.b.t. setpoint
    - Verwarmen: iets boven
    """
    if koelstand == 0:  # koelen
        return min(setpoint, fitness - 0.1)
    else:               # verwarmen
        return max(setpoint, fitness + 0.1)

def clamp_band(klep, buiten, recycle, fitness, setpoint, koelstand):
    """
    Pas bandgrenzen toe (overshoot/undershoot) om te voorkomen dat mengtemp buiten veilige band valt.
    Als buiten <-> recycle combinatie problematisch is, pas analytische aanpassing toe.
    """
    if koelstand == 0:
        min_temp = setpoint - KOEL_ONDERSHOOT
        mt = bereken_mengtemp(buiten, recycle, klep)
        if mt < min_temp:
            klep2 = analytische_klep(buiten, recycle, min_temp)
            return min(klep, klep2)
        return klep
    else:
        max_temp = setpoint + WARM_OVERSHOOT
        mt = bereken_mengtemp(buiten, recycle, klep)
        if mt > max_temp:
            klep2 = analytische_klep(buiten, recycle, max_temp)
            return max(klep, klep2)
        return klep
# smartcontrol_ai.py  — Deel 3/4
# ----------------------------
# Overzicht:
# - WeerCache (weersvoorspelling-check)
# - EXTRA_OVERRIDE flag en SmartController inclusief CO2-protectie
# - Runtime logging van meetwaarden

# -------------------- WEER / NACHTKOELING --------------------
class WeerCache:
    """
    Houdt eenvoudige cache voor weervoorspelling zodat we niet te vaak de API aanroepen.
    Bepaalt of er 'warm_in_aantocht' is op basis van max temperatuur in VOORUITKIJK_DAGEN.
    """
    def __init__(self):
        self.laatste_check = 0.0
        self.warm_in_aantocht = False
        self.last_max_temp = None

    def check_weer_voorspelling(self):
        """Query open-meteo (indien nodig) en return boolean warm_in_aantocht."""
        nu = time.time()
        if nu - self.laatste_check < WEERVOORSPELLING_CHECK_INTERVAL:
            return self.warm_in_aantocht
        try:
            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={WEER_COORDINATEN['lat']}&longitude={WEER_COORDINATEN['lon']}"
                f"&daily=temperature_2m_max&timezone=Europe/Amsterdam"
            )
            r = requests.get(url, timeout=8)
            data = r.json()
            max_temp = max(data["daily"]["temperature_2m_max"][:VOORUITKIJK_DAGEN])
            self.last_max_temp = max_temp
            self.warm_in_aantocht = bool(max_temp >= HITTEGRENS)
            if DEBUG:
                write_log("DEBUG", f"[WEER] Max komende {VOORUITKIJK_DAGEN} dagen: {max_temp}°C → {'WARM' if self.warm_in_aantocht else 'OK'}")
        except Exception as e:
            if DEBUG:
                write_log("ERROR", f"[WEER] Fout bij ophalen weersvoorspelling: {e}")
            self.warm_in_aantocht = False
            self.last_max_temp = None
        self.laatste_check = nu
        return self.warm_in_aantocht

# -------------------- GLOBALE OVERRIDE --------------------
EXTRA_OVERRIDE = False  # globale vlag: als True, beperkt write_klep gedrag (prioriteit extra-lucht)

# -------------------- CONTROLLER --------------------
class SmartController:
    """
    Centrale controllerklasse:
    - beheert training checks
    - bevat step() die 1 regulatiecyclus doorloopt
    - start/stop extra lucht run (met merker synchronisatie)
    - bevat CO2-protectie: als CO2 > 1200 ppm -> klep 100%
    """
    def __init__(self, io, learner, weer):
        self.io = io
        self.learner = learner
        self.weer = weer
        self.last_excel_mtime = None
        self.extra_running = False
        self._initial_train()

    def _initial_train(self):
        """Initial training run (indien data aanwezig)."""
        if DEBUG: write_log("DEBUG", "[LEARN] Initial training opstarten...")
        ok = self.learner.fit_from_excel_and_runtime(EXCEL_PAD, RUNTIME_LOG, train_setpoint=TRAIN_SETPOINT)
        if DEBUG:
            write_log("DEBUG", f"[LEARN] Initial training resultaat: {'OK' if ok else 'GEEN DATA / FOUT'}")

    def _maybe_retrain_on_excel_change(self):
        """Controleer of Excel is gewijzigd en retrain indien dat zo is."""
        try:
            if EXCEL_PAD and os.path.exists(EXCEL_PAD):
                mtime = os.path.getmtime(EXCEL_PAD)
                if self.last_excel_mtime is None or mtime > self.last_excel_mtime:
                    if DEBUG:
                        write_log("DEBUG", "[LEARN] Excel gewijzigd → opnieuw trainen...")
                    ok = self.learner.fit_from_excel_and_runtime(EXCEL_PAD, RUNTIME_LOG, train_setpoint=TRAIN_SETPOINT)
                    self.last_excel_mtime = mtime
                    if DEBUG:
                        write_log("DEBUG", f"[LEARN] Fit: {'OK' if ok else 'GEEN DATA'}")
        except Exception as e:
            if DEBUG:
                write_log("ERROR", f"[LEARN] Fout retrain check: {e}")

    def _maybe_retrain_daily(self):
        """Dagelijkse retrain als model ouder is dan vandaag."""
        try:
            last_date = self.learner.last_fit_date()
            today = date.today()
            if last_date is None or last_date < today:
                if DEBUG:
                    write_log("DEBUG", f"[LEARN] Dagelijkse retrain: laatste fit {last_date} -> nu trainen.")
                ok = self.learner.fit_from_excel_and_runtime(EXCEL_PAD, RUNTIME_LOG, train_setpoint=TRAIN_SETPOINT)
                if DEBUG:
                    write_log("DEBUG", f"[LEARN] Dagelijkse retrain resultaat: {'OK' if ok else 'GEEN DATA'}")
        except Exception as e:
            if DEBUG:
                write_log("ERROR", f"[LEARN] Fout bij dagelijkse retrain: {e}")

    def _log_runtime(self, buiten, recycle, fitness, setpoint, klep, koelstand, co2=None):
        """
        Schrijf runtime-CSV met meetwaardes. Co2 is optioneel (voegt kolom toe indien aanwezig).
        Kolomvolgorde: timestamp,buiten,recycle,fitness,setpoint,klep,koelstand,co2
        """
        try:
            ensure_dir(os.path.dirname(RUNTIME_LOG))
            header_needed = not os.path.exists(RUNTIME_LOG)
            with open(RUNTIME_LOG, "a", encoding="utf-8") as f:
                if header_needed:
                    # header aangepast voor CO2-kolom
                    f.write("timestamp,buiten,recycle,fitness,setpoint,klep,koelstand,co2\n")
                co2_val = int(co2) if co2 is not None else ""
                f.write(f"{datetime.now().isoformat()},{buiten:.2f},{recycle:.2f},{fitness:.2f},{setpoint:.2f},{int(round(klep))},{koelstand},{co2_val}\n")
        except Exception:
            pass

    global EXTRA_OVERRIDE
    def _start_extra_lucht(self):
        """
        Start de extra-lucht-run:
        - zet merker (coil) in PLC indien mogelijk
        - forceert output_klep = 100% gedurende EXTRA_LUCHT_DUUR_SEC
        - respecteert het feit dat PLC-merker uit PLC de run kan stoppen
        - herstelt registers/merkers aan het einde
        """
        if self.extra_running:
            return
        self.extra_running = True
        EXTRA_OVERRIDE = True

        extra_addr = _get_addr("extra_lucht")

        def run():
            try:
                # Zorg dat merker aan staat (indien mogelijk)
                try:
                    if extra_addr is not None:
                        self.io.write_coil(extra_addr, True)
                        if DEBUG:
                            write_log("DEBUG", "[EXTRA] Merker (extra_lucht) gezet op 1 bij start.")
                except Exception as e:
                    if DEBUG:
                        write_log("ERROR", f"[EXTRA] Kon merker niet zetten bij start: {e}")

                # direct klep naar 100%
                ok = self.io.write_klep(_get_addr("output_klep"), 100)
                if INFO:
                    write_log("INFO", f"[EXTRA LUCHT] 100% voor {EXTRA_LUCHT_DUUR_SEC}s → {'OK' if ok else 'MISLUKT'}")
                t0 = time.time()
                while time.time() - t0 < EXTRA_LUCHT_DUUR_SEC:
                    # controleer of merker in PLC is uitgezet (dan stoppen we onmiddelijk)
                    try:
                        di_extra_now = self.io.read_coil(extra_addr, default=1)
                        if di_extra_now == 0:
                            if INFO or DEBUG:
                                write_log("INFO", "[EXTRA] Merker in PLC uitgezet -> stop extra lucht (onmiddellijk).")
                            break
                    except Exception:
                        pass
                    try:
                        self.io.write_klep(_get_addr("output_klep"), 100)
                    except:
                        pass
                    time.sleep(0.5)

                # einde extra: zet klep terug naar 0
                self.io.write_klep(_get_addr("output_klep"), 0)
                if DEBUG:
                    write_log("DEBUG", "[EXTRA] output_klep teruggezet naar 0")

                # zet hand_of_auto terug naar AUTO (coil False / 0)
                try:
                    hand_addr = _get_addr("hand_of_auto")
                    if hand_addr is not None:
                        self.io.write_coil(hand_addr, False)
                        if DEBUG:
                            write_log("DEBUG", "[EXTRA] hand_of_auto teruggezet naar AUTO (0).")
                except Exception as e:
                    if DEBUG:
                        write_log("ERROR", f"[EXTRA] kon hand_of_auto niet zetten: {e}")

                # zet instuur_lucht (VW52) -> gebruik INSTUUR_AFTER_EXTRA (raw value)
                try:
                    instuur_addr = _get_addr("instuur_lucht")
                    if instuur_addr is not None:
                        vw_value = int(INSTUUR_AFTER_EXTRA)
                        ok2 = self.io.write_register(instuur_addr, vw_value)
                        if DEBUG:
                            write_log("DEBUG", f"[EXTRA] instuur_lucht (addr {instuur_addr}) gezet op raw {vw_value} -> {'OK' if ok2 else 'MISLUKT'}")
                except Exception as e:
                    if DEBUG:
                        write_log("ERROR", f"[EXTRA] kon instuur_lucht niet terugzetten: {e}")

                if INFO:
                    write_log("INFO", "[EXTRA LUCHT] Klaar, terug naar automatische regeling.")

            finally:
                # herstel flags en forceer 1 regelstap zodat de controller onmiddellijk weer overneemt
                self.extra_running = False
                EXTRA_OVERRIDE = False
                # zorg dat merker wordt uitgezet
                try:
                    if extra_addr is not None:
                        self.io.write_coil(extra_addr, False)
                        if DEBUG:
                            write_log("DEBUG", "[EXTRA] Merker (extra_lucht) teruggezet op 0 bij einde.")
                except Exception as e:
                    if DEBUG:
                        write_log("ERROR", f"[EXTRA] kon merker niet uitzetten bij einde: {e}")
                try:
                    # voer direct een controlecyclus uit zodat er niet gewacht wordt op de volgende geplande stap
                    self.step()
                except Exception:
                    pass

        threading.Thread(target=run, daemon=True).start()

    def step(self):
        """
        Eén regel-cyclus:
        - retrain checks
        - lees hand/auto, koel/verwarm en sensoren (incl. CO2)
        - ALTIJD: CO2-protectie check vroeg (hoogste prioriteit)
        - nachtkoeling check
        - normale berekening: analytic + ML, blending, bandclamping, uitvoeren (indien auto)
        - logging naar runtime CSV en naar GUI logger
        """
        # retrain checks
        self._maybe_retrain_on_excel_change()
        self._maybe_retrain_daily()

        warm_in_aantocht = self.weer.check_weer_voorspelling() if WEER_NACHT_KOELING else False

        # hand/auto en verwarm/koel lezen
        handauto = self.io.read_coil(_get_addr("hand_of_auto"), default=0)
        koelverwarm = self.io.read_coil(_get_addr("koel_of_warm"), default=0)  # 0=koelen, 1=verwarmen

        # --- BELANGRIJKE WIJZIGING: lees extra_lucht als COIL, niet als discrete input ---
        extra_addr = _get_addr("extra_lucht")
        di_extra = self.io.read_coil(extra_addr, default=0)

        # Als merker in PLC gezet is: start extra-lucht en sla verdere regeling over (absolute prioriteit)
        if di_extra == 1:
            try:
                self._start_extra_lucht()
            except Exception as e:
                if DEBUG:
                    write_log("ERROR", f"[EXTRA] Fout bij starten extra lucht: {e}")
            return "[EXTRA] Merker actief -> extra lucht gestart en overige regeling overgeslagen.\n"

        # Als er al een override actief is (extra-run bezig), sla de reguliere regeling volledig over
        if EXTRA_OVERRIDE or self.extra_running:
            return "[EXTRA] Extra-lucht override actief — regeling tijdelijk uitgeschakeld.\n"

        # -------------------- NORMALE SENSORLEES --------------------
        instuur_addr = _get_addr("instuur_lucht")
        instuur = None
        if instuur_addr is not None:
            instuur = self.io.read_word(instuur_addr, scale=0.1, default=0.0)

        recycle = self.io.read_word(_get_addr("recycle_lucht"), scale=0.1, default=0.0)
        buiten = self.io.read_word(_get_addr("buitenlucht"), scale=0.1, default=0.0)
        fitness = self.io.read_word(_get_addr("fitness_temp"), scale=0.1, default=0.0)
        setpoint = float(self.io.read_word_raw(_get_addr("setpoint"), default=20))

        # === NIEUW: lees CO2-sensorwaarde (ppm) ===
        # We gebruiken read_word_raw zodat we de registerwaarde (ppm) direct krijgen.
        co2 = self.io.read_word_raw(_get_addr("co2_sensor"), default=400)

        # -------------------- CO2-BESLISSING (PRIORITEIT) --------------------
        # Als CO2 boven drempel, direct buitenluchtklep op 100% (veiligheidsmaatregel).
        CO2_THRESHOLD = 1200  # ppm, harde grens (kan naar SETTINGS indien gewenst)
        if co2 and int(co2) > CO2_THRESHOLD:
            # schrijf direct 100% naar output_klep
            ok = self.io.write_klep(_get_addr("output_klep"), 100)
            if INFO or DEBUG:
                write_log("INFO", f"[CO2] Hoog CO2-niveau gedetecteerd ({co2} ppm) → buitenluchtklep op 100% gezet.")
            # log runtime inclusief co2
            self._log_runtime(buiten, recycle, fitness, setpoint, 100, koelverwarm, co2=co2)
            instuur_txt = f"{instuur:.1f}°C" if instuur is not None else "N/A"
            return (f"Ingestuur: {instuur_txt} | Recycle: {recycle:.1f}°C | Buiten: {buiten:.1f}°C | "
                    f"Fitness: {fitness:.1f}°C | SP: {setpoint:.1f}°C | CO₂: {int(co2)} ppm\n"
                    f"Mode: {'Hand' if handauto else 'Auto'} | Stand: {'Verwarmen' if koelverwarm else 'Koelen'} | "
                    f"Klep: 100% (CO₂ > {CO2_THRESHOLD} ppm)\n")

        # -------------------- NACHTKOELING (prioriteit na CO2) --------------------
        if handauto == 0 and WEER_NACHT_KOELING and warm_in_aantocht and is_nacht():
            if koelverwarm == 0 and buiten < fitness:
                ok = self.io.write_klep(_get_addr("output_klep"), 100)
                if INFO or DEBUG:
                    write_log("INFO", "[NACHTKOELING] Warmte in aantocht + nacht + buiten < fitness → 100% buitenlucht")
                    write_log("INFO", f"[NACHTKOELING] write_klep => {'OK' if ok else 'MISLUKT'}")
                self._log_runtime(buiten, recycle, fitness, setpoint, 100, koelverwarm, co2=co2)
                instuur_txt = f"{instuur:.1f}°C" if instuur is not None else "N/A"
                return (f"Ingestuur: {instuur_txt} | Recycle: {recycle:.1f}°C | Buiten: {buiten:.1f}°C | "
                        f"Fitness: {fitness:.1f}°C | SP: {setpoint:.1f}°C\n"
                        f"Mode: {'Hand' if handauto else 'Auto'} | Stand: {'Verwarmen' if koelverwarm else 'Koelen'} | "
                        f"Klep: 100% (nachtkoeling)\n")

        # -------------------- NORMALE REGELING (ML + analytisch + blending) --------------------
        target = kies_target(fitness, setpoint, koelverwarm)
        klep_analytic = analytische_klep(buiten, recycle, target)
        klep_ml = self.learner.predict(buiten, recycle, fitness, setpoint, fallback_klep=klep_analytic, warm_flag=1 if warm_in_aantocht else 0)

        klep_raw = (0.5 * klep_ml + 0.5 * klep_analytic) * AGRESSIVITEIT
        klep_raw = max(0, min(100, klep_raw))

        try:
            # kleine bias afhankelijk van koelen/verwarmen
            if koelverwarm == 0:
                klep_raw = min(100.0, klep_raw + 5.0)
            else:
                klep_raw = max(0.0, klep_raw - 5.0)
        except Exception:
            pass

        klep_cmd = clamp_band(klep_raw, buiten, recycle, fitness, setpoint, koelverwarm)

        write_ok = False
        if handauto == 0:
            write_ok = self.io.write_klep(_get_addr("output_klep"), klep_cmd)

        # log runtime inclusief CO2 (indien bekend)
        self._log_runtime(buiten, recycle, fitness, setpoint, klep_cmd, koelverwarm, co2=co2)

        # bouw debug/info output string
        instuur_txt = f"{instuur:.1f}°C" if instuur is not None else "N/A"
        mt_after = bereken_mengtemp(buiten, recycle, klep_cmd)
        debug_lines = []
        debug_lines.append(f"Ingestuur: {instuur_txt} | Recycle: {recycle:.1f}°C | Buiten: {buiten:.1f}°C | Fitness: {fitness:.1f}°C | SP: {setpoint:.1f}°C")
        debug_lines.append(f"Mode: {'Hand' if handauto else 'Auto'} | Stand: {'Verwarmen' if koelverwarm else 'Koelen'}")
        debug_lines.append(f"Target temp: {target:.2f}°C")
        debug_lines.append(f"Klep (analytic): {klep_analytic:.1f}% | Klep (ML): {klep_ml:.1f}%")
        debug_lines.append(f"Klep (raw blended, agressiviteit {AGRESSIVITEIT}): {klep_raw:.1f}% | Klep (clamped/final): {klep_cmd:.1f}%")
        debug_lines.append(f"Predicted mengtemp (na commando): {mt_after:.2f}°C")
        debug_lines.append(f"CO2: {int(co2) if co2 is not None else 'N/A'} ppm")
        debug_lines.append(f"Weer-anticipatie (warm_in_aantocht): {'JA' if warm_in_aantocht else 'NEE'} (max voorspeld: {self.weer.last_max_temp if self.weer.last_max_temp is not None else 'N/A'})")
        debug_lines.append(f"Write-register resultaat: {'OK' if write_ok else 'MISLUKT of Geen client/adres'}")
        if DEBUG:
            debug_lines.append("(DEBUG aan: extra intern-output volgt...)")

        if INFO or DEBUG:
            if DEBUG:
                debug_lines.append(f"[DEBUG] Learner model present: {'ja' if self.learner.model is not None else 'nee'}")
            return "\n".join(debug_lines) + "\n"
        return ""
# smartcontrol_ai.py  — Deel 4/4
# ----------------------------
# Overzicht:
# - GUI (tkinter) met knoppen: Start/Stop, Extra lucht, Admin instellingen
# - Integratie met SmartController.step()
# - Opslaan van instellingen via GUI
# - Main-app start

# -------------------- GUI --------------------
class App(tk.Tk):
    """
    Tkinter GUI wrapper:
    - knoppen voor start/stop, extra-lucht
    - admin paneel (instellingen) verborgen totdat login
    - vervangende tekstweergave (compact of debug)
    """
    def __init__(self):
        super().__init__()
        self.title("SmartControl — Goorkensweg 6, Uden")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Modbus IO, learner en weer-cache initialiseren
        self.io = IO(PLC_IP, MODBUS_PORT)
        self.learner = SelfLearningController(MODEL_BESTAND)
        self.weer = WeerCache()
        self.ctrl = SmartController(self.io, self.learner, self.weer)

        self.running = False
        self.is_admin = False
        self.loop_thread = None

        self._build_ui()

        try:
            delay_ms = max(0, int(AUTO_START_DELAY_SEC) * 1000)
            self.after(delay_ms, self._auto_start_if_idle)
        except Exception:
            self.after(10000, self._auto_start_if_idle)

    def _auto_start_if_idle(self):
        """Start automatisch de regel-loop na een korte vertraging (configuratie)."""
        if not self.running:
            self._start_loop()

    def _build_ui(self):
        """Bouw GUI: Notebook met hoofdpaneel, admin en geavanceerd."""
        nb = ttk.Notebook(self)
        self.tab_main = ttk.Frame(nb)
        self.tab_admin = ttk.Frame(nb)
        self.tab_advanced = ttk.Frame(nb)
        nb.add(self.tab_main, text="Hoofdscherm")
        nb.add(self.tab_admin, text="Instellingen")
        nb.add(self.tab_advanced, text="Geavanceerde instellingen")
        nb.pack(expand=1, fill="both")

        frm = ttk.Frame(self.tab_main, padding=10)
        frm.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # admin login/logout
        self.btn_login = ttk.Button(frm, text="Log in als beheerder", command=self._admin_login)
        self.btn_login.grid(column=0, row=0, sticky="w")
        self.btn_logout = ttk.Button(frm, text="Log uit", command=self._admin_logout, state="disabled")
        self.btn_logout.grid(column=1, row=0, sticky="e")

        # start/stop knoppen
        self.btn_start = ttk.Button(frm, text="Start", command=self._start_loop)
        self.btn_start.grid(column=0, row=1, sticky="w", pady=5)
        self.btn_stop = ttk.Button(frm, text="Stop", command=self._stop_loop, state="disabled")
        self.btn_stop.grid(column=1, row=1, sticky="e", pady=5)

        # Extra-lucht knop (GUI zet merker in PLC en start lokale run)
        self.btn_extra = ttk.Button(frm, text="Extra lucht (100% voor korte tijd)", command=self._extra_lucht)
        self.btn_extra.grid(column=0, row=2, columnspan=2, sticky="ew", pady=5)

        # tekstgebied (vervangend; we tonen slechts de meest relevante regel tenzij DEBUG)
        self.txt = tk.Text(frm, height=18, width=100, wrap="none")
        self.txt.grid(column=0, row=3, columnspan=2, pady=10)
        self.txt.configure(state="normal")
        self.txt.insert(tk.END, "Log gestart...\n")
        self.txt.configure(state="disabled")

        self.lbl_info = ttk.Label(frm, text="Status: Idle")
        self.lbl_info.grid(column=0, row=4, columnspan=2, sticky="w")

        # ------------- Admin paneel (basisinstellingen) -------------
        adm = ttk.LabelFrame(self.tab_admin, text="Instellingen (basis)", padding=10)
        adm.pack(fill="both", expand=True)

        self.var_debug = tk.BooleanVar(value=DEBUG)
        self.var_auto = tk.BooleanVar(value=AUTOMATISCH_SCHAKELEN)
        self.chk_debug = ttk.Checkbutton(adm, text="Debugmodus", variable=self.var_debug, command=self._toggle_debug, state="normal")
        self.chk_auto = ttk.Checkbutton(adm, text="Automatisch schakelen", variable=self.var_auto, command=self._toggle_auto, state="disabled")
        self.chk_debug.grid(column=0, row=0, sticky="w", padx=3, pady=3)
        self.chk_auto.grid(column=1, row=0, sticky="w", padx=3, pady=3)

        base_labels = [
            ("AGRESSIVITEIT", str(AGRESSIVITEIT)),
            ("INTERVAL_SEC", str(INTERVAL_SEC)),
            ("KOEL_ONDERSHOOT", str(KOEL_ONDERSHOOT)),
            ("WARM_OVERSHOOT", str(WARM_OVERSHOOT)),
            ("EXTRA_LUCHT_DUUR_SEC", str(EXTRA_LUCHT_DUUR_SEC)),
            ("TRAIN_SETPOINT", str(TRAIN_SETPOINT)),
        ]
        self.entries_base = {}
        for i,(k,v) in enumerate(base_labels, start=1):
            ttk.Label(adm, text=k).grid(column=0, row=i, sticky="w", padx=3, pady=3)
            e = ttk.Entry(adm, width=40)
            e.insert(0, v)
            e.configure(state="readonly")
            e.grid(column=1, row=i, sticky="w")
            self.entries_base[k] = e

        self.btn_retrain = ttk.Button(adm, text="Nu opnieuw trainen", command=self._retrain_now, state="disabled")
        self.btn_save_base = ttk.Button(adm, text="Opslaan (basis) → settings.json", command=self._save_base, state="disabled")
        self.btn_retrain.grid(column=0, row=len(base_labels)+1, sticky="w", padx=3, pady=6)
        self.btn_save_base.grid(column=1, row=len(base_labels)+1, sticky="w", padx=3, pady=6)

        for i in range(2):
            adm.columnconfigure(i, weight=1)

        # ------------- Geavanceerde instellingen (read-only tenzij admin) -------------
        adv = ttk.LabelFrame(self.tab_advanced, text="Geavanceerd — alle variabelen", padding=10)
        adv.pack(fill="both", expand=True)

        canvas = tk.Canvas(adv)
        scrollbar = ttk.Scrollbar(adv, orient="vertical", command=canvas.yview)
        self.adv_inner = ttk.Frame(canvas)
        self.adv_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.adv_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.entries_adv = {}
        # Let op: we hebben ADDR_co2_sensor toegevoegd aan deze lijst
        advanced_keys = [
            "PLC_IP","MODBUS_PORT","INFO","DEBUG","AUTOMATISCH_SCHAKELEN",
            "WEER_NACHT_KOELING","HITTEGRENS","NACHTUREN",
            "WEER_COORD_LAT","WEER_COORD_LON","WEERVOORSPELLING_CHECK_INTERVAL","VOORUITKIJK_DAGEN",
            "DATA_DIR","EXCEL_PAD","MODEL_BESTAND","RUNTIME_LOG","TRAINING_DATA_FILE",
            "AUTO_START_DELAY_SEC",
            "ADDR_instuur_lucht","ADDR_recycle_lucht","ADDR_buitenlucht","ADDR_fitness_temp",
            "ADDR_setpoint","ADDR_output_klep","ADDR_hand_of_auto","ADDR_koel_of_warm","ADDR_extra_lucht","ADDR_co2_sensor","INSTUUR_AFTER_EXTRA"
        ]
        r = 0
        for k in advanced_keys:
            ttk.Label(self.adv_inner, text=k).grid(column=0, row=r, sticky="w", padx=4, pady=2)
            e = ttk.Entry(self.adv_inner, width=60)
            e.insert(0, str(SETTINGS.get(k, "")))
            e.configure(state="readonly")
            e.grid(column=1, row=r, sticky="w", padx=4, pady=2)
            self.entries_adv[k] = e
            r += 1

        self.btn_pick_excel = ttk.Button(self.adv_inner, text="Kies Excel...", command=self._pick_excel, state="disabled")
        self.btn_pick_excel.grid(column=0, row=r, sticky="w", padx=4, pady=6)
        self.btn_save_all = ttk.Button(self.adv_inner, text="Opslaan ALLES → settings.json", command=self._save_all, state="disabled")
        self.btn_save_all.grid(column=1, row=r, sticky="w", padx=4, pady=6)

        self.btn_shutdown = ttk.Button(self.adv_inner, text="Afsluiten", command=self._shutdown, state="disabled")
        self.btn_shutdown.grid(column=0, row=r+1, sticky="w", padx=4, pady=6)

        self._notebook = nb
        nb.tab(1, state="hidden")
        nb.tab(2, state="hidden")

    # -------------------- admin acties --------------------
    def _admin_login(self):
        pwd = simpledialog.askstring("Beheerder", "Voer wachtwoord in:", show="*")
        if pwd == "admin":
            self.is_admin = True
            self.chk_auto.configure(state="normal")
            for e in self.entries_base.values(): e.configure(state="normal")
            for e in self.entries_adv.values(): e.configure(state="normal")
            self.btn_retrain.configure(state="normal")
            self.btn_save_base.configure(state="normal")
            self.btn_pick_excel.configure(state="normal")
            self.btn_save_all.configure(state="normal")
            self.btn_shutdown.configure(state="normal")
            self._notebook.tab(1, state="normal")
            self._notebook.tab(2, state="normal")
            self._notebook.select(1)
            self.btn_login.configure(state="disabled")
            self.btn_logout.configure(state="normal")

    def _admin_logout(self):
        self.is_admin = False
        self.chk_auto.configure(state="disabled")
        for e in self.entries_base.values(): e.configure(state="readonly")
        for e in self.entries_adv.values(): e.configure(state="readonly")
        self.btn_retrain.configure(state="disabled")
        self.btn_save_base.configure(state="disabled")
        self.btn_pick_excel.configure(state="disabled")
        self.btn_save_all.configure(state="disabled")
        self.btn_shutdown.configure(state="disabled")
        self._notebook.tab(1, state="hidden")
        self._notebook.tab(2, state="hidden")
        self._notebook.select(0)
        self.btn_login.configure(state="normal")
        self.btn_logout.configure(state="disabled")

    def _toggle_debug(self):
        global DEBUG, SETTINGS
        DEBUG = self.var_debug.get()
        SETTINGS["DEBUG"] = DEBUG
        if DEBUG:
            self._append_text("[DEBUG] Debugmodus ingeschakeld\n")
            write_log("INFO", "[GUI] Debugmodus ingeschakeld")
        else:
            self._append_text("[DEBUG] Debugmodus uitgeschakeld\n")
            write_log("INFO", "[GUI] Debugmodus uitgeschakeld")

    def _toggle_auto(self):
        global AUTOMATISCH_SCHAKELEN, SETTINGS
        AUTOMATISCH_SCHAKELEN = self.var_auto.get()
        SETTINGS["AUTOMATISCH_SCHAKELEN"] = AUTOMATISCH_SCHAKELEN

    # -------------------- loop besturing --------------------
    def _start_loop(self):
        if self.running:
            return
        if not self.io.connected:
            messagebox.showwarning("Modbus", "Geen verbinding met PLC. Controleer IP/poort/adressen.")
        self.running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()

    def _stop_loop(self):
        self.running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _run_loop(self):
        """Cyclic runner die ctrl.step() aanroept en output toont in GUI (vervangend)."""
        while self.running:
            try:
                buf = io.StringIO()
                sys_stdout = sys.stdout
                sys.stdout = buf
                out = self.ctrl.step()
                sys.stdout = sys_stdout
                combined = (out or "") + buf.getvalue()
                self._append_text(combined)
                self.lbl_info.configure(text=f"Laatste cyclus: {datetime.now().strftime('%H:%M:%S')}")
                time.sleep(INTERVAL_SEC)
            except Exception as e:
                sys.stdout = sys.__stdout__
                self._append_text(f"[FOUT] {e}\n")
                self._stop_loop()
                break

    def _append_text(self, text):
        """
        Vervangende GUI-weergave:
        - Als DEBUG aan: toon volledige tekst
        - Anders: toon compacte eerste relevante regel (voorkeur voor 'Ingestuur')
        Daarnaast: schrijf elke getoonde regel naar CSV-log via write_log.
        """
        try:
            txt = str(text or "")
            if DEBUG:
                to_show = txt.strip()
            else:
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                chosen = None
                for l in lines:
                    if l.lower().startswith("ingestuur") or "ingestuur" in l.lower():
                        chosen = l
                        break
                if chosen is None:
                    chosen = lines[0] if lines else ""
                to_show = chosen

            # vervangende weergave (geen append)
            self.txt.configure(state="normal")
            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, to_show + "\n")
            self.txt.configure(state="disabled")

            # log iedere regel die we ontvingen (splitten en niveaus bepalen)
            for ln in txt.splitlines():
                line = ln.strip()
                if not line:
                    continue
                level = "INFO"
                if "FOUT" in line or "[FOUT]" in line or "ERROR" in line or "Exception" in line:
                    level = "ERROR"
                elif "[DEBUG]" in line or DEBUG:
                    level = "DEBUG" if "[DEBUG]" in line or line.startswith("[DEBUG]") or DEBUG else level
                elif "WAARSCHUWING" in line or "WARN" in line.upper():
                    level = "WARNING"
                else:
                    level = "INFO"
                write_log(level, line)

        except Exception as e:
            try:
                write_log("ERROR", f"[GUI_APPEND] Fout: {e}")
            except:
                pass

    def _extra_lucht(self):
        """
        GUI actie: zet merker in PLC en start lokale extra-lucht run.
        De lokale run zorgt ervoor dat merker later weer wordt uitgezet.
        """
        extra_addr = _get_addr("extra_lucht")
        try:
            if extra_addr is not None:
                ok = self.io.write_coil(extra_addr, True)
                if ok:
                    write_log("INFO", "[GUI] Extra-lucht knop ingedrukt: merker in PLC gezet op 1")
                else:
                    write_log("WARNING", "[GUI] Extra-lucht knop ingedrukt: kon merker niet zetten (write_coil mislukte)")
        except Exception as e:
            write_log("ERROR", f"[GUI] Kon merker niet zetten bij extra-lucht knop: {e}")
        # start altijd de lokale run; de run zet de merker later weer uit.
        self.ctrl._start_extra_lucht()

    # -------------------- trainingen & opslaan --------------------
    def _retrain_now(self):
        global TRAIN_SETPOINT
        TRAIN_SETPOINT = _get_float(self.entries_base["TRAIN_SETPOINT"].get(), TRAIN_SETPOINT)
        SETTINGS["TRAIN_SETPOINT"] = TRAIN_SETPOINT
        ok = self.learner.fit_from_excel_and_runtime(EXCEL_PAD, RUNTIME_LOG, train_setpoint=TRAIN_SETPOINT)
        messagebox.showinfo("Trainen", "Model getraind." if ok else "Geen bruikbare data gevonden (trainingsdata-bestand is wel aangemaakt).")
        save_settings()
        self._append_text(f"[LEARN] Handmatig retrain: {'OK' if ok else 'FAILED'}")

    def _save_base(self):
        global AGRESSIVITEIT, INTERVAL_SEC, KOEL_ONDERSHOOT, WARM_OVERSHOOT, EXTRA_LUCHT_DUUR_SEC, TRAIN_SETPOINT
        try:
            AGRESSIVITEIT = _get_float(self.entries_base["AGRESSIVITEIT"].get(), AGRESSIVITEIT)
            INTERVAL_SEC = int(float(self.entries_base["INTERVAL_SEC"].get()))
            KOEL_ONDERSHOOT = _get_float(self.entries_base["KOEL_ONDERSHOOT"].get(), KOEL_ONDERSHOOT)
            WARM_OVERSHOOT = _get_float(self.entries_base["WARM_OVERSHOOT"].get(), WARM_OVERSHOOT)
            EXTRA_LUCHT_DUUR_SEC = int(float(self.entries_base["EXTRA_LUCHT_DUUR_SEC"].get()))
            TRAIN_SETPOINT = _get_float(self.entries_base["TRAIN_SETPOINT"].get(), TRAIN_SETPOINT)

            SETTINGS.update({
                "AGRESSIVITEIT": AGRESSIVITEIT,
                "INTERVAL_SEC": INTERVAL_SEC,
                "KOEL_ONDERSHOOT": KOEL_ONDERSHOOT,
                "WARM_OVERSHOOT": WARM_OVERSHOOT,
                "EXTRA_LUCHT_DUUR_SEC": EXTRA_LUCHT_DUUR_SEC,
                "TRAIN_SETPOINT": TRAIN_SETPOINT
            })
            ok = save_settings()
            messagebox.showinfo("Instellingen", f"Basisinstellingen {'opgeslagen' if ok else 'niet opgeslagen'}.")
        except Exception as e:
            messagebox.showerror("Fout", f"Kon basisinstellingen niet opslaan: {e}")

    def _pick_excel(self):
        path = filedialog.askopenfilename(title="Kies Excel", filetypes=[("Excel", "*.xlsx;*.xls;*.csv")])
        if path:
            self.entries_adv["EXCEL_PAD"].delete(0, tk.END)
            self.entries_adv["EXCEL_PAD"].insert(0, path)

    def _save_all(self):
        """
        Sla alle geavanceerde instellingen op (admin only).
        Let op: ADDR_co2_sensor wordt meegenomen.
        """
        global PLC_IP, MODBUS_PORT, INFO, DEBUG, AUTOMATISCH_SCHAKELEN
        global WEER_NACHT_KOELING, HITTEGRENS, NACHTUREN, WEER_COORDINATEN
        global WEERVOORSPELLING_CHECK_INTERVAL, VOORUITKIJK_DAGEN
        global DATA_DIR, EXCEL_PAD, MODEL_BESTAND, RUNTIME_LOG, TRAINING_DATA_FILE
        global AUTO_START_DELAY_SEC, ADDR, INSTUUR_AFTER_EXTRA, LOG_FILE

        try:
            for k,e in self.entries_adv.items():
                SETTINGS[k] = e.get().strip()

            SETTINGS["MODBUS_PORT"] = int(float(SETTINGS["MODBUS_PORT"]))
            SETTINGS["INFO"] = _to_bool(SETTINGS["INFO"], True)
            SETTINGS["DEBUG"] = _to_bool(SETTINGS["DEBUG"], False)
            SETTINGS["AUTOMATISCH_SCHAKELEN"] = _to_bool(SETTINGS["AUTOMATISCH_SCHAKELEN"], True)
            SETTINGS["WEER_NACHT_KOELING"] = _to_bool(SETTINGS["WEER_NACHT_KOELING"], True)
            SETTINGS["HITTEGRENS"] = float(SETTINGS["HITTEGRENS"])
            SETTINGS["NACHTUREN"] = SETTINGS["NACHTUREN"]
            SETTINGS["WEER_COORD_LAT"] = float(SETTINGS["WEER_COORD_LAT"])
            SETTINGS["WEER_COORD_LON"] = float(SETTINGS["WEER_COORD_LON"])
            SETTINGS["WEERVOORSPELLING_CHECK_INTERVAL"] = int(float(SETTINGS["WEERVOORSPELLING_CHECK_INTERVAL"]))
            SETTINGS["VOORUITKIJK_DAGEN"] = int(float(SETTINGS["VOORUITKIJK_DAGEN"]))
            SETTINGS["AUTO_START_DELAY_SEC"] = int(float(SETTINGS["AUTO_START_DELAY_SEC"]))
            SETTINGS["INSTUUR_AFTER_EXTRA"] = int(float(SETTINGS.get("INSTUUR_AFTER_EXTRA", INSTUUR_AFTER_EXTRA)))

            DATA_DIR = SETTINGS["DATA_DIR"]
            ensure_dir(DATA_DIR)
            def abs_or_join(v):
                return v if os.path.isabs(v) else os.path.join(DATA_DIR, v)
            SETTINGS["EXCEL_PAD"] = abs_or_join(SETTINGS["EXCEL_PAD"])
            SETTINGS["MODEL_BESTAND"] = abs_or_join(SETTINGS["MODEL_BESTAND"])
            SETTINGS["RUNTIME_LOG"] = abs_or_join(SETTINGS["RUNTIME_LOG"])
            SETTINGS["TRAINING_DATA_FILE"] = abs_or_join(SETTINGS["TRAINING_DATA_FILE"])

            ok = save_settings()

            PLC_IP = SETTINGS["PLC_IP"]
            MODBUS_PORT = SETTINGS["MODBUS_PORT"]
            INFO = SETTINGS["INFO"]; DEBUG = SETTINGS["DEBUG"]
            AUTOMATISCH_SCHAKELEN = SETTINGS["AUTOMATISCH_SCHAKELEN"]
            WEER_NACHT_KOELING = SETTINGS["WEER_NACHT_KOELING"]
            HITTEGRENS = SETTINGS["HITTEGRENS"]
            NACHTUREN = _to_tuple_hours(SETTINGS["NACHTUREN"], (0,7))
            WEER_COORDINATEN = {"lat": SETTINGS["WEER_COORD_LAT"], "lon": SETTINGS["WEER_COORD_LON"]}
            WEERVOORSPELLING_CHECK_INTERVAL = SETTINGS["WEERVOORSPELLING_CHECK_INTERVAL"]
            VOORUITKIJK_DAGEN = SETTINGS["VOORUITKIJK_DAGEN"]
            DATA_DIR = SETTINGS["DATA_DIR"]
            EXCEL_PAD = SETTINGS["EXCEL_PAD"]
            MODEL_BESTAND = SETTINGS["MODEL_BESTAND"]
            RUNTIME_LOG = SETTINGS["RUNTIME_LOG"]
            TRAINING_DATA_FILE = SETTINGS["TRAINING_DATA_FILE"]
            AUTO_START_DELAY_SEC = SETTINGS["AUTO_START_DELAY_SEC"]
            INSTUUR_AFTER_EXTRA = int(SETTINGS.get("INSTUUR_AFTER_EXTRA", INSTUUR_AFTER_EXTRA))
            ADDR = {
                "instuur_lucht": SETTINGS["ADDR_instuur_lucht"],
                "recycle_lucht": int(SETTINGS["ADDR_recycle_lucht"]),
                "buitenlucht": int(SETTINGS["ADDR_buitenlucht"]),
                "fitness_temp": int(SETTINGS["ADDR_fitness_temp"]),
                "setpoint": int(SETTINGS["ADDR_setpoint"]),
                "output_klep": int(SETTINGS["ADDR_output_klep"]),
                "hand_of_auto": int(SETTINGS["ADDR_hand_of_auto"]),
                "koel_of_warm": int(SETTINGS["ADDR_koel_of_warm"]),
                "extra_lucht": int(SETTINGS["ADDR_extra_lucht"]),
                "co2_sensor": int(SETTINGS["ADDR_co2_sensor"]),
            }
            LOG_FILE = SETTINGS.get("LOG_FILE", LOG_FILE)

            messagebox.showinfo("Instellingen", f"Alle instellingen {'opgeslagen' if ok else 'niet opgeslagen'}.\nHerstart de app of de regel-loop voor volledige toepassing.")
        except Exception as e:
            messagebox.showerror("Fout", f"Kon geavanceerde instellingen niet opslaan: {e}")

    def _shutdown(self):
        pwd = simpledialog.askstring("Bevestiging", "Beheerderwachtwoord vereist om af te sluiten:", show="*")
        if pwd == "admin":
            self._stop_loop()
            self.io.close()
            self.destroy()
        else:
            messagebox.showerror("Fout", "Ongeldig wachtwoord")

    def _on_close(self):
        if not self.is_admin:
            messagebox.showwarning("Actie geblokkeerd", "Je moet als beheerder ingelogd zijn om af te sluiten.")
            return
        self._shutdown()

# -------------------- UTIL --------------------
def _get_float(s, default):
    """Veilige float-parsing (komma to punt)"""
    try:
        return float(str(s).replace(",", "."))
    except:
        return float(default)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    # waarschuw gebruiker als packages ontbreken
    if MISSING:
        print("[LET OP] Ontbrekende modules:", ", ".join(MISSING))
        print("Installeer met bijvoorbeeld:")
        print("  pip install pandas numpy pymodbus requests")
    load_settings()
    ensure_dir(DATA_DIR)
    app = App()
    app.mainloop()
