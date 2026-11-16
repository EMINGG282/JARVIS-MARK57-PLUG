# -*- coding: utf-8 -*-
"""
plugins/earth.py -- Interactive Mark LVII 3D Earth Globe inside the TOP HUD.

Drops an interactive HoloPanel directly into MainWindow._hud_cam_stack
(the big top area where the Reactor / Avatar sits), safely marshalled onto
the Qt main thread from this worker-thread plugin.

Primary renderer: CesiumJS 3D WGS84 WebGL Globe (QWebEngineView) with Google
Satellite / Streets / Labels, Esri fallback, NASA VIIRS Black Marble night
lights, solar terminator, atmospheric glow, country hover/click highlighting,
USGS earthquakes, live flights, and Web Worker SGP4 satellites.

Fallback renderer: Hardware-fast 3D Textured Orthographic Qt Globe (numpy +
QImage) using real stitched Google Satellite / Google Streets / NASA Black Marble
textures with real-time solar day/night terminator + slippy z=4..20 Google/Carto
tiles on zoom so every theme works cleanly even if QtWebEngine cannot attach.

All dependencies (PyQt6-WebEngine, numpy) and core globe assets are checked and
installed proactively at startup when the plugin loads.
"""
from __future__ import annotations

import atexit
import concurrent.futures
import http.server
import importlib
import importlib.util
import json
import math
import os
import re
import site
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Enable shared OpenGL contexts before QApplication is constructed so QtWebEngine
# initializes cleanly even when loaded dynamically.
try:
    from PyQt6.QtCore import QCoreApplication, Qt
    if QCoreApplication.instance() is None and hasattr(Qt.ApplicationAttribute, "AA_ShareOpenGLContexts"):
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
except Exception:
    pass


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_CACHE_DIR = _base_dir() / "memory" / "globe_cache"
_CESIUM_DIR = _CACHE_DIR / "cesium"
_ASSETS_DIR = _CACHE_DIR / "assets"

_LOCK = threading.Lock()
_INSTALL_LOCK = threading.Lock()
_INSTALL_IN_PROGRESS = False
_INSTALL_ATTEMPTED = False
_WEBENGINE_INSTALL_NOTE = ""
_LAST_INSTALL_PROG: tuple[int, str] | None = None
_ACTIVE_BRIDGE = None
_BRIDGE = None  # module-level ref so Qt objects aren't garbage-collected
_SERVER = None
_HUB = None

_USER_AGENT = "MarkLVII-EarthGlobe/2.1 (desktop-hud-telemetry)"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mKHJ]")

_CESIUM_VERSION = "1.147.0"
_CESIUM_TAG = "1.147"

PLUGIN = {
    "name": "earth_globe",
    "description": (
        "Interactive 3D Earth globe in the main TOP HUD area powered by Google Satellite/Maps, "
        "NASA city lights day/night terminator, live earthquakes, live global/regional flights, "
        "and satellites. Use action='open'/'show' to display it, 'close' to return to the "
        "reactor/face, 'focus' to fly to a city/address/coords/country/satellite/flight, "
        "'mode' to switch appearance (auto, city_lights, street, satellite, dark), "
        "'layers' to toggle layers, or 'refresh' to reload live feeds."
    ),
    "brief": "Interactive 3D Google/Cesium Earth globe: day/night city lights, streets, quakes, flights, satellites.",
    "tier": "extra",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "enum": ["open", "show", "close", "focus", "mode", "layers", "refresh", "status"],
                "description": (
                    "open/show = display globe in top HUD | close = return to reactor/face | "
                    "focus = fly to location/country/satellite/flight | mode = change basemap | "
                    "layers = enable/disable overlays | refresh = re-fetch feeds | status = summary"
                ),
            },
            "location": {
                "type": "STRING",
                "description": "City, street address, country, 'lat,lon', satellite name (e.g. 'ISS'), or flight callsign.",
            },
            "zoom": {
                "type": "STRING",
                "description": "Zoom level number (0-19) or preset: 'globe', 'region', 'city', 'street'.",
            },
            "map_mode": {
                "type": "STRING",
                "enum": ["auto", "city_lights", "street", "satellite", "dark"],
                "description": "auto = Day/Night Google Satellite + City Lights | city_lights | street | satellite | dark",
            },
            "layers": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                    "enum": ["earthquakes", "flights", "satellites", "city_lights", "streets"],
                },
                "description": "Optional list of active layers to enable.",
            },
            "sat_group": {
                "type": "STRING",
                "enum": ["all", "stations", "starlink", "nav", "weather"],
                "description": "Satellite catalog filter: stations (key craft, clean default) | all | starlink | nav | weather.",
            },
        },
        "required": ["action"],
    },
}


# -- Dependency Detection & Proactive Startup Auto-Install --------------------

def _log_sys(player, message: str, console: bool = False) -> None:
    if player is not None and hasattr(player, "write_log"):
        try:
            player.write_log(f"SYS: {message}")
        except Exception:
            pass
    if console and sys.stdout is not None:
        try:
            print(f"[EarthGlobe] {message}", flush=True)
        except Exception:
            pass


def _refresh_import_paths() -> None:
    """Ensure newly installed packages (including --user site) are visible to importlib."""
    importlib.invalidate_caches()
    try:
        user_site = site.getusersitepackages()
        if user_site and os.path.isdir(user_site):
            if user_site not in sys.path:
                site.addsitedir(user_site)
            pyqt_mod = sys.modules.get("PyQt6")
            if pyqt_mod is not None and hasattr(pyqt_mod, "__path__"):
                pyqt_user_dir = os.path.join(user_site, "PyQt6")
                if os.path.isdir(pyqt_user_dir) and pyqt_user_dir not in pyqt_mod.__path__:
                    pyqt_mod.__path__.append(pyqt_user_dir)
    except Exception:
        pass


def _is_webengine_installed() -> bool:
    """Check package presence safely without loading Qt DLLs on worker threads."""
    try:
        _refresh_import_paths()
        return (
            importlib.util.find_spec("PyQt6.QtWebEngineWidgets") is not None
            and importlib.util.find_spec("PyQt6.QtWebEngineCore") is not None
        )
    except Exception:
        return False


def _is_numpy_installed() -> bool:
    try:
        _refresh_import_paths()
        return importlib.util.find_spec("numpy") is not None
    except Exception:
        return False


def _short_wheel_name(raw_name: str) -> str:
    name = raw_name.strip().split("/")[-1]
    if "-" in name:
        return name.split("-")[0]
    return name[:32]


def _run_pip_with_progress(cmd: list[str], emit_progress, timeout_sec: float = 240.0) -> tuple[int, str, bool]:
    """Run pip install streaming stdout/stderr chunks for live progress parsing."""
    popen_kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "bufsize": 0,
    }
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_COLOR"] = "1"
    popen_kwargs["env"] = env

    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    proc = subprocess.Popen(cmd, **popen_kwargs)
    timed_out = threading.Event()

    def _kill_on_timeout():
        timed_out.set()
        try:
            proc.kill()
        except Exception:
            pass

    watchdog = threading.Timer(timeout_sec, _kill_on_timeout)
    watchdog.daemon = True
    watchdog.start()

    buf = bytearray()
    last_lines: list[str] = []
    raw_unsupported = False
    cur_file = "dependencies"
    last_emit_ts = 0.0
    best_pct = 4

    def _emit_monotonic(pct: int, msg: str, log_milestone: bool = False):
        nonlocal best_pct
        if pct >= 0:
            best_pct = max(best_pct, min(99, pct))
            out_pct = best_pct
        else:
            out_pct = -1
        emit_progress(out_pct, msg, log_milestone=log_milestone)

    def _handle_line(line: str):
        nonlocal cur_file, raw_unsupported, last_emit_ts
        ln = _ANSI_RE.sub("", line).strip()
        if not ln:
            return

        low = ln.lower()
        if "progress-bar" in low and ("no such option" in low or "invalid choice" in low or "unrecognized" in low):
            raw_unsupported = True

        m_raw = re.match(r"^Progress\s+(\d+)\s+of\s+(\d+)$", ln)
        if m_raw:
            cur_b = int(m_raw.group(1))
            tot_b = int(m_raw.group(2))
            now = time.monotonic()
            if tot_b > 0:
                dl_pct = max(0, min(100, int((cur_b * 100) / tot_b)))
                if tot_b >= 5_000_000:
                    bar_pct = 20 + int(dl_pct * 0.68)
                else:
                    bar_pct = 10 + int(dl_pct * 0.10)
                if dl_pct == 100 or (bar_pct >= best_pct + 2 and now - last_emit_ts >= 0.2):
                    cur_mb = cur_b / 1_048_576.0
                    tot_mb = tot_b / 1_048_576.0
                    _emit_monotonic(
                        bar_pct,
                        f"Downloading {cur_file}: {dl_pct}% ({cur_mb:.1f} / {tot_mb:.1f} MB)",
                        log_milestone=(dl_pct in (25, 50, 75, 100)),
                    )
                    last_emit_ts = now
            else:
                if now - last_emit_ts >= 0.6:
                    cur_mb = cur_b / 1_048_576.0
                    _emit_monotonic(-1, f"Downloading {cur_file} ({cur_mb:.1f} MB)...", log_milestone=False)
                    last_emit_ts = now
            return

        m_pct = re.search(r"(\d{1,3})%", ln)
        if m_pct and ("|" in ln or "mb" in low or "kb" in low):
            dl_pct = max(0, min(100, int(m_pct.group(1))))
            bar_pct = 12 + int(dl_pct * 0.76)
            now = time.monotonic()
            if dl_pct == 100 or (bar_pct >= best_pct + 4 and now - last_emit_ts >= 0.35):
                _emit_monotonic(bar_pct, f"Downloading {cur_file}: {dl_pct}%", log_milestone=(dl_pct in (50, 100)))
                last_emit_ts = now
            return

        last_lines.append(ln)
        if len(last_lines) > 12:
            last_lines.pop(0)

        if ln.startswith("Collecting "):
            pkg = ln[len("Collecting "):].split()[0]
            _emit_monotonic(6, f"Resolving {pkg}...", log_milestone=True)
        elif ln.startswith("Downloading "):
            parts = ln.split()
            if len(parts) >= 2:
                cur_file = _short_wheel_name(parts[1])
            size_hint = ln[ln.find("("):] if "(" in ln and ln.endswith(")") else ""
            _emit_monotonic(10, f"Downloading {cur_file} {size_hint}".strip(), log_milestone=True)
        elif ln.startswith("Installing collected packages"):
            _emit_monotonic(90, "Installing Globe dependencies...", log_milestone=True)
        elif ln.startswith("Successfully installed"):
            _emit_monotonic(98, "Finalizing 3D Earth Globe...", log_milestone=True)

    try:
        fd = proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 1024)
            except OSError:
                break
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                idx_n = buf.find(b"\n")
                idx_r = buf.find(b"\r")
                if idx_n == -1 and idx_r == -1:
                    break
                if idx_n == -1:
                    cut = idx_r
                elif idx_r == -1:
                    cut = idx_n
                else:
                    cut = min(idx_n, idx_r)
                line_bytes = bytes(buf[:cut])
                del buf[: cut + 1]
                _handle_line(line_bytes.decode("utf-8", errors="replace"))
        if buf:
            _handle_line(bytes(buf).decode("utf-8", errors="replace"))
    finally:
        watchdog.cancel()
        try:
            proc.stdout.close()
        except Exception:
            pass

    rc = proc.wait()
    if timed_out.is_set():
        return -1, "pip install timed out", raw_unsupported
    err_msg = last_lines[-1] if last_lines else f"exit code {rc}"
    return rc, err_msg, raw_unsupported


def _install_dependencies_worker(player=None, console: bool = False) -> tuple[bool, str]:
    """Install missing PyQt6-WebEngine & numpy while reporting live progress."""
    global _INSTALL_IN_PROGRESS, _WEBENGINE_INSTALL_NOTE, _LAST_INSTALL_PROG

    def _emit_progress(pct: int, text: str, log_milestone: bool = False):
        global _LAST_INSTALL_PROG
        _LAST_INSTALL_PROG = (int(pct), str(text))
        b = _ACTIVE_BRIDGE
        if b is not None:
            b._install_progress_sig.emit(int(pct), str(text))
        if log_milestone:
            _log_sys(player, f"[Earth Globe] {text}", console=console)

    def _finish(ok: bool, note: str) -> tuple[bool, str]:
        global _INSTALL_IN_PROGRESS, _WEBENGINE_INSTALL_NOTE, _LAST_INSTALL_PROG
        _WEBENGINE_INSTALL_NOTE = note
        _LAST_INSTALL_PROG = None
        with _INSTALL_LOCK:
            _INSTALL_IN_PROGRESS = False
        b = _ACTIVE_BRIDGE
        if b is not None:
            b._install_done_sig.emit(bool(ok), str(note))
        return ok, note

    need_web = not _is_webengine_installed()
    need_np = not _is_numpy_installed()
    if not need_web and not need_np:
        return _finish(True, "")

    if getattr(sys, "frozen", False):
        note = "PyQt6-WebEngine is not bundled in this frozen build; using 3D Qt Globe."
        _log_sys(player, note, console=console)
        return _finish(False, note)

    _emit_progress(2, "Installing missing Earth Globe dependencies...", log_milestone=True)

    extra_pkgs = ["numpy"] if need_np else []
    pkg_sets: list[list[str]] = []

    if need_web:
        try:
            from PyQt6.QtCore import PYQT_VERSION_STR
            parts = PYQT_VERSION_STR.split(".")
            pkg_sets.append([f"PyQt6-WebEngine=={PYQT_VERSION_STR}"] + extra_pkgs)
            if len(parts) >= 2:
                pkg_sets.append([f"PyQt6-WebEngine=={parts[0]}.{parts[1]}.*"] + extra_pkgs)
        except Exception:
            pass
        pkg_sets.append(["PyQt6-WebEngine"] + extra_pkgs)
    elif extra_pkgs:
        pkg_sets.append(extra_pkgs)

    in_venv = sys.prefix != sys.base_prefix
    last_err = ""
    installed = False
    last_log_ts = 0.0

    def _throttled_progress(pct: int, text: str, log_milestone: bool = False):
        nonlocal last_log_ts
        now = time.monotonic()
        should_log = log_milestone and (now - last_log_ts >= 0.8 or pct >= 90 or "Downloading" in text)
        if should_log:
            last_log_ts = now
        _emit_progress(pct, text, log_milestone=should_log)

    for pkgs in pkg_sets:
        base_variants = [
            [sys.executable, "-m", "pip", "install", "--no-input", "--no-color", "--progress-bar=raw"] + pkgs,
        ]
        if not in_venv:
            base_variants.append(
                [sys.executable, "-m", "pip", "install", "--user", "--no-input", "--no-color", "--progress-bar=raw"] + pkgs
            )

        for cmd in base_variants:
            rc, err_line, raw_unsupported = _run_pip_with_progress(cmd, _throttled_progress)
            if raw_unsupported and rc != 0:
                fallback_cmd = [c for c in cmd if not c.startswith("--progress-bar")]
                rc, err_line, _ = _run_pip_with_progress(fallback_cmd, _throttled_progress)

            if rc == 0:
                installed = True
                break
            last_err = err_line
        if installed:
            break

    if not installed:
        note = f"Dependency auto-install failed ({last_err or 'pip error'}); using 3D Qt Globe."
        _log_sys(player, note, console=console)
        return _finish(False, note)

    if _is_webengine_installed():
        note = "Earth Globe dependencies installed successfully."
        _emit_progress(100, note, log_milestone=True)
        return _finish(True, note)

    note = "Dependencies installed (restart app if WebGL requires a fresh Qt session)."
    _log_sys(player, note, console=console)
    return _finish(False, note)


def _start_dependency_install(player=None, console: bool = False) -> str:
    """Trigger background installation if PyQt6-WebEngine or numpy is missing."""
    global _INSTALL_IN_PROGRESS, _INSTALL_ATTEMPTED, _LAST_INSTALL_PROG
    if _is_webengine_installed() and _is_numpy_installed():
        return ""
    with _INSTALL_LOCK:
        if _INSTALL_IN_PROGRESS:
            b = _ACTIVE_BRIDGE
            if b is not None and _LAST_INSTALL_PROG is not None:
                b._install_progress_sig.emit(*_LAST_INSTALL_PROG)
            return "Installing Earth Globe dependencies in background"
        if _INSTALL_ATTEMPTED and player is None:
            return _WEBENGINE_INSTALL_NOTE
        if _INSTALL_ATTEMPTED and _WEBENGINE_INSTALL_NOTE and "installed successfully" in _WEBENGINE_INSTALL_NOTE:
            return _WEBENGINE_INSTALL_NOTE
        _INSTALL_ATTEMPTED = True
        _INSTALL_IN_PROGRESS = True

    _LAST_INSTALL_PROG = (2, "Preparing Earth Globe dependency install...")
    b = _ACTIVE_BRIDGE
    if b is not None:
        b._install_progress_sig.emit(*_LAST_INSTALL_PROG)

    threading.Thread(
        target=_install_dependencies_worker,
        args=(player, console),
        daemon=True,
    ).start()
    return "Installing Earth Globe dependencies in background (live progress shown in HUD)"


# -- Offline Fallback Seed Data & Quick City Index ----------------------------

_QUICK_PLACES = {
    "dubai": (25.2048, 55.2708, 12.0),
    "abu dhabi": (24.4539, 54.3773, 12.0),
    "new york": (40.7128, -74.0060, 12.0),
    "london": (51.5072, -0.1276, 12.0),
    "tokyo": (35.6762, 139.6503, 12.0),
    "paris": (48.8566, 2.3522, 12.0),
    "san francisco": (37.7749, -122.4194, 12.0),
    "singapore": (1.3521, 103.8198, 12.0),
    "sydney": (-33.8688, 151.2093, 12.0),
    "cairo": (30.0444, 31.2357, 12.0),
    "riyadh": (24.7136, 46.6753, 12.0),
    "los angeles": (34.0522, -118.2437, 12.0),
    "mauritania": (20.2540, -10.9408, 5.2),
    "earth": (20.0, 10.0, 1.8),
    "globe": (20.0, 10.0, 1.8),
}

_FALLBACK_TLES = """ISS (ZARYA)
1 25544U 98067A   26317.50000000  .00016717  00000-0  30000-3 0  9991
2 25544  51.6416 210.4500 0004500 120.0000 240.0000 15.50100000480001
CSS (TIANHE)
1 48274U 21035A   26317.50000000  .00021000  00000-0  22000-3 0  9995
2 48274  41.4740  85.2100 0003900  45.0000 315.0000 15.60500000310002
HST
1 20580U 90037B   26317.50000000  .00001500  00000-0  70000-4 0  9993
2 20580  28.4690 150.1000 0002800 200.0000 160.0000 15.09200000900004
NOAA 19
1 33591U 09005A   26317.50000000  .00000110  00000-0  82000-4 0  9997
2 33591  99.1500 300.5000 0014000  95.0000 265.0000 14.12400000910009
GPS BIIR-2 (PRN 13)
1 24876U 97035A   26317.50000000 -.00000030  00000-0  00000-0 0  9992
2 24876  55.4300  60.2000 0045000 210.0000 150.0000  2.00560000210003
STARLINK-1007
1 44713U 19074A   26317.50000000  .00001200  00000-0  95000-4 0  9998
2 44713  53.0540 112.4000 0001500  85.0000 275.0000 15.06400000380007
"""


# -- Pure-Python TLE Parser & J2 Orbital Propagator ---------------------------

_MU_EARTH = 398600.4418
_RE_KM = 6378.137
_J2 = 1.08262668e-3


def _tle_epoch_to_jd(epoch_str: str) -> float:
    s = epoch_str.strip()
    yy = int(s[:2])
    year = (2000 + yy) if yy < 57 else (1900 + yy)
    day_of_year = float(s[2:])
    y = year - 1
    a = y // 100
    b = 2 - a + (a // 4)
    jd_jan0 = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * 14) + b - 1524.5
    return jd_jan0 + day_of_year


def _datetime_to_jd(dt: datetime) -> float:
    u = dt.astimezone(timezone.utc)
    y = u.year
    m = u.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + (a // 4)
    day_frac = (u.hour + (u.minute + (u.second + u.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + u.day + day_frac + b - 1524.5


def _classify_satellite(name: str) -> str:
    up = name.upper()
    if any(k in up for k in ("ISS", "ZARYA", "TIANHE", "CSS", "TIANGONG", "HST", "JWST", "SOYUZ", "DRAGON")):
        return "stations"
    if "STARLINK" in up or "ONEWEB" in up:
        return "starlink"
    if any(k in up for k in ("GPS", "NAVSTAR", "GLONASS", "GALILEO", "BEIDOU", "GSAT")):
        return "nav"
    if any(k in up for k in ("NOAA", "GOES", "METEOSAT", "METEOR", "HIMAWARI", "FENGYUN", "TERRA", "AQUA", "SUOMI")):
        return "weather"
    return "other"


def _parse_tles(raw_text: str) -> list[dict]:
    lines = [ln.rstrip("\r\n") for ln in raw_text.splitlines() if ln.strip()]
    sats: list[dict] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name = f"SAT-{ln[2:7].strip()}"
            l1, l2 = ln, lines[i + 1]
            i += 2
        elif (
            not ln.startswith(("1 ", "2 "))
            and i + 2 < len(lines)
            and lines[i + 1].startswith("1 ")
            and lines[i + 2].startswith("2 ")
        ):
            name = ln.strip()
            if name.startswith("0 "):
                name = name[2:].strip()
            l1, l2 = lines[i + 1], lines[i + 2]
            i += 3
        else:
            i += 1
            continue

        try:
            norad = l1[2:7].strip()
            epoch_jd = _tle_epoch_to_jd(l1[18:32])
            inc_deg = float(l2[8:16])
            raan_deg = float(l2[17:25])
            ecc = float("0." + l2[26:33].strip())
            argp_deg = float(l2[34:42])
            ma_deg = float(l2[43:51])
            mm_rev_day = float(l2[52:63])
            if mm_rev_day <= 0 or ecc >= 0.99:
                continue
            n_rad_s = mm_rev_day * 2.0 * math.pi / 86400.0
            a_km = (_MU_EARTH / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
            p_km = a_km * (1.0 - ecc * ecc)
            inc_rad = math.radians(inc_deg)
            j2_fac = 1.5 * _J2 * ((_RE_KM / max(p_km, 1.0)) ** 2) * n_rad_s
            raan_dot = -j2_fac * math.cos(inc_rad)
            argp_dot = 0.5 * j2_fac * (5.0 * math.cos(inc_rad) ** 2 - 1.0)
            sats.append({
                "id": norad,
                "name": name,
                "cat": _classify_satellite(name),
                "l1": l1,
                "l2": l2,
                "epoch_jd": epoch_jd,
                "inc_deg": round(inc_deg, 2),
                "inc": inc_rad,
                "raan0": math.radians(raan_deg),
                "ecc": ecc,
                "argp0": math.radians(argp_deg),
                "ma0": math.radians(ma_deg),
                "n": n_rad_s,
                "a": a_km,
                "raan_dot": raan_dot,
                "argp_dot": argp_dot,
            })
        except Exception:
            continue
    return sats


def _propagate_sat_geodetic(sat: dict, jd_now: float) -> tuple[float, float, float, float] | None:
    dt_sec = (jd_now - sat["epoch_jd"]) * 86400.0
    e = sat["ecc"]
    M = (sat["ma0"] + sat["n"] * dt_sec) % (2.0 * math.pi)
    E = M + e * math.sin(M)
    for _ in range(4):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        if abs(fp) > 1e-10:
            E -= f / fp

    sin_E = math.sin(E)
    cos_E = math.cos(E)
    sqrt_1me2 = math.sqrt(max(1e-12, 1.0 - e * e))
    nu = math.atan2(sqrt_1me2 * sin_E, cos_E - e)
    r = sat["a"] * (1.0 - e * cos_E)
    if r <= 6378.0:
        return None

    u = sat["argp0"] + sat["argp_dot"] * dt_sec + nu
    raan = sat["raan0"] + sat["raan_dot"] * dt_sec
    inc = sat["inc"]

    cos_raan = math.cos(raan)
    sin_raan = math.sin(raan)
    cos_u = math.cos(u)
    sin_u = math.sin(u)
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)

    x_eci = r * (cos_raan * cos_u - sin_raan * sin_u * cos_i)
    y_eci = r * (sin_raan * cos_u + cos_raan * sin_u * cos_i)
    z_eci = r * (sin_u * sin_i)

    d = jd_now - 2451545.0
    t_cent = d / 36525.0
    gmst_deg = (280.46061837 + 360.98564736629 * d + 0.000387933 * t_cent * t_cent) % 360.0
    gmst = math.radians(gmst_deg)

    lon_rad = math.atan2(y_eci, x_eci) - gmst
    lon_deg = ((math.degrees(lon_rad) + 180.0) % 360.0) - 180.0
    lat_deg = math.degrees(math.atan2(z_eci, math.hypot(x_eci, y_eci)))
    alt_km = r - 6371.0
    vel_kms = math.sqrt(max(0.0, _MU_EARTH * (2.0 / r - 1.0 / sat["a"])))
    if not (math.isfinite(lon_deg) and math.isfinite(lat_deg)):
        return None
    return (round(lon_deg, 4), round(lat_deg, 4), round(alt_km, 1), round(vel_kms, 2))


# -- Telemetry & Asset Data Hub (HTTP Fetch + Disk Cache) ---------------------

def _http_get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _guess_mime(subpath: str) -> str:
    low = subpath.lower().split("?", 1)[0]
    if low.endswith((".js", ".mjs", ".cjs")):
        return "application/javascript; charset=utf-8"
    if low.endswith(".css"):
        return "text/css; charset=utf-8"
    if low.endswith(".wasm"):
        return "application/wasm"
    if low.endswith((".json", ".geojson")):
        return "application/json; charset=utf-8"
    if low.endswith(".png"):
        return "image/png"
    if low.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if low.endswith(".svg"):
        return "image/svg+xml"
    if low.endswith(".woff2"):
        return "font/woff2"
    if low.endswith(".woff"):
        return "font/woff"
    if low.endswith(".gif"):
        return "image/gif"
    if low.endswith(".glsl"):
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def _tile_urls_for_layer(layer_tag: str, z: int, x: int, y: int) -> tuple[str, ...]:
    s = (x + y) % 4
    if layer_tag == "sat":
        return (
            f"https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        )
    if layer_tag == "hybrid":
        return (
            f"https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            f"https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        )
    if layer_tag == "street":
        return (
            f"https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        )
    if layer_tag == "lights":
        if z <= 8:
            return (
                f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_Black_Marble/default/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png",
                f"https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
            )
        return (f"https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",)
    return (
        f"https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    )


class _GlobeDataHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._tile_lock = threading.Lock()
        self._asset_lock = threading.Lock()
        self._tex_lock = threading.Lock()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CESIUM_DIR.mkdir(parents=True, exist_ok=True)
        _ASSETS_DIR.mkdir(parents=True, exist_ok=True)

        self.earthquakes: dict = {"type": "FeatureCollection", "features": [], "meta": {"source": "pending"}}
        self.flights: dict = {"type": "FeatureCollection", "features": [], "meta": {"source": "pending"}}
        self.sat_tles: list[dict] = _parse_tles(_FALLBACK_TLES)
        self.sat_meta: dict = {"count": len(self.sat_tles), "source": "seed", "updated": ""}
        self.countries_bytes: bytes | None = None
        self._country_rings: list[list[tuple[float, float]]] | None = None

        self._eq_ts = 0.0
        self._flt_ts = 0.0
        self._flt_backoff_until = 0.0
        self._flt_is_regional = False
        self._last_flt_center = (25.2, 55.27)
        self._sat_ts = 0.0
        self._view_lat = 20.0
        self._view_lon = 12.0
        self._selected: dict | None = None
        self._tile_cache: dict[str, bytes] = {}
        self._geocode_cache: dict[str, dict] = {}
        self._world_tex_building: set[str] = set()
        self._fetching_eq = False
        self._fetching_flt = False
        self._fetching_sat = False
        self._prewarmed = False

        self._load_cached_tles()
        self._load_cached_countries()

    def set_view_center(self, lat: float, lon: float) -> None:
        lat = max(-89.9, min(89.9, float(lat)))
        lon = ((float(lon) + 180.0) % 360.0) - 180.0
        with self._lock:
            self._view_lat = lat
            self._view_lon = lon
            if self._flt_is_regional:
                dlat = abs(lat - self._last_flt_center[0])
                dlon = abs(lon - self._last_flt_center[1])
                if dlat > 3.0 or dlon > 3.0:
                    self._flt_ts = 0.0

    def set_selected(self, item: dict | None) -> None:
        with self._lock:
            self._selected = item

    def get_selected(self) -> dict | None:
        with self._lock:
            return dict(self._selected) if self._selected else None

    def _load_cached_tles(self) -> None:
        tle_path = _CACHE_DIR / "celestrak_active.tle"
        if tle_path.exists():
            try:
                raw = tle_path.read_text(encoding="utf-8", errors="replace")
                parsed = _parse_tles(raw)
                if parsed:
                    self.sat_tles = parsed
                    mtime = tle_path.stat().st_mtime
                    age_h = (time.time() - mtime) / 3600.0
                    self.sat_meta = {
                        "count": len(parsed),
                        "source": f"cache ({age_h:.1f}h old)",
                        "updated": datetime.fromtimestamp(mtime).strftime("%H:%M"),
                    }
                    if age_h < 6.0:
                        self._sat_ts = mtime
            except Exception:
                pass

    def _load_cached_countries(self) -> None:
        cpath = _CACHE_DIR / "countries.geojson"
        if cpath.exists():
            try:
                raw = cpath.read_bytes()
                if len(raw) > 1000:
                    self.countries_bytes = raw
                    self._build_country_rings(raw)
            except Exception:
                pass

    def _build_country_rings(self, raw: bytes) -> None:
        try:
            gj = json.loads(raw.decode("utf-8"))
            rings: list[list[tuple[float, float]]] = []
            for feat in gj.get("features") or []:
                geom = feat.get("geometry") or {}
                gtype = geom.get("type")
                coords = geom.get("coordinates") or []
                polys = [coords] if gtype == "Polygon" else (coords if gtype == "MultiPolygon" else [])
                for poly in polys:
                    if not poly:
                        continue
                    outer = poly[0]
                    step = max(1, len(outer) // 48)
                    ring = [(float(pt[1]), float(pt[0])) for pt in outer[::step] if len(pt) >= 2]
                    if len(ring) >= 3:
                        rings.append(ring)
            with self._lock:
                self._country_rings = rings
        except Exception:
            pass

    def get_country_rings(self) -> list[list[tuple[float, float]]]:
        with self._lock:
            return self._country_rings or []

    def get_countries_geojson_bytes(self) -> bytes:
        with self._lock:
            if self.countries_bytes is not None:
                return self.countries_bytes

        cpath = _CACHE_DIR / "countries.geojson"
        urls = (
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
            "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/geojson/ne_110m_admin_0_countries.geojson",
            "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json",
        )
        for url in urls:
            try:
                raw = _http_get(url, timeout=12.0)
                if len(raw) > 5000:
                    cpath.write_bytes(raw)
                    with self._lock:
                        self.countries_bytes = raw
                    self._build_country_rings(raw)
                    return raw
            except Exception:
                continue
        return b'{"type":"FeatureCollection","features":[]}'

    def fetch_cached_cesium_file(self, subpath: str) -> bytes | None:
        """Serve CesiumJS Build/Cesium/* from local cache or fetch & cache on demand."""
        rel = subpath.lstrip("/").split("?", 1)[0]
        if not rel or ".." in rel.split("/"):
            return None

        target = (_CESIUM_DIR / rel).resolve()
        try:
            if not target.is_relative_to(_CESIUM_DIR.resolve()):
                return None
        except Exception:
            return None

        if target.exists() and target.stat().st_size > 0:
            try:
                return target.read_bytes()
            except Exception:
                pass

        with self._asset_lock:
            if target.exists() and target.stat().st_size > 0:
                try:
                    return target.read_bytes()
                except Exception:
                    pass

            urls = (
                f"https://cdn.jsdelivr.net/npm/cesium@{_CESIUM_VERSION}/Build/Cesium/{rel}",
                f"https://cesium.com/downloads/cesiumjs/releases/{_CESIUM_TAG}/Build/Cesium/{rel}",
                f"https://unpkg.com/cesium@{_CESIUM_VERSION}/Build/Cesium/{rel}",
            )
            for url in urls:
                try:
                    data = _http_get(url, timeout=25.0)
                    if data:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tmp = target.with_suffix(target.suffix + ".tmp")
                        tmp.write_bytes(data)
                        tmp.replace(target)
                        return data
                except Exception:
                    continue
        return None

    def fetch_cached_asset(self, filename: str, urls: tuple[str, ...]) -> bytes | None:
        target = (_ASSETS_DIR / filename).resolve()
        if target.exists() and target.stat().st_size > 0:
            try:
                return target.read_bytes()
            except Exception:
                pass

        with self._asset_lock:
            if target.exists() and target.stat().st_size > 0:
                try:
                    return target.read_bytes()
                except Exception:
                    pass
            for url in urls:
                try:
                    data = _http_get(url, timeout=20.0)
                    if data:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tmp = target.with_suffix(target.suffix + ".tmp")
                        tmp.write_bytes(data)
                        tmp.replace(target)
                        return data
                except Exception:
                    continue
        return None

    def get_world_texture_path(self, layer_tag: str) -> Path:
        return _CACHE_DIR / f"world_{layer_tag}_z3.png"

    def ensure_world_texture(self, layer_tag: str, on_ready=None) -> Path | None:
        """Build & disk-cache a 1536x768 equirectangular world map from z=3 Google/NASA tiles."""
        out_path = self.get_world_texture_path(layer_tag)
        if out_path.exists() and out_path.stat().st_size > 8000:
            return out_path

        with self._tex_lock:
            if layer_tag in self._world_tex_building:
                return None
            self._world_tex_building.add(layer_tag)

        def _build():
            try:
                from PyQt6.QtGui import QImage, QPainter
                z = 3
                grid = 2 ** z  # 8x8 = 64 tiles -> 2048x2048 Mercator
                coords = [(x, y) for y in range(grid) for x in range(grid)]

                def _dl(xy):
                    tx, ty = xy
                    return (tx, ty, self.fetch_tile(layer_tag, z, tx, ty))

                merc = QImage(grid * 256, grid * 256, QImage.Format.Format_RGBA8888)
                merc.fill(0)
                p = QPainter(merc)
                ok_count = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    for tx, ty, raw in pool.map(_dl, coords):
                        if raw:
                            tile_img = QImage.fromData(raw)
                            if not tile_img.isNull():
                                p.drawImage(tx * 256, ty * 256, tile_img.scaled(256, 256))
                                ok_count += 1
                p.end()

                if ok_count < 16:
                    return

                # Reproject Web Mercator (2048x2048) -> Equirectangular (1536x768)
                w_eq, h_eq = 1536, 768
                h_merc = merc.height()
                merc_scaled = merc.scaled(w_eq, h_merc)
                equi = QImage(w_eq, h_eq, QImage.Format.Format_RGBA8888)
                pe = QPainter(equi)
                for r in range(h_eq):
                    lat_deg = 90.0 - ((r + 0.5) * 180.0 / h_eq)
                    lat_clamped = max(-85.051128, min(85.051128, lat_deg))
                    lat_rad = math.radians(lat_clamped)
                    y_norm = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) * 0.5
                    mr = min(h_merc - 1, max(0, int(y_norm * h_merc)))
                    pe.drawImage(0, r, merc_scaled, 0, mr, w_eq, 1)
                pe.end()

                tmp = out_path.with_suffix(".tmp.png")
                if equi.save(str(tmp), "PNG"):
                    tmp.replace(out_path)
                    if on_ready is not None:
                        on_ready()
            except Exception:
                pass
            finally:
                with self._tex_lock:
                    self._world_tex_building.discard(layer_tag)

        threading.Thread(target=_build, daemon=True).start()
        return None

    def prewarm_assets(self) -> None:
        """Download and cache core JS/GeoJSON/world-texture assets right at app startup."""
        with self._lock:
            if self._prewarmed:
                return
            self._prewarmed = True

        try:
            self.fetch_cached_asset(
                "satellite.min.js",
                (
                    "https://cdn.jsdelivr.net/npm/satellite.js@4.0.0/dist/satellite.min.js",
                    "https://unpkg.com/satellite.js@4.0.0/dist/satellite.min.js",
                ),
            )
            self.fetch_cached_cesium_file("Widgets/widgets.css")
            self.fetch_cached_cesium_file("Cesium.js")
            self.get_countries_geojson_bytes()
            self.ensure_world_texture("sat")
            self.ensure_world_texture("lights")
            self.ensure_world_texture("street")
            self.ensure_earthquakes()
            self.ensure_satellites()
        except Exception:
            pass

    def ensure_earthquakes(self, force: bool = False) -> dict:
        now = time.time()
        with self._lock:
            eq = self.earthquakes
            need = force or (now - self._eq_ts >= 120.0) or not eq.get("features")
            if need and not self._fetching_eq:
                self._fetching_eq = True
                threading.Thread(target=self._fetch_earthquakes, daemon=True).start()
            return eq

    def _fetch_earthquakes(self) -> None:
        cache_path = _CACHE_DIR / "earthquakes.json"
        try:
            raw = _http_get("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson", timeout=12.0)
            data = json.loads(raw.decode("utf-8"))
            feats = []
            for f in data.get("features", []):
                try:
                    geom = f.get("geometry") or {}
                    coords = geom.get("coordinates") or []
                    props = f.get("properties") or {}
                    if len(coords) < 2:
                        continue
                    lon = float(coords[0])
                    lat = float(coords[1])
                    if not (math.isfinite(lon) and math.isfinite(lat)):
                        continue
                    mag = props.get("mag")
                    mag_f = float(mag) if mag is not None else 0.0
                    dep_f = float(coords[2]) if len(coords) > 2 and coords[2] is not None else 0.0
                    feats.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "properties": {
                            "id": str(f.get("id") or ""),
                            "mag": round(mag_f, 1),
                            "depth": round(dep_f, 1),
                            "place": str(props.get("place") or "Unknown location"),
                            "time": props.get("time") or 0,
                            "tsunami": int(props.get("tsunami") or 0),
                        },
                    })
                except (TypeError, ValueError):
                    continue
            fc = {
                "type": "FeatureCollection",
                "features": feats,
                "meta": {"source": "USGS all_day", "count": len(feats), "updated": datetime.now().strftime("%H:%M:%S")},
            }
            with self._lock:
                self.earthquakes = fc
                self._eq_ts = time.time()
            try:
                cache_path.write_text(json.dumps(fc), encoding="utf-8")
            except Exception:
                pass
        except Exception:
            with self._lock:
                empty = not self.earthquakes.get("features")
            if empty and cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    with self._lock:
                        self.earthquakes = cached
                except Exception:
                    pass
        finally:
            with self._lock:
                self._fetching_eq = False

    def ensure_flights(self, force: bool = False) -> dict:
        now = time.time()
        with self._lock:
            fl = self.flights
            need = force or (now - self._flt_ts >= 25.0) or not fl.get("features")
            if need and not self._fetching_flt:
                self._fetching_flt = True
                threading.Thread(target=self._fetch_flights, args=(force,), daemon=True).start()
            return fl

    def _fetch_flights(self, force: bool = False) -> None:
        cache_path = _CACHE_DIR / "flights.json"
        feats: list[dict] = []
        source_label = "none"
        is_regional = False
        now = time.time()

        try:
            with self._lock:
                backoff_until = self._flt_backoff_until
                view_lat = self._view_lat
                view_lon = self._view_lon

            if now >= backoff_until or force:
                try:
                    raw = _http_get("https://opensky-network.org/api/states/all", timeout=14.0)
                    data = json.loads(raw.decode("utf-8"))
                    for st in data.get("states") or []:
                        if not isinstance(st, list) or len(st) < 11:
                            continue
                        try:
                            lon_v, lat_v = st[5], st[6]
                            if lon_v is None or lat_v is None:
                                continue
                            lon_f = float(lon_v)
                            lat_f = float(lat_v)
                            if not (math.isfinite(lon_f) and math.isfinite(lat_f)):
                                continue
                            alt_m = float(st[7]) if st[7] is not None else 0.0
                            vel_ms = float(st[9]) if st[9] is not None else 0.0
                            hdg = float(st[10]) if st[10] is not None else 0.0
                            vrate = float(st[11]) if len(st) > 11 and st[11] is not None else 0.0
                            callsign = str(st[1] or st[0] or "N/A").strip()
                            feats.append({
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [round(lon_f, 4), round(lat_f, 4)]},
                                "properties": {
                                    "icao": str(st[0] or ""),
                                    "callsign": callsign,
                                    "country": str(st[2] or ""),
                                    "alt_ft": int(alt_m * 3.28084),
                                    "spd_kts": int(vel_ms * 1.94384),
                                    "heading": round(hdg, 1),
                                    "vrate_fpm": int(vrate * 196.85),
                                    "on_ground": bool(st[8]),
                                    "type": "",
                                },
                            })
                        except (TypeError, ValueError):
                            continue
                    if feats:
                        source_label = "OpenSky Global"
                except urllib.error.HTTPError as he:
                    if he.code == 429:
                        with self._lock:
                            self._flt_backoff_until = time.time() + 180.0
                except Exception:
                    pass

            if not feats:
                lat_c = round(view_lat, 3)
                lon_c = round(view_lon, 3)
                for label, url in (
                    ("adsb.lol", f"https://api.adsb.lol/v2/point/{lat_c}/{lon_c}/250"),
                    ("adsb.fi", f"https://opendata.adsb.fi/api/v3/lat/{lat_c}/{lon_c}/dist/250"),
                ):
                    try:
                        raw = _http_get(url, timeout=10.0)
                        data = json.loads(raw.decode("utf-8"))
                        for ac in data.get("ac") or []:
                            try:
                                lat_v, lon_v = ac.get("lat"), ac.get("lon")
                                if lat_v is None or lon_v is None:
                                    continue
                                lat_f = float(lat_v)
                                lon_f = float(lon_v)
                                if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
                                    continue
                                alt_raw = ac.get("alt_baro", 0)
                                on_g = (alt_raw == "ground")
                                alt_ft = 0 if on_g else int(float(alt_raw or 0))
                                gs_val = float(ac.get("gs") or 0)
                                trk_val = float(ac.get("track") or 0)
                                baro_val = float(ac.get("baro_rate") or 0)
                                callsign = str(ac.get("flight") or ac.get("r") or ac.get("hex") or "N/A").strip()
                                feats.append({
                                    "type": "Feature",
                                    "geometry": {"type": "Point", "coordinates": [round(lon_f, 4), round(lat_f, 4)]},
                                    "properties": {
                                        "icao": str(ac.get("hex") or ""),
                                        "callsign": callsign,
                                        "country": str(ac.get("r") or ""),
                                        "alt_ft": alt_ft,
                                        "spd_kts": int(gs_val),
                                        "heading": round(trk_val, 1),
                                        "vrate_fpm": int(baro_val),
                                        "on_ground": on_g,
                                        "type": str(ac.get("t") or ""),
                                    },
                                })
                            except (TypeError, ValueError):
                                continue
                        if feats:
                            source_label = f"{label} ({lat_c:.1f},{lon_c:.1f})"
                            is_regional = True
                            with self._lock:
                                self._last_flt_center = (lat_c, lon_c)
                            break
                    except Exception:
                        continue

            if feats:
                fc = {
                    "type": "FeatureCollection",
                    "features": feats,
                    "meta": {"source": source_label, "count": len(feats), "updated": datetime.now().strftime("%H:%M:%S")},
                }
                with self._lock:
                    self.flights = fc
                    self._flt_ts = time.time()
                    self._flt_is_regional = is_regional
                try:
                    cache_path.write_text(json.dumps(fc), encoding="utf-8")
                except Exception:
                    pass
            else:
                with self._lock:
                    empty = not self.flights.get("features")
                if empty and cache_path.exists():
                    try:
                        cached = json.loads(cache_path.read_text(encoding="utf-8"))
                        with self._lock:
                            self.flights = cached
                    except Exception:
                        pass
        finally:
            with self._lock:
                self._fetching_flt = False

    def ensure_satellites(self, force: bool = False) -> list[dict]:
        now = time.time()
        with self._lock:
            sats = self.sat_tles
            need = force or (now - self._sat_ts >= 21600.0) or len(sats) <= 20
            if need and not self._fetching_sat:
                self._fetching_sat = True
                threading.Thread(target=self._fetch_satellites, daemon=True).start()
            return sats

    def _fetch_satellites(self) -> None:
        tle_path = _CACHE_DIR / "celestrak_active.tle"
        urls = [
            ("CelesTrak active", "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"),
            ("CelesTrak visual", "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle"),
        ]
        try:
            for label, url in urls:
                try:
                    raw = _http_get(url, timeout=20.0).decode("utf-8", errors="replace")
                    parsed = _parse_tles(raw)
                    if len(parsed) > 10:
                        try:
                            tle_path.write_text(raw, encoding="utf-8")
                        except Exception:
                            pass
                        with self._lock:
                            self.sat_tles = parsed
                            self._sat_ts = time.time()
                            self.sat_meta = {
                                "count": len(parsed),
                                "source": label,
                                "updated": datetime.now().strftime("%H:%M:%S"),
                            }
                        break
                except Exception:
                    continue
        finally:
            with self._lock:
                self._fetching_sat = False

    def get_satellite_tles_payload(self) -> dict:
        sats = self.ensure_satellites()
        with self._lock:
            meta = dict(self.sat_meta)
        compact = [[s["id"], s["name"], s["cat"], s["l1"], s["l2"], s["inc_deg"]] for s in sats]
        return {"meta": meta, "tles": compact}

    def get_satellite_positions(self, group: str = "all") -> dict:
        sats = self.ensure_satellites()
        with self._lock:
            meta = dict(self.sat_meta)
        jd_now = _datetime_to_jd(datetime.now(timezone.utc))
        feats = []
        for s in sats:
            if group != "all" and s["cat"] != group:
                continue
            pos = _propagate_sat_geodetic(s, jd_now)
            if pos is None:
                continue
            lon, lat, alt_km, vel_kms = pos
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": s["id"],
                    "name": s["name"],
                    "cat": s["cat"],
                    "alt_km": alt_km,
                    "vel_kms": vel_kms,
                    "inc_deg": s["inc_deg"],
                },
            })
        return {
            "type": "FeatureCollection",
            "features": feats,
            "meta": {**meta, "visible": len(feats)},
        }

    def get_satellite_orbit(self, sat_id: str) -> dict:
        with self._lock:
            sats = self.sat_tles
        target = None
        for s in sats:
            if s["id"] == sat_id or s["name"].upper() == sat_id.upper():
                target = s
                break
        if target is None:
            return {"points": []}
        jd_now = _datetime_to_jd(datetime.now(timezone.utc))
        period_days = (2.0 * math.pi / target["n"]) / 86400.0
        steps = 120
        pts: list[list[float]] = []
        for k in range(steps + 1):
            frac = (k / steps) * min(period_days * 1.05, 0.25)
            pos = _propagate_sat_geodetic(target, jd_now + frac)
            if pos is not None:
                pts.append([pos[0], pos[1], pos[2]])
        return {"id": target["id"], "name": target["name"], "points": pts}

    def geocode(self, query: str) -> dict | None:
        q = query.strip()
        if not q:
            return None

        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*$", q)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return {"lat": lat, "lon": lon, "zoom": 13.0, "label": f"{lat:.4f}, {lon:.4f}", "kind": "coords"}

        ql = q.casefold()

        if ql in _QUICK_PLACES:
            lat, lon, z = _QUICK_PLACES[ql]
            return {"lat": lat, "lon": lon, "zoom": z, "label": q.title(), "kind": "city"}

        jd_now = _datetime_to_jd(datetime.now(timezone.utc))
        with self._lock:
            sats = self.sat_tles
            flights = self.flights.get("features", [])
        for s in sats:
            if ql == s["id"].casefold() or ql in s["name"].casefold():
                pos = _propagate_sat_geodetic(s, jd_now)
                if pos:
                    return {
                        "lat": pos[1], "lon": pos[0], "alt_km": pos[2], "zoom": 4.2,
                        "label": f"SAT {s['name']} ({pos[2]:.0f} km)", "kind": "sat", "sat_id": s["id"],
                    }

        for f in flights:
            pr = f.get("properties") or {}
            if ql == str(pr.get("callsign", "")).casefold() or ql == str(pr.get("icao", "")).casefold():
                lon, lat = f["geometry"]["coordinates"]
                return {
                    "lat": lat, "lon": lon, "zoom": 8.0,
                    "label": f"FLT {pr.get('callsign')} ({pr.get('alt_ft')} ft)", "kind": "flight",
                }

        with self._lock:
            cached = self._geocode_cache.get(ql)
        if cached is not None:
            return cached
        try:
            url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({
                "q": q, "format": "json", "limit": 1,
            })
            raw = _http_get(url, timeout=8.0)
            arr = json.loads(raw.decode("utf-8"))
            if arr:
                lat = float(arr[0]["lat"])
                lon = float(arr[0]["lon"])
                disp = str(arr[0].get("display_name") or q)
                place_type = str(arr[0].get("type") or "")
                z_guess = 5.2 if place_type in ("country", "administrative") else 14.5
                res = {"lat": lat, "lon": lon, "zoom": z_guess, "label": disp.split(",")[0], "full_name": disp, "kind": "street"}
                with self._lock:
                    self._geocode_cache[ql] = res
                return res
        except Exception:
            pass
        return None

    def fetch_tile(self, layer_tag: str, z: int, x: int, y: int) -> bytes | None:
        key = f"{layer_tag}/{z}/{x}/{y}"
        with self._tile_lock:
            if key in self._tile_cache:
                return self._tile_cache[key]
        for url in _tile_urls_for_layer(layer_tag, z, x, y):
            try:
                data = _http_get(url, timeout=6.0)
                if data:
                    with self._tile_lock:
                        if len(self._tile_cache) > 512:
                            self._tile_cache.clear()
                        self._tile_cache[key] = data
                    return data
            except Exception:
                continue
        return None

    def refresh_all(self) -> None:
        self.ensure_earthquakes(force=True)
        self.ensure_flights(force=True)
        self.ensure_satellites(force=True)


def _get_hub() -> _GlobeDataHub:
    global _HUB
    with _LOCK:
        if _HUB is None:
            _HUB = _GlobeDataHub()
        return _HUB


# -- Background Web Worker JS for Off-Thread SGP4 Satellite Propagation -------

_SAT_WORKER_JS = r"""
importScripts("/assets/satellite.min.js");
var SAT_RECS = [];

self.onmessage = function (ev) {
  var msg = ev.data || {};
  if (msg.type === "catalog") {
    var arr = msg.tles || [];
    var recs = [];
    for (var i = 0; i < arr.length; i++) {
      var item = arr[i];
      try {
        var sr = satellite.twoline2satrec(item[3], item[4]);
        recs.push({ id: item[0], name: item[1], cat: item[2], inc: item[5], sr: sr });
      } catch (e) {}
    }
    SAT_RECS = recs;
    propagateBatch(msg.group || "stations");
  } else if (msg.type === "step") {
    propagateBatch(msg.group || "stations");
  }
};

function propagateBatch(group) {
  if (!SAT_RECS.length || typeof satellite === "undefined") return;
  var now = new Date();
  var gmst = satellite.gstime(now);
  var out = new Float32Array(SAT_RECS.length * 4);
  var meta = [];
  var n = 0;
  for (var i = 0; i < SAT_RECS.length; i++) {
    var s = SAT_RECS[i];
    if (group !== "all" && s.cat !== group) continue;
    var pv = satellite.propagate(s.sr, now);
    if (!pv || !pv.position) continue;
    var gd = satellite.eciToGeodetic(pv.position, gmst);
    var lon = satellite.degreesLong(gd.longitude);
    var lat = satellite.degreesLat(gd.latitude);
    var alt = gd.height;
    if (!isFinite(lon) || !isFinite(lat) || !isFinite(alt) || alt < 80) continue;
    var base = n * 4;
    out[base] = lon;
    out[base + 1] = lat;
    out[base + 2] = alt;
    out[base + 3] = (s.cat === "stations" ? 1 : (s.cat === "nav" ? 2 : (s.cat === "weather" ? 3 : 0)));
    meta.push([s.id, s.name, s.cat, s.inc]);
    n++;
  }
  var sliced = out.slice(0, n * 4);
  self.postMessage({ type: "positions", count: n, buf: sliced.buffer, meta: meta }, [sliced.buffer]);
}
"""


# -- Interactive CesiumJS + Google Earth 3D WebGL Globe HTML ------------------

_GLOBE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>MARK LVII // EARTH GLOBE</title>
<script>window.CESIUM_BASE_URL = "/cesium/";</script>
<link rel="stylesheet" href="/cesium/Widgets/widgets.css" />
<style>
  * { box-sizing: border-box; user-select: none; }
  html, body, #cesiumContainer {
    margin: 0; padding: 0; width: 100%; height: 100%;
    background: #01050b; color: #c7eeff;
    font-family: "Consolas", "Segoe UI", monospace;
    overflow: hidden;
  }
  .cesium-widget-credits { display: none !important; }
  .hud-panel {
    position: absolute; z-index: 20; pointer-events: none;
    background: rgba(3, 12, 24, 0.72);
    border: 1px solid rgba(56, 217, 255, 0.38);
    border-radius: 3px;
    padding: 6px 10px;
    backdrop-filter: blur(5px);
    box-shadow: 0 0 14px rgba(0, 180, 255, 0.16);
  }
  #hud-tl { top: 8px; left: 8px; min-width: 195px; }
  #hud-clock { font-size: 16px; font-weight: bold; color: #eefaff; letter-spacing: 0.6px; }
  #hud-coords { font-size: 10.5px; color: #7ce2ff; margin-top: 2px; line-height: 1.35; }
  #hud-bl {
    bottom: 8px; left: 8px;
    display: flex; gap: 10px; align-items: center; font-size: 10.5px;
  }
  #hud-br { bottom: 8px; right: 8px; font-size: 10.5px; color: #90e0ff; text-align: right; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
  #popup-card {
    position: absolute; z-index: 30; display: none;
    min-width: 210px; max-width: 280px;
    background: rgba(3, 14, 28, 0.92);
    border: 1px solid #38d9ff;
    border-radius: 3px;
    padding: 8px 10px; font-size: 11px; color: #dff7ff;
    box-shadow: 0 0 18px rgba(0, 212, 255, 0.30);
    pointer-events: auto;
  }
  #popup-close {
    float: right; cursor: pointer; color: #38d9ff; font-weight: bold;
    margin-left: 8px; padding: 0 3px;
  }
  #err-banner {
    position: absolute; inset: 0; display: none; align-items: center;
    justify-content: center; text-align: center; padding: 24px;
    background: #020810; color: #ff9a76; font-size: 13px; z-index: 40;
  }
</style>
</head>
<body>
<div id="cesiumContainer"></div>

<div id="hud-tl" class="hud-panel">
  <div id="hud-clock">--:-- UTC</div>
  <div id="hud-coords">LAT 20.000, LON 12.000<br>ALT 18,500 km</div>
</div>

<div id="hud-bl" class="hud-panel">
  <span><i class="dot" style="background:#ff9500"></i>QUAKES</span>
  <span><i class="dot" style="background:#38d9ff"></i>FLIGHTS</span>
  <span><i class="dot" style="background:#ffe65c"></i>STATIONS/SATS</span>
  <span><i class="dot" style="background:#ffd369"></i>CITY LIGHTS</span>
</div>

<div id="hud-br" class="hud-panel">MODE: AUTO | SYNCING...</div>

<div id="popup-card">
  <span id="popup-close" onclick="document.getElementById('popup-card').style.display='none'">x</span>
  <div id="popup-body"></div>
</div>

<div id="err-banner"></div>

<script src="/assets/satellite.min.js"></script>
<script src="/cesium/Cesium.js"></script>
<script>
(function () {
  window.__MARK_QUEUE__ = [];
  window.__MARK_CALL__ = function (method, args) {
    if (window.__MARK_GLOBE__ && typeof window.__MARK_GLOBE__[method] === "function") {
      window.__MARK_GLOBE__[method].apply(null, args || []);
    } else {
      window.__MARK_QUEUE__.push([method, args || []]);
    }
  };

  function showErr(msg) {
    var b = document.getElementById("err-banner");
    b.style.display = "flex";
    b.innerText = msg;
  }

  if (typeof Cesium === "undefined") {
    showErr("CesiumJS failed to load. Check internet connection or click '2D/QT' fallback.");
    return;
  }

  var STATE = {
    active: true,
    mapMode: "auto",
    satGroup: "stations",
    layers: {
      earthquakes: true,
      flights: true,
      satellites: true,
      city_lights: true,
      streets: true
    },
    counts: { quakes: 0, flights: 0, sats: 0 },
    allFlights: [],
    satWorker: null,
    satRecordsCount: 0,
    satMeta: [],
    satFloatData: null,
    selectedSatId: null,
    highlightedCountry: null,
    hoveredCountry: null,
    lastCamAlt: 18500000
  };

  var viewer;
  try {
    viewer = new Cesium.Viewer("cesiumContainer", {
      baseLayer: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      vrButton: false,
      infoBox: false,
      selectionIndicator: false,
      shadows: false,
      shouldAnimate: true,
      requestRenderMode: false,
      creditContainer: document.createElement("div")
    });
  } catch (err) {
    showErr("WebGL 3D Globe failed to initialize (" + (err && err.message ? err.message : err) + "). Click '2D/QT' mode.");
    return;
  }

  var scene = viewer.scene;
  var globe = scene.globe;
  globe.baseColor = Cesium.Color.fromCssColorString("#020a18");
  globe.enableLighting = true;
  globe.dynamicAtmosphereLighting = true;
  globe.dynamicAtmosphereLightingFromSun = true;
  globe.showGroundAtmosphere = true;
  globe.atmosphereLightIntensity = 18.0;
  globe.tileCacheSize = 256;
  globe.preloadSiblings = true;

  if (scene.skyAtmosphere) {
    scene.skyAtmosphere.hueShift = -0.02;
    scene.skyAtmosphere.saturationShift = 0.22;
    scene.skyAtmosphere.brightnessShift = 0.08;
  }
  scene.fog.enabled = true;
  scene.fog.density = 0.00012;
  scene.postProcessStages.fxaa.enabled = true;
  viewer.resolutionScale = Math.min(window.devicePixelRatio || 1, 1.5);
  viewer.clock.currentTime = Cesium.JulianDate.now();
  viewer.clock.shouldAnimate = true;

  function makeXYZ(url, maxLvl, subs) {
    var opts = { url: url, maximumLevel: maxLvl || 20, enablePickFeatures: false };
    if (subs) opts.subdomains = subs;
    return new Cesium.UrlTemplateImageryProvider(opts);
  }

  var esriLayer = viewer.imageryLayers.addImageryProvider(
    makeXYZ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 18)
  );
  var googleSatLayer = viewer.imageryLayers.addImageryProvider(
    makeXYZ("https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 20, ["0", "1", "2", "3"])
  );
  var googleStreetLayer = viewer.imageryLayers.addImageryProvider(
    makeXYZ("https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", 20, ["0", "1", "2", "3"])
  );
  var nightLightsLayer = viewer.imageryLayers.addImageryProvider(
    makeXYZ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_Black_Marble/default/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png", 8)
  );
  var googleLabelsLayer = viewer.imageryLayers.addImageryProvider(
    makeXYZ("https://mt{s}.google.com/vt/lyrs=h&x={x}&y={y}&z={z}", 20, ["0", "1", "2", "3"])
  );

  viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(12.0, 20.0, 18500000.0)
  });

  function makeBeaconSprite(innerHex, outerRgba, size) {
    var c = document.createElement("canvas");
    c.width = size; c.height = size;
    var ctx = c.getContext("2d");
    var cx = size / 2;
    var g = ctx.createRadialGradient(cx, cx, 1, cx, cx, cx - 1);
    g.addColorStop(0.0, "#ffffff");
    g.addColorStop(0.28, innerHex);
    g.addColorStop(0.65, outerRgba);
    g.addColorStop(1.0, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cx, cx - 1, 0, Math.PI * 2);
    ctx.fill();
    return c;
  }

  function makeRingSprite(strokeHex, size) {
    var c = document.createElement("canvas");
    c.width = size; c.height = size;
    var ctx = c.getContext("2d");
    var cx = size / 2;
    ctx.strokeStyle = strokeHex;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.arc(cx, cx, cx * 0.38, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = strokeHex;
    ctx.beginPath();
    ctx.arc(cx, cx, 2.8, 0, Math.PI * 2);
    ctx.fill();
    return c;
  }

  function makePlaneSprite(colorHex) {
    var c = document.createElement("canvas");
    c.width = 28; c.height = 28;
    var ctx = c.getContext("2d");
    ctx.fillStyle = colorHex;
    ctx.strokeStyle = "#010913";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(14, 2);
    ctx.lineTo(16.5, 9.5);
    ctx.lineTo(25, 14.5);
    ctx.lineTo(25, 17);
    ctx.lineTo(16.5, 14.5);
    ctx.lineTo(16.5, 21.5);
    ctx.lineTo(20, 24.5);
    ctx.lineTo(20, 26);
    ctx.lineTo(14, 24);
    ctx.lineTo(8, 26);
    ctx.lineTo(8, 24.5);
    ctx.lineTo(11.5, 21.5);
    ctx.lineTo(11.5, 14.5);
    ctx.lineTo(3, 17);
    ctx.lineTo(3, 14.5);
    ctx.lineTo(11.5, 9.5);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    return c;
  }

  var SPRITES = {
    quakeLow: makeBeaconSprite("#ffd60a", "rgba(255, 214, 10, 0.35)", 28),
    quakeMed: makeBeaconSprite("#ff9500", "rgba(255, 149, 0, 0.45)", 34),
    quakeHigh: makeBeaconSprite("#ff3b30", "rgba(255, 59, 48, 0.55)", 42),
    station: makeRingSprite("#ffe65c", 32),
    planeHigh: makePlaneSprite("#fff3a6"),
    planeCruise: makePlaneSprite("#38d9ff"),
    planeLow: makePlaneSprite("#00ffa6"),
    planeGround: makePlaneSprite("#8a9aa8")
  };

  var quakeBillboards = scene.primitives.add(new Cesium.BillboardCollection());
  var quakeLabels = scene.primitives.add(new Cesium.LabelCollection());
  var flightBillboards = scene.primitives.add(new Cesium.BillboardCollection());
  var stationBillboards = scene.primitives.add(new Cesium.BillboardCollection());
  var stationLabels = scene.primitives.add(new Cesium.LabelCollection());
  var satPoints = scene.primitives.add(new Cesium.PointPrimitiveCollection());
  var orbitPolylines = scene.primitives.add(new Cesium.PolylineCollection());
  var satPointPool = [];

  var countryNormalMat = Cesium.Color.fromCssColorString("#00d4ff").withAlpha(0.01);
  var countryNormalOutline = Cesium.Color.fromCssColorString("#38d9ff").withAlpha(0.20);
  var countryHoverMat = Cesium.Color.fromCssColorString("#00d4ff").withAlpha(0.18);
  var countrySelectMat = Cesium.Color.fromCssColorString("#00d4ff").withAlpha(0.28);
  var countryBrightOutline = Cesium.Color.fromCssColorString("#8ceeff");

  Cesium.GeoJsonDataSource.load("/api/countries", {
    stroke: countryNormalOutline,
    fill: countryNormalMat,
    strokeWidth: 1.0,
    clampToGround: false
  }).then(function (ds) {
    viewer.dataSources.add(ds);
    var ents = ds.entities.values;
    for (var i = 0; i < ents.length; i++) {
      var e = ents[i];
      if (e.polygon) {
        e.polygon.material = countryNormalMat;
        e.polygon.outlineColor = countryNormalOutline;
      }
    }
  }).catch(function () {});

  function applyVisualState() {
    var m = STATE.mapMode;
    var L = STATE.layers;
    var camH = STATE.lastCamAlt;
    var isClose = camH < 320000;
    var isStreetScale = camH < 120000;

    if (m === "auto") {
      esriLayer.show = true;
      googleSatLayer.show = true;
      esriLayer.dayAlpha = 1.0; esriLayer.nightAlpha = 0.14;
      googleSatLayer.dayAlpha = 1.0; googleSatLayer.nightAlpha = 0.14;
      googleSatLayer.alpha = 1.0;

      globe.enableLighting = !isClose;
      nightLightsLayer.show = L.city_lights && !isClose;
      nightLightsLayer.dayAlpha = 0.0;
      nightLightsLayer.nightAlpha = 1.0;
      nightLightsLayer.brightness = 1.45;
      nightLightsLayer.contrast = 1.15;

      googleStreetLayer.show = false;
      googleLabelsLayer.show = L.streets && camH < 4800000;
      googleLabelsLayer.alpha = isStreetScale ? 0.95 : 0.78;
    } else if (m === "city_lights") {
      esriLayer.show = true;
      googleSatLayer.show = true;
      esriLayer.dayAlpha = 0.35; esriLayer.nightAlpha = 0.10;
      googleSatLayer.dayAlpha = 0.35; googleSatLayer.nightAlpha = 0.10;
      globe.enableLighting = true;
      nightLightsLayer.show = L.city_lights;
      nightLightsLayer.dayAlpha = 0.85;
      nightLightsLayer.nightAlpha = 1.0;
      nightLightsLayer.brightness = 1.6;
      googleStreetLayer.show = false;
      googleLabelsLayer.show = L.streets;
    } else if (m === "street") {
      esriLayer.show = false;
      googleSatLayer.show = false;
      nightLightsLayer.show = false;
      globe.enableLighting = false;
      googleStreetLayer.show = true;
      googleLabelsLayer.show = false;
    } else if (m === "satellite") {
      esriLayer.show = true;
      googleSatLayer.show = true;
      esriLayer.dayAlpha = 1.0; esriLayer.nightAlpha = 1.0;
      googleSatLayer.dayAlpha = 1.0; googleSatLayer.nightAlpha = 1.0;
      globe.enableLighting = false;
      nightLightsLayer.show = false;
      googleStreetLayer.show = false;
      googleLabelsLayer.show = L.streets && camH < 6000000;
    } else {
      esriLayer.show = true;
      googleSatLayer.show = true;
      esriLayer.dayAlpha = 0.25; esriLayer.nightAlpha = 0.12;
      googleSatLayer.dayAlpha = 0.25; googleSatLayer.nightAlpha = 0.12;
      globe.enableLighting = false;
      nightLightsLayer.show = L.city_lights;
      nightLightsLayer.dayAlpha = 0.65;
      nightLightsLayer.nightAlpha = 0.85;
      googleStreetLayer.show = false;
      googleLabelsLayer.show = L.streets;
    }

    quakeBillboards.show = !!L.earthquakes;
    quakeLabels.show = !!L.earthquakes;
    flightBillboards.show = !!L.flights;
    stationBillboards.show = !!L.satellites;
    stationLabels.show = !!L.satellites;
    satPoints.show = !!L.satellites && STATE.satGroup !== "stations";
    orbitPolylines.show = !!L.satellites;
    updateTelemetryBanner();
  }

  function updateTelemetryBanner(extra) {
    var el = document.getElementById("hud-br");
    el.innerText =
      "MODE: " + STATE.mapMode.toUpperCase()
      + "  |  SATS: " + STATE.counts.sats
      + "  |  FLIGHTS: " + STATE.counts.flights
      + "  |  QUAKES: " + STATE.counts.quakes
      + (extra ? "  |  " + extra : "");
  }

  function updateTopLeftHud() {
    var d = new Date();
    var hh = String(d.getUTCHours()).padStart(2, "0");
    var mm = String(d.getUTCMinutes()).padStart(2, "0");
    document.getElementById("hud-clock").innerText = hh + ":" + mm + " UTC";

    var cpos = viewer.camera.positionCartographic;
    if (cpos) {
      var lat = Cesium.Math.toDegrees(cpos.latitude);
      var lon = Cesium.Math.toDegrees(cpos.longitude);
      var altM = Math.max(0, cpos.height);
      STATE.lastCamAlt = altM;
      var altStr = altM >= 1000 ? (Math.round(altM / 1000).toLocaleString() + " km") : (Math.round(altM) + " m");
      document.getElementById("hud-coords").innerHTML =
        "LAT " + lat.toFixed(3) + ", LON " + lon.toFixed(3) + "<br>ALT " + altStr;
    }
  }

  setInterval(updateTopLeftHud, 1000);
  var lastViewSync = 0;
  var lastCloseBucket = false;
  var lastLblBucket = false;
  viewer.camera.percentageChanged = 0.02;
  viewer.camera.changed.addEventListener(function () {
    updateTopLeftHud();
    var isClose = STATE.lastCamAlt < 320000;
    var isLbl = STATE.lastCamAlt < 4800000;
    if (isClose !== lastCloseBucket || isLbl !== lastLblBucket) {
      lastCloseBucket = isClose;
      lastLblBucket = isLbl;
      applyVisualState();
    }
    var now = Date.now();
    if (now - lastViewSync > 800) {
      lastViewSync = now;
      var cpos = viewer.camera.positionCartographic;
      if (cpos) {
        var lat = Cesium.Math.toDegrees(cpos.latitude);
        var lon = Cesium.Math.toDegrees(cpos.longitude);
        fetch("/api/view?lat=" + lat.toFixed(3) + "&lon=" + lon.toFixed(3)).catch(function () {});
      }
      renderVisibleFlights();
    }
  });

  function renderEarthquakes(fc) {
    quakeBillboards.removeAll();
    quakeLabels.removeAll();
    var feats = (fc && fc.features) || [];
    for (var i = 0; i < feats.length; i++) {
      var f = feats[i];
      var coords = f.geometry && f.geometry.coordinates;
      if (!coords || coords.length < 2) continue;
      var lon = coords[0], lat = coords[1];
      var p = f.properties || {};
      var mag = Number(p.mag || 0);
      var pos = Cesium.Cartesian3.fromDegrees(lon, lat, 1500);
      var img = mag >= 5.0 ? SPRITES.quakeHigh : (mag >= 3.0 ? SPRITES.quakeMed : SPRITES.quakeLow);
      var sc = mag >= 5.0 ? 0.95 : (mag >= 3.0 ? 0.72 : 0.45);
      quakeBillboards.add({
        position: pos,
        image: img,
        scale: sc,
        id: { kind: "quake", lon: lon, lat: lat, altM: 1500, props: p }
      });
      if (mag >= 3.0) {
        quakeLabels.add({
          position: pos,
          text: mag.toFixed(1),
          font: "bold 11px Consolas, monospace",
          fillColor: Cesium.Color.fromCssColorString("#ffe29a"),
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(12, -2)
        });
      }
    }
    STATE.counts.quakes = feats.length;
    updateTelemetryBanner();
  }

  function renderVisibleFlights() {
    flightBillboards.removeAll();
    var all = STATE.allFlights;
    if (!all.length) {
      STATE.counts.flights = 0;
      updateTelemetryBanner();
      return;
    }
    var cpos = viewer.camera.positionCartographic;
    var cLat = cpos ? Cesium.Math.toDegrees(cpos.latitude) : 20.0;
    var cLon = cpos ? Cesium.Math.toDegrees(cpos.longitude) : 12.0;
    var subset = all;
    if (all.length > 800) {
      var ranked = [];
      for (var i = 0; i < all.length; i++) {
        var f = all[i];
        var dLat = f.lat - cLat;
        var dLon = f.lon - cLon;
        if (dLon > 180) dLon -= 360;
        if (dLon < -180) dLon += 360;
        ranked.push({ f: f, d2: dLat * dLat + dLon * dLon });
      }
      ranked.sort(function (a, b) { return a.d2 - b.d2; });
      subset = ranked.slice(0, 800).map(function (x) { return x.f; });
    }
    for (var j = 0; j < subset.length; j++) {
      var fl = subset[j];
      var p = fl.props;
      var altM = p.on_ground ? 60 : Math.max(300, (p.alt_ft || 0) * 0.3048);
      var img = p.on_ground ? SPRITES.planeGround : (p.alt_ft > 32000 ? SPRITES.planeHigh : (p.alt_ft > 15000 ? SPRITES.planeCruise : SPRITES.planeLow));
      flightBillboards.add({
        position: Cesium.Cartesian3.fromDegrees(fl.lon, fl.lat, altM),
        image: img,
        scale: STATE.lastCamAlt < 600000 ? 0.72 : 0.50,
        rotation: -Cesium.Math.toRadians(p.heading || 0),
        alignedAxis: Cesium.Cartesian3.UNIT_Z,
        id: { kind: "flight", lon: fl.lon, lat: fl.lat, altM: altM, props: p }
      });
    }
    STATE.counts.flights = all.length;
    updateTelemetryBanner();
  }

  function applySatelliteBuffer(count, buffer, meta) {
    var arr = new Float32Array(buffer);
    STATE.satFloatData = arr;
    STATE.satMeta = meta;
    stationBillboards.removeAll();
    stationLabels.removeAll();

    var showPoints = (STATE.satGroup !== "stations");
    var ptIdx = 0;
    var colorDefault = Cesium.Color.fromCssColorString("#5eff9c").withAlpha(0.75);
    var colorNav = Cesium.Color.fromCssColorString("#4dd0e1").withAlpha(0.78);
    var colorWeather = Cesium.Color.fromCssColorString("#c792ea").withAlpha(0.78);

    for (var i = 0; i < count; i++) {
      var base = i * 4;
      var lon = arr[base];
      var lat = arr[base + 1];
      var altKm = arr[base + 2];
      var catCode = arr[base + 3];
      var m = meta[i];
      var pos = Cesium.Cartesian3.fromDegrees(lon, lat, Math.max(120, altKm) * 1000.0);

      if (catCode === 1) {
        stationBillboards.add({
          position: pos,
          image: SPRITES.station,
          scale: 0.78,
          id: { kind: "sat", lon: lon, lat: lat, altM: altKm * 1000.0, props: { id: m[0], name: m[1], cat: m[2], inc_deg: m[3], alt_km: Math.round(altKm) } }
        });
        stationLabels.add({
          position: pos,
          text: m[1],
          font: "bold 10px Consolas, monospace",
          fillColor: Cesium.Color.fromCssColorString("#fff59d"),
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(11, 0),
          distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 28000000)
        });
      } else if (showPoints) {
        var col = catCode === 2 ? colorNav : (catCode === 3 ? colorWeather : colorDefault);
        var pt;
        if (ptIdx < satPointPool.length) {
          pt = satPointPool[ptIdx];
          pt.show = true;
          pt.position = pos;
          pt.color = col;
          pt.pixelSize = 2.6;
        } else {
          pt = satPoints.add({ position: pos, color: col, pixelSize: 2.6 });
          satPointPool.push(pt);
        }
        pt.id = { kind: "sat", lon: lon, lat: lat, altM: altKm * 1000.0, props: { id: m[0], name: m[1], cat: m[2], inc_deg: m[3], alt_km: Math.round(altKm) } };
        ptIdx++;
      }
    }
    for (var k = ptIdx; k < satPointPool.length; k++) {
      satPointPool[k].show = false;
    }
    satPoints.show = !!STATE.layers.satellites && showPoints;
    STATE.counts.sats = count;
    updateTelemetryBanner();
  }

  function drawSatelliteOrbit(satId) {
    orbitPolylines.removeAll();
    fetch("/api/satellites/orbit?id=" + encodeURIComponent(satId))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var pts = d.points || [];
        if (pts.length < 2) return;
        var flat = [];
        for (var i = 0; i < pts.length; i++) {
          flat.push(pts[i][0], pts[i][1], Math.max(120, pts[i][2]) * 1000.0);
        }
        orbitPolylines.add({
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(flat),
          width: 1.8,
          material: Cesium.Material.fromType("Color", {
            color: Cesium.Color.fromCssColorString("#7dffb3").withAlpha(0.85)
          })
        });
      }).catch(function () {});
  }

  function postSelection(payload) {
    fetch("/api/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).catch(function () {});
  }

  function showPopupAt(lon, lat, altM, html, selectObj) {
    var card = document.getElementById("popup-card");
    document.getElementById("popup-body").innerHTML = html;
    var xy = scene.cartesianToCanvasCoordinates(Cesium.Cartesian3.fromDegrees(lon, lat, altM || 0));
    if (xy) {
      var left = Math.max(12, Math.min(window.innerWidth - 290, xy.x + 12));
      var top = Math.max(12, Math.min(window.innerHeight - 130, xy.y - 24));
      card.style.left = left + "px";
      card.style.top = top + "px";
    } else {
      card.style.left = "14px";
      card.style.top = "95px";
    }
    card.style.display = "block";
    if (selectObj) postSelection(selectObj);
  }

  function setCountryStyle(ent, mat, outCol) {
    if (!ent || !ent.polygon) return;
    ent.polygon.material = mat;
    ent.polygon.outlineColor = outCol;
  }

  var handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);
  var lastMovePick = 0;
  handler.setInputAction(function (mv) {
    var now = Date.now();
    if (now - lastMovePick < 45) return;
    lastMovePick = now;
    var picked = scene.pick(mv.endPosition);
    var nextHover = (picked && picked.id instanceof Cesium.Entity && picked.id.polygon) ? picked.id : null;
    if (STATE.hoveredCountry && STATE.hoveredCountry !== nextHover && STATE.hoveredCountry !== STATE.highlightedCountry) {
      setCountryStyle(STATE.hoveredCountry, countryNormalMat, countryNormalOutline);
    }
    if (nextHover && nextHover !== STATE.highlightedCountry) {
      setCountryStyle(nextHover, countryHoverMat, countryBrightOutline);
    }
    STATE.hoveredCountry = nextHover;
    scene.canvas.style.cursor = (picked && picked.id) ? "pointer" : "";
  }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

  handler.setInputAction(function (click) {
    var picked = scene.pick(click.position);
    if (!picked || !picked.id) {
      document.getElementById("popup-card").style.display = "none";
      return;
    }
    var item = picked.id;
    if (item instanceof Cesium.Entity && item.polygon) {
      if (STATE.highlightedCountry && STATE.highlightedCountry !== item) {
        setCountryStyle(STATE.highlightedCountry, countryNormalMat, countryNormalOutline);
      }
      STATE.highlightedCountry = item;
      setCountryStyle(item, countrySelectMat, countryBrightOutline);
      var cname = (item.properties && item.properties.ADMIN && item.properties.ADMIN.getValue())
        || item.name || "Region";
      var ray = viewer.camera.getPickRay(click.position);
      var cart = ray ? globe.pick(ray, scene) : null;
      var cpos = cart ? Cesium.Cartographic.fromCartesian(cart) : null;
      var lon = cpos ? Cesium.Math.toDegrees(cpos.longitude) : 0;
      var lat = cpos ? Cesium.Math.toDegrees(cpos.latitude) : 0;
      showPopupAt(lon, lat, 0, "<b>" + cname + "</b><br>Click ZOOM or search city to inspect.", {
        type: "country", summary: "Country: " + cname, data: { name: cname, lat: lat, lon: lon }
      });
      return;
    }

    if (item.kind === "quake") {
      var p = item.props;
      var t = p.time ? new Date(Number(p.time)).toUTCString() : "";
      showPopupAt(item.lon, item.lat, item.altM,
        "<b>EARTHQUAKE M " + p.mag + "</b><br>" + p.place + "<br>Depth: " + p.depth + " km"
        + (p.tsunami ? "<br><span style='color:#ff453a'>TSUNAMI ALERT</span>" : "") + "<br>" + t,
        { type: "earthquake", summary: "M" + p.mag + " Earthquake -- " + p.place + " (depth " + p.depth + " km)", data: p });
    } else if (item.kind === "flight") {
      var p = item.props;
      showPopupAt(item.lon, item.lat, item.altM,
        "<b>FLIGHT " + p.callsign + "</b> (" + (p.type || p.icao) + ")<br>"
        + "Origin/Reg: " + (p.country || "-") + "<br>"
        + "Alt: " + p.alt_ft + " ft \u00b7 Spd: " + p.spd_kts + " kts \u00b7 Hdg: " + p.heading + "\u00b0",
        { type: "flight", summary: "Flight " + p.callsign + " | " + p.alt_ft + " ft | " + p.spd_kts + " kts | Hdg " + p.heading + "\u00b0", data: p });
    } else if (item.kind === "sat") {
      var p = item.props;
      STATE.selectedSatId = String(p.id);
      drawSatelliteOrbit(STATE.selectedSatId);
      showPopupAt(item.lon, item.lat, item.altM,
        "<b>" + p.name + "</b> (NORAD " + p.id + ")<br>"
        + "Class: " + String(p.cat || "").toUpperCase() + " \u00b7 Inc: " + p.inc_deg + "\u00b0<br>"
        + "Alt: " + p.alt_km + " km",
        { type: "satellite", summary: "Satellite " + p.name + " (NORAD " + p.id + ") alt " + p.alt_km + " km", data: p });
    }
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

  try {
    STATE.satWorker = new Worker("/assets/sat-worker.js");
    STATE.satWorker.onmessage = function (ev) {
      var d = ev.data || {};
      if (d.type === "positions") {
        applySatelliteBuffer(d.count, d.buf, d.meta);
      }
    };
  } catch (e) {}

  function fetchEarthquakes() {
    if (!STATE.active) return;
    fetch("/api/earthquakes").then(function (r) { return r.json(); }).then(renderEarthquakes).catch(function () {});
  }

  function fetchFlights() {
    if (!STATE.active) return;
    fetch("/api/flights").then(function (r) { return r.json(); }).then(function (fc) {
      var feats = (fc && fc.features) || [];
      var list = [];
      for (var i = 0; i < feats.length; i++) {
        var f = feats[i];
        var c = f.geometry && f.geometry.coordinates;
        if (!c || c.length < 2) continue;
        list.push({ lon: c[0], lat: c[1], props: f.properties || {} });
      }
      STATE.allFlights = list;
      renderVisibleFlights();
    }).catch(function () {});
  }

  function fetchSatelliteCatalog() {
    if (!STATE.active) return;
    fetch("/api/satellites/tles").then(function (r) { return r.json(); }).then(function (d) {
      var tles = (d && d.tles) || [];
      STATE.satRecordsCount = tles.length;
      if (STATE.satWorker) {
        STATE.satWorker.postMessage({ type: "catalog", tles: tles, group: STATE.satGroup });
      }
    }).catch(function () {});
  }

  function zoomToHeightMeters(z) {
    if (!z || z <= 0.5) return 18500000;
    return Math.max(180.0, 62000000.0 / Math.pow(2.0, z));
  }

  applyVisualState();
  updateTopLeftHud();
  fetchEarthquakes();
  fetchFlights();
  fetchSatelliteCatalog();
  setTimeout(fetchEarthquakes, 3200);
  setTimeout(fetchFlights, 3600);
  setTimeout(fetchSatelliteCatalog, 4200);

  setInterval(fetchEarthquakes, 60000);
  setInterval(fetchFlights, 12000);
  setInterval(function () {
    if (STATE.active && STATE.satWorker && STATE.satRecordsCount > 0) {
      STATE.satWorker.postMessage({ type: "step", group: STATE.satGroup });
    }
  }, 1500);
  setInterval(fetchSatelliteCatalog, 120000);

  window.__MARK_GLOBE__ = {
    setActive: function (on) {
      STATE.active = !!on;
      if (STATE.active && STATE.satWorker && STATE.satRecordsCount > 0) {
        STATE.satWorker.postMessage({ type: "step", group: STATE.satGroup });
      }
    },
    setMode: function (mode) {
      STATE.mapMode = mode || "auto";
      applyVisualState();
    },
    setLayers: function (layersObj) {
      Object.keys(layersObj || {}).forEach(function (k) {
        if (k in STATE.layers) STATE.layers[k] = !!layersObj[k];
      });
      applyVisualState();
    },
    setSatGroup: function (grp) {
      STATE.satGroup = grp || "stations";
      applyVisualState();
      if (STATE.satWorker && STATE.satRecordsCount > 0) {
        STATE.satWorker.postMessage({ type: "step", group: STATE.satGroup });
      } else {
        fetchSatelliteCatalog();
      }
    },
    flyTo: function (lat, lon, zoom, satId, altKm) {
      var h = satId ? Math.max(2500000, ((altKm || 420) * 1000) + 1800000) : zoomToHeightMeters(zoom);
      viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(lon, lat, h),
        duration: 1.5
      });
      if (satId) {
        STATE.selectedSatId = String(satId);
        drawSatelliteOrbit(STATE.selectedSatId);
      }
    },
    refreshAll: function () {
      fetchEarthquakes();
      fetchFlights();
      fetchSatelliteCatalog();
      updateTelemetryBanner("SYNCING");
    }
  };

  if (window.__MARK_QUEUE__ && window.__MARK_QUEUE__.length) {
    var q = window.__MARK_QUEUE__.splice(0, window.__MARK_QUEUE__.length);
    for (var i = 0; i < q.length; i++) {
      window.__MARK_CALL__(q[i][0], q[i][1]);
    }
  }
})();
</script>
</body>
</html>
"""


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler(hub: _GlobeDataHub):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send_bytes(self, raw: bytes, content_type: str, cache_ctrl: str = "no-store", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", cache_ctrl)
            self.end_headers()
            self.wfile.write(raw)

        def _send_json(self, obj: dict, status: int = 200) -> None:
            raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
            self._send_bytes(raw, "application/json; charset=utf-8", "no-store", status)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                self._send_bytes(_GLOBE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/assets/sat-worker.js":
                self._send_bytes(_SAT_WORKER_JS.encode("utf-8"), "application/javascript; charset=utf-8")
                return

            if path == "/assets/satellite.min.js":
                data = hub.fetch_cached_asset(
                    "satellite.min.js",
                    (
                        "https://cdn.jsdelivr.net/npm/satellite.js@4.0.0/dist/satellite.min.js",
                        "https://unpkg.com/satellite.js@4.0.0/dist/satellite.min.js",
                    ),
                )
                if data is not None:
                    self._send_bytes(data, "application/javascript; charset=utf-8", "public, max-age=86400")
                else:
                    self.send_error(502)
                return

            if path.startswith("/cesium/"):
                sub = path[len("/cesium/"):]
                data = hub.fetch_cached_cesium_file(sub)
                if data is not None:
                    self._send_bytes(data, _guess_mime(sub), "public, max-age=86400")
                else:
                    self.send_error(404)
                return

            if path == "/api/countries":
                raw = hub.get_countries_geojson_bytes()
                self._send_bytes(raw, "application/json; charset=utf-8", "public, max-age=86400")
                return

            if path == "/api/earthquakes":
                self._send_json(hub.ensure_earthquakes())
                return

            if path == "/api/flights":
                self._send_json(hub.ensure_flights())
                return

            if path == "/api/satellites/tles":
                self._send_json(hub.get_satellite_tles_payload())
                return

            if path == "/api/satellites/positions":
                grp = (qs.get("group") or ["stations"])[0]
                self._send_json(hub.get_satellite_positions(grp))
                return

            if path == "/api/satellites/orbit":
                sat_id = (qs.get("id") or [""])[0]
                self._send_json(hub.get_satellite_orbit(sat_id))
                return

            if path == "/api/view":
                try:
                    lat = float((qs.get("lat") or ["20.0"])[0])
                    lon = float((qs.get("lon") or ["12.0"])[0])
                    hub.set_view_center(lat, lon)
                except Exception:
                    pass
                self._send_json({"ok": True})
                return

            self.send_error(404)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/select":
                try:
                    length = int(self.headers.get("Content-Length", "0") or 0)
                except ValueError:
                    length = 0
                body = self.rfile.read(max(0, min(length, 65536)))
                try:
                    obj = json.loads(body.decode("utf-8"))
                    hub.set_selected(obj)
                except Exception:
                    pass
                self._send_json({"ok": True})
                return
            self.send_error(404)

    return _Handler


def _get_server_url(hub: _GlobeDataHub) -> str:
    global _SERVER
    with _LOCK:
        if _SERVER is None:
            srv = _ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(hub))
            threading.Thread(target=srv.serve_forever, daemon=True).start()

            def _stop_server(s=srv):
                try:
                    s.shutdown()
                    s.server_close()
                except Exception:
                    pass

            atexit.register(_stop_server)
            _SERVER = srv
        port = _SERVER.server_address[1]
    return f"http://127.0.0.1:{port}/"


# -- Main-Thread Qt Bridge & Mark 57 Top-HUD UI -------------------------------

def _get_bridge(player):
    """Create the Qt bridge once and move it onto the Qt main thread."""
    global _BRIDGE, _ACTIVE_BRIDGE
    if _BRIDGE is not None:
        return _BRIDGE

    app = getattr(player, "_app", None)
    win = getattr(player, "_win", None)
    if app is None or win is None or not hasattr(win, "_hud_cam_stack"):
        return None

    from PyQt6.QtCore import QObject, QPointF, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
    from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, QPixmap
    from PyQt6.QtWidgets import (
        QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
    )
    from gui.theme import C
    from gui.style import btn_css, hud_font, px_font
    from gui.widgets import HoloPanel, IconButton

    class _PureQtGlobeCanvas(QWidget):
        """Hardware-fast 3D Textured Google Satellite/Streets + NASA City Lights Qt Globe + Slippy Tiles."""

        _tile_ready = pyqtSignal()

        def __init__(self, hub: _GlobeDataHub):
            super().__init__()
            self._hub = hub
            self._lat = 20.0
            self._lon = 12.0
            self._zoom = 1.8
            self._dragging = False
            self._last_pos = None
            self._layers = {
                "earthquakes": True, "flights": True, "satellites": True,
                "city_lights": True, "streets": True,
            }
            self._sat_group = "stations"
            self._map_mode = "auto"
            self._tile_lock = threading.Lock()
            self._img_cache: dict[str, QImage] = {}
            self._pix_cache: dict[str, QPixmap] = {}
            self._pending_tiles: set[str] = set()
            self._cached_sats: list[dict] = []
            self._cached_sats_ts = 0.0
            self._cached_sats_grp = ""
            self._tex_arrays: dict[str, object] = {}
            self._tex_loading = False
            self._tile_ready.connect(self.update, Qt.ConnectionType.QueuedConnection)
            self.setMouseTracking(True)
            self.setCursor(Qt.CursorShape.OpenHandCursor)

            # Ensure z=3 equirectangular textures are ready
            for tag in ("sat", "street", "lights"):
                self._hub.ensure_world_texture(tag, on_ready=self._tile_ready.emit)

        def set_mode(self, mode: str):
            self._map_mode = mode
            self.update()

        def set_layers(self, layers: dict):
            self._layers.update(layers)
            self.update()

        def set_sat_group(self, grp: str):
            self._sat_group = grp
            self._cached_sats_ts = 0.0
            self.update()

        def fly_to(self, lat: float, lon: float, zoom: float):
            self._lat = max(-85.0, min(85.0, lat))
            self._lon = ((lon + 180.0) % 360.0) - 180.0
            self._zoom = max(1.0, min(18.0, zoom))
            self._hub.set_view_center(self._lat, self._lon)
            self.update()

        def mousePressEvent(self, ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                self._dragging = True
                self._last_pos = ev.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

        def mouseReleaseEvent(self, ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                self._dragging = False
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                self._hub.set_view_center(self._lat, self._lon)
                self.update()

        def mouseMoveEvent(self, ev):
            if self._dragging and self._last_pos is not None:
                dx = ev.position().x() - self._last_pos.x()
                dy = ev.position().y() - self._last_pos.y()
                self._last_pos = ev.position()
                if self._zoom >= 4.2:
                    deg_lon = 180.0 / (256.0 * (2.0 ** (self._zoom - 1.0)))
                    deg_lat = deg_lon * max(0.15, math.cos(math.radians(self._lat)))
                else:
                    deg_lon = 0.45 / max(0.5, self._zoom)
                    deg_lat = deg_lon
                self._lon = ((self._lon - dx * deg_lon + 180.0) % 360.0) - 180.0
                self._lat = max(-85.0, min(85.0, self._lat + dy * deg_lat))
                self.update()

        def wheelEvent(self, ev):
            delta = ev.angleDelta().y()
            if delta == 0:
                return
            step = 0.6 if delta > 0 else -0.6
            self._zoom = max(1.0, min(18.0, self._zoom + step))
            self.update()

        def _active_tile_layer(self) -> str:
            m = self._map_mode
            show_lbl = self._layers.get("streets", True)
            if m == "street":
                return "street"
            if m == "city_lights":
                return "lights"
            if m == "dark":
                return "dark"
            return "hybrid" if show_lbl else "sat"

        def _fetch_tile_async(self, layer_tag: str, z: int, x: int, y: int):
            key = f"{layer_tag}/{z}/{x}/{y}"
            with self._tile_lock:
                if key in self._pix_cache or key in self._img_cache or key in self._pending_tiles:
                    return
                self._pending_tiles.add(key)

            def _job():
                try:
                    data = self._hub.fetch_tile(layer_tag, z, x, y)
                    if data:
                        img = QImage.fromData(data)
                        if not img.isNull():
                            with self._tile_lock:
                                if len(self._img_cache) > 128:
                                    self._img_cache.clear()
                                self._img_cache[key] = img
                            self._tile_ready.emit()
                finally:
                    with self._tile_lock:
                        self._pending_tiles.discard(key)

            threading.Thread(target=_job, daemon=True).start()

        def _load_tex_array(self, tag: str):
            if tag in self._tex_arrays:
                return self._tex_arrays[tag]
            p = self._hub.get_world_texture_path(tag)
            if not p.exists() or p.stat().st_size < 8000:
                self._hub.ensure_world_texture(tag, on_ready=self._tile_ready.emit)
                return None
            try:
                import numpy as np
                img = QImage(str(p)).convertToFormat(QImage.Format.Format_RGBA8888)
                if img.isNull():
                    return None
                w, h = img.width(), img.height()
                ptr = img.bits()
                ptr.setsize(h * w * 4)
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4)).copy()
                self._tex_arrays[tag] = arr
                return arr
            except Exception:
                return None

        def _render_textured_sphere(self, r_px: float) -> QImage | None:
            """Render real 3D Google Earth / NASA City Lights sphere using numpy."""
            try:
                import numpy as np
            except Exception:
                return None

            m = self._map_mode
            if m == "street":
                tex_main = self._load_tex_array("street") or self._load_tex_array("sat")
            elif m == "city_lights":
                tex_main = self._load_tex_array("lights")
            else:
                tex_main = self._load_tex_array("sat")

            tex_lights = self._load_tex_array("lights") if self._layers.get("city_lights", True) else None
            if tex_main is None:
                return None

            # Use slightly smaller internal buffer while actively dragging for 60 FPS
            size = int(max(180, min(420 if self._dragging else 520, r_px * 2.0)))
            if size < 64:
                return None

            H_eq, W_eq, _ = tex_main.shape
            lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
            nx, ny = np.meshgrid(lin, lin)
            x = nx
            y = -ny
            rho2 = x * x + y * y
            mask = rho2 <= 1.0
            cos_c = np.zeros_like(rho2)
            cos_c[mask] = np.sqrt(np.maximum(0.0, 1.0 - rho2[mask]))

            lat0 = math.radians(self._lat)
            lon0 = math.radians(self._lon)
            sin_la0 = math.sin(lat0)
            cos_la0 = math.cos(lat0)

            xm = x[mask]
            ym = y[mask]
            cc = cos_c[mask]

            sin_lat = np.clip(cc * sin_la0 + ym * cos_la0, -1.0, 1.0)
            lat = np.arcsin(sin_lat)
            lon = lon0 + np.arctan2(xm, cos_la0 * cc - ym * sin_la0)

            u = np.mod((lon + math.pi) / (2.0 * math.pi), 1.0)
            v = np.clip(0.5 - lat / math.pi, 0.0, 0.9999)
            tx = (u * (W_eq - 1)).astype(np.int32)
            ty = (v * (H_eq - 1)).astype(np.int32)

            rgb = tex_main[ty, tx, :3].astype(np.float32)

            # Real-time solar day/night terminator + NASA city lights
            if m in ("auto", "city_lights", "dark"):
                now = datetime.now(timezone.utc)
                doy = now.timetuple().tm_yday
                utc_h = now.hour + now.minute / 60.0 + now.second / 3600.0
                sun_lat = math.radians(-23.44 * math.cos(2.0 * math.pi * (doy + 10.0) / 365.25))
                sun_lon = math.radians((12.0 - utc_h) * 15.0)
                cos_sun = sin_lat * math.sin(sun_lat) + np.cos(lat) * math.cos(sun_lat) * np.cos(lon - sun_lon)

                if m == "auto":
                    day_f = np.clip((cos_sun + 0.14) / 0.28, 0.0, 1.0)[:, None]
                    night_rgb = rgb * 0.18
                    if tex_lights is not None:
                        lh, lw, _ = tex_lights.shape
                        ltx = (u * (lw - 1)).astype(np.int32)
                        lty = (v * (lh - 1)).astype(np.int32)
                        night_rgb = night_rgb + tex_lights[lty, ltx, :3].astype(np.float32) * 1.45
                    rgb = day_f * rgb + (1.0 - day_f) * np.clip(night_rgb, 0.0, 255.0)
                elif m == "city_lights":
                    rgb = np.clip(rgb * 1.5, 0.0, 255.0)
                elif m == "dark":
                    dark_rgb = rgb * 0.28
                    if tex_lights is not None:
                        lh, lw, _ = tex_lights.shape
                        ltx = (u * (lw - 1)).astype(np.int32)
                        lty = (v * (lh - 1)).astype(np.int32)
                        dark_rgb = dark_rgb + tex_lights[lty, ltx, :3].astype(np.float32) * 0.9
                    rgb = np.clip(dark_rgb, 0.0, 255.0)

            # Atmospheric blue limb Fresnel glow
            rim = np.power(np.maximum(0.0, 1.0 - cc), 3.0)[:, None]
            rgb[:, 0] += 18.0 * rim[:, 0]
            rgb[:, 1] += 130.0 * rim[:, 0]
            rgb[:, 2] += 250.0 * rim[:, 0]

            out = np.zeros((size, size, 4), dtype=np.uint8)
            out[mask, :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
            out[mask, 3] = 255

            qimg = QImage(out.data, size, size, size * 4, QImage.Format.Format_RGBA8888)
            return qimg.copy()

        def _project_ortho(self, lat: float, lon: float, cx: float, cy: float, r: float):
            la1 = math.radians(self._lat)
            lo1 = math.radians(self._lon)
            la2 = math.radians(lat)
            lo2 = math.radians(lon)
            cos_c = math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(lo2 - lo1)
            if cos_c < 0:
                return None
            x = cx + r * math.cos(la2) * math.sin(lo2 - lo1)
            y = cy - r * (math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1))
            return (x, y)

        def paintEvent(self, ev):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            w, h = self.width(), self.height()
            p.fillRect(0, 0, w, h, QColor("#01050b"))

            use_slippy = (self._zoom >= 4.2)
            if use_slippy:
                self._paint_street_mode(p, w, h)
            else:
                self._paint_globe_mode(p, w, h)

            now = datetime.now(timezone.utc)
            p.setPen(QColor("#8edfff"))
            p.setFont(QFont("Consolas", 9))
            mode_label = self._map_mode.upper()
            p.drawText(
                10, 18,
                f"{now.strftime('%H:%M UTC')} | {mode_label} | LAT {self._lat:.2f}, LON {self._lon:.2f} | Z {self._zoom:.1f}",
            )
            p.end()

        def _paint_street_mode(self, p: QPainter, w: int, h: int):
            layer_tag = self._active_tile_layer()
            z = max(1, min(19, int(round(self._zoom))))
            n = 2 ** z
            cx_tile = (self._lon + 180.0) / 360.0 * n
            lat_rad = math.radians(max(-85.0, min(85.0, self._lat)))
            cy_tile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
            tx0 = int(math.floor(cx_tile - w / 512.0)) - 1
            tx1 = int(math.floor(cx_tile + w / 512.0)) + 1
            ty0 = int(math.floor(cy_tile - h / 512.0)) - 1
            ty1 = int(math.floor(cy_tile + h / 512.0)) + 1

            for tx in range(tx0, tx1 + 1):
                for ty in range(max(0, ty0), min(n - 1, ty1) + 1):
                    wx = tx % n
                    px = w / 2.0 + (tx - cx_tile) * 256.0
                    py = h / 2.0 + (ty - cy_tile) * 256.0
                    key = f"{layer_tag}/{z}/{wx}/{ty}"
                    with self._tile_lock:
                        pix = self._pix_cache.get(key)
                        if pix is None and key in self._img_cache:
                            pix = QPixmap.fromImage(self._img_cache.pop(key))
                            if len(self._pix_cache) > 384:
                                self._pix_cache.clear()
                            self._pix_cache[key] = pix
                    if pix is not None:
                        p.drawPixmap(int(px), int(py), 256, 256, pix)
                    else:
                        p.setPen(QColor("#103048"))
                        p.drawRect(int(px), int(py), 256, 256)
                        self._fetch_tile_async(layer_tag, z, wx, ty)

            def to_xy(lat, lon):
                tx = (lon + 180.0) / 360.0 * n
                lr = math.radians(max(-85.0, min(85.0, lat)))
                ty = (1.0 - math.asinh(math.tan(lr)) / math.pi) / 2.0 * n
                return (w / 2.0 + (tx - cx_tile) * 256.0, h / 2.0 + (ty - cy_tile) * 256.0)

            self._paint_overlays(p, w, h, to_xy)

        def _paint_globe_mode(self, p: QPainter, w: int, h: int):
            cx, cy = w / 2.0, h / 2.0
            r = min(w, h) * 0.43 * min(2.2, max(0.75, self._zoom * 0.50))

            # Outer atmospheric blue halo
            for ring_w, alpha in ((14.0, 26), (7.0, 58), (2.5, 130)):
                p.setPen(QPen(QColor(0, 175, 255, alpha), ring_w))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(cx, cy), r + 2.0, r + 2.0)

            sphere_img = self._render_textured_sphere(r)
            if sphere_img is not None:
                p.drawImage(int(cx - r), int(cy - r), sphere_img.scaled(int(r * 2), int(r * 2), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                p.setPen(QPen(QColor(0, 212, 255, 130), 2.0))
                p.setBrush(QColor("#031936"))
                p.drawEllipse(QPointF(cx, cy), r, r)
                rings = self._hub.get_country_rings()
                if rings:
                    p.setPen(QPen(QColor(56, 217, 255, 70), 1.0))
                    p.setBrush(QColor("#12394a"))
                    for ring in rings:
                        pts = []
                        for lat, lon in ring:
                            pt = self._project_ortho(lat, lon, cx, cy, r)
                            if pt:
                                pts.append(QPointF(pt[0], pt[1]))
                        if len(pts) >= 3:
                            p.drawPolygon(QPolygonF(pts))

            self._paint_overlays(p, w, h, lambda la, lo: self._project_ortho(la, lo, cx, cy, r))

        def _get_cached_sats(self) -> list[dict]:
            now = time.monotonic()
            if (now - self._cached_sats_ts >= 1.5) or (self._cached_sats_grp != self._sat_group):
                self._cached_sats = self._hub.get_satellite_positions(self._sat_group).get("features", [])
                self._cached_sats_ts = now
                self._cached_sats_grp = self._sat_group
            return self._cached_sats

        def _paint_overlays(self, p: QPainter, w: int, h: int, proj_fn):
            if self._layers.get("satellites", True):
                sats = self._get_cached_sats()
                p.setPen(Qt.PenStyle.NoPen)
                for f in sats:
                    lon, lat = f["geometry"]["coordinates"]
                    pt = proj_fn(lat, lon)
                    if pt and 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                        is_st = f["properties"].get("cat") == "stations"
                        p.setBrush(QColor("#ffea00" if is_st else "#5eff9c"))
                        sz = 3.8 if is_st else 1.8
                        p.drawEllipse(QPointF(*pt), sz, sz)

            if self._layers.get("flights", True):
                flts = self._hub.ensure_flights().get("features", [])[:500]
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#38d9ff"))
                for f in flts:
                    lon, lat = f["geometry"]["coordinates"]
                    pt = proj_fn(lat, lon)
                    if pt and 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                        p.drawEllipse(QPointF(*pt), 2.1, 2.1)

            if self._layers.get("earthquakes", True):
                eqs = self._hub.ensure_earthquakes().get("features", [])
                for f in eqs:
                    lon, lat = f["geometry"]["coordinates"]
                    mag = float(f["properties"].get("mag", 1.0))
                    pt = proj_fn(lat, lon)
                    if pt and 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                        rad = max(3.0, mag * 2.4)
                        p.setPen(QPen(QColor("#ff9500"), 1.5))
                        p.setBrush(QColor(255, 149, 0, 105))
                        p.drawEllipse(QPointF(*pt), rad, rad)

    class _TopGlobeWidget(HoloPanel):
        _search_done_sig = pyqtSignal(str, object)

        def __init__(self, main_win, on_close):
            super().__init__("EARTH GLOBE", "GOOGLE / CESIUM 3D")
            self._win = main_win
            self._on_close = on_close
            self._hub = _get_hub()
            self._web_view = None
            self._web_loaded = False
            self._pending_js: list[str] = []
            self._use_web = False
            self._web_error = ""
            self._current_mode = "auto"
            self._current_sat_group = "stations"
            self._current_layers = {
                "earthquakes": True, "flights": True, "satellites": True,
                "city_lights": True, "streets": True,
            }
            self._last_fly_target = None  # (lat, lon, zoom, sat_id, alt_km)
            self._search_done_sig.connect(self._on_search_done, Qt.ConnectionType.QueuedConnection)

            # Header controls
            self._mode_btn = QPushButton("2D/QT")
            self._mode_btn.setFixedHeight(22)
            self._mode_btn.setToolTip("Toggle between WebGL Cesium Globe and Textured Qt Globe")
            self._mode_btn.setFont(px_font(9, spacing=1.0))
            self._mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._mode_btn.setStyleSheet(btn_css("dim", align="center"))
            self._mode_btn.clicked.connect(self._toggle_renderer)
            self._mode_btn.setVisible(False)
            self.add_header_widget(self._mode_btn)

            refresh_btn = QPushButton("SYNC")
            refresh_btn.setFixedHeight(22)
            refresh_btn.setFont(px_font(9, spacing=1.0))
            refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            refresh_btn.setStyleSheet(btn_css("dim", align="center"))
            refresh_btn.clicked.connect(self._sync_all)
            self.add_header_widget(refresh_btn)

            close_btn = IconButton("close", 22)
            close_btn.setToolTip("Close Earth Globe (return to HUD)")
            close_btn.clicked.connect(self._on_close)
            self.add_header_widget(close_btn)

            # Top control strip
            ctrl = QHBoxLayout()
            ctrl.setSpacing(6)

            self._search_edit = QLineEdit()
            self._search_edit.setPlaceholderText("Fly to city, street, country, lat,lon, ISS, or flight...")
            self._search_edit.setFont(hud_font(9))
            self._search_edit.setFixedHeight(26)
            self._search_edit.setStyleSheet(f"""
                QLineEdit {{
                    background: {C.PANEL2}; color: {C.WHITE};
                    border: 1px solid {C.BORDER}; padding: 2px 8px;
                }}
                QLineEdit:focus {{ border: 1px solid {C.PRI}; }}
            """)
            self._search_edit.returnPressed.connect(self._on_search)
            ctrl.addWidget(self._search_edit, stretch=2)

            go_btn = QPushButton("FLY TO")
            go_btn.setFixedHeight(26)
            go_btn.setFont(px_font(9, spacing=1.0))
            go_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            go_btn.setStyleSheet(btn_css("primary", align="center"))
            go_btn.clicked.connect(self._on_search)
            ctrl.addWidget(go_btn)

            combo_css = f"""
                QComboBox {{
                    background: {C.PANEL2}; color: {C.TEXT};
                    border: 1px solid {C.BORDER}; padding: 2px 6px;
                }}
                QComboBox QAbstractItemView {{
                    background: {C.DARK}; color: {C.TEXT};
                    selection-background-color: {C.PRI_GHO};
                    border: 1px solid {C.BORDER_B};
                }}
            """
            self._mode_box = QComboBox()
            self._mode_box.addItem("AUTO (DAY/NIGHT\u2192STREET)", "auto")
            self._mode_box.addItem("CITY LIGHTS", "city_lights")
            self._mode_box.addItem("GOOGLE STREETS", "street")
            self._mode_box.addItem("GOOGLE SATELLITE", "satellite")
            self._mode_box.addItem("DARK HUD", "dark")
            self._mode_box.setFont(px_font(9))
            self._mode_box.setFixedHeight(26)
            self._mode_box.setStyleSheet(combo_css)
            self._mode_box.currentIndexChanged.connect(self._on_mode_changed)
            ctrl.addWidget(self._mode_box)

            self._sat_box = QComboBox()
            self._sat_box.addItem("KEY STATIONS", "stations")
            self._sat_box.addItem("ALL SATS", "all")
            self._sat_box.addItem("STARLINK", "starlink")
            self._sat_box.addItem("GPS / NAV", "nav")
            self._sat_box.addItem("WEATHER", "weather")
            self._sat_box.setFont(px_font(9))
            self._sat_box.setFixedHeight(26)
            self._sat_box.setStyleSheet(combo_css)
            self._sat_box.currentIndexChanged.connect(self._on_sat_group_changed)
            ctrl.addWidget(self._sat_box)

            ask_btn = QPushButton("TO JARVIS")
            ask_btn.setFixedHeight(26)
            ask_btn.setFont(px_font(9, spacing=1.0))
            ask_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            ask_btn.setStyleSheet(btn_css("choice", align="center"))
            ask_btn.clicked.connect(self._send_to_jarvis)
            ctrl.addWidget(ask_btn)

            self.body.addLayout(ctrl)

            # Second bar: layer toggle checkboxes
            lyr_row = QHBoxLayout()
            lyr_row.setSpacing(10)
            self._layer_checks: dict[str, QCheckBox] = {}
            chk_css = f"""
                QCheckBox {{ color: {C.TEXT}; spacing: 4px; background: transparent; }}
                QCheckBox::indicator {{
                    width: 12px; height: 12px;
                    border: 1px solid {C.BORDER_B}; background: {C.PANEL2};
                }}
                QCheckBox::indicator:checked {{
                    background: {C.PRI}; border: 1px solid {C.WHITE};
                }}
            """
            for key, label in (
                ("earthquakes", "EARTHQUAKES"),
                ("flights", "FLIGHTS"),
                ("satellites", "SATELLITES"),
                ("city_lights", "CITY LIGHTS"),
                ("streets", "LABELS/STREETS"),
            ):
                cb = QCheckBox(label)
                cb.setChecked(True)
                cb.setFont(px_font(9))
                cb.setStyleSheet(chk_css)
                cb.toggled.connect(self._on_layers_changed)
                lyr_row.addWidget(cb)
                self._layer_checks[key] = cb

            lyr_row.addStretch(1)
            self._status_lbl = QLabel("Ready")
            self._status_lbl.setFont(px_font(9))
            self._status_lbl.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
            lyr_row.addWidget(self._status_lbl)
            self.body.addLayout(lyr_row)

            # Live dependency auto-install progress bar (hidden unless installing)
            self._install_bar_wrap = QWidget()
            self._install_bar_wrap.setStyleSheet(f"background: {C.PANEL2}; border: 1px solid {C.BORDER};")
            ibar_lay = QHBoxLayout(self._install_bar_wrap)
            ibar_lay.setContentsMargins(8, 4, 8, 4)
            ibar_lay.setSpacing(8)

            self._install_lbl = QLabel("Installing Earth Globe dependencies...")
            self._install_lbl.setFont(px_font(9))
            self._install_lbl.setStyleSheet(f"color: {C.PRI}; border: none; background: transparent;")
            ibar_lay.addWidget(self._install_lbl, stretch=1)

            self._install_prog = QProgressBar()
            self._install_prog.setFixedHeight(14)
            self._install_prog.setFixedWidth(210)
            self._install_prog.setRange(0, 100)
            self._install_prog.setValue(0)
            self._install_prog.setTextVisible(True)
            self._install_prog.setFont(px_font(8))
            self._install_prog.setStyleSheet(f"""
                QProgressBar {{
                    background: {C.DARK}; color: {C.WHITE};
                    border: 1px solid {C.BORDER_B}; text-align: center;
                }}
                QProgressBar::chunk {{
                    background: {C.PRI};
                }}
            """)
            ibar_lay.addWidget(self._install_prog)
            self._install_bar_wrap.setVisible(False)
            self.body.addWidget(self._install_bar_wrap)

            # Viewport stack (QWebEngineView primary when available, Textured 3D Qt fallback)
            self._view_stack = QStackedWidget()
            self._qt_canvas = _PureQtGlobeCanvas(self._hub)
            self._view_stack.addWidget(self._qt_canvas)
            self.body.addWidget(self._view_stack, stretch=1)

            if _is_webengine_installed():
                self._activate_webengine()

            self._timer = QTimer(self)
            self._timer.setInterval(2500)
            self._timer.timeout.connect(self._update_hud_badge)
            self._timer.start()

            self._hub.ensure_earthquakes()
            self._hub.ensure_flights()
            self._hub.ensure_satellites()

        def _activate_webengine(self) -> bool:
            """Import and attach QWebEngineView on the Qt main thread."""
            if self._web_view is not None:
                self._use_web = True
                self._mode_btn.setVisible(True)
                self._view_stack.setCurrentWidget(self._web_view)
                return True

            try:
                _refresh_import_paths()
                from PyQt6.QtWebEngineWidgets import QWebEngineView
                from PyQt6.QtWebEngineCore import QWebEngineSettings

                self._web_loaded = False
                wv = QWebEngineView()
                wv.page().setBackgroundColor(QColor("#01050b"))
                settings = wv.settings()
                settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
                wv.loadFinished.connect(self._on_web_load_finished)
                wv.setUrl(QUrl(_get_server_url(self._hub)))
                self._web_view = wv
                self._use_web = True
                self._web_error = ""
                self._mode_btn.setVisible(True)
                self._view_stack.addWidget(wv)
                self._view_stack.setCurrentWidget(wv)
                self._sync_state_to_web()
                return True
            except Exception as exc:
                self._web_error = str(exc)
                self._web_view = None
                self._use_web = False
                self._mode_btn.setVisible(False)
                self._view_stack.setCurrentWidget(self._qt_canvas)
                self._status_lbl.setText(f"Qt 3D Mode ({self._web_error[:45]})")
                if sys.stdout is not None:
                    try:
                        print(f"[EarthGlobe] WebEngine fallback reason: {exc}", flush=True)
                    except Exception:
                        pass
                return False

        def _on_web_load_finished(self, ok: bool):
            self._web_loaded = True
            if self._pending_js:
                queued = self._pending_js[:]
                self._pending_js.clear()
                if self._web_view is not None:
                    for script in queued:
                        self._web_view.page().runJavaScript(script)

        def _sync_state_to_web(self):
            self._run_js(f"window.__MARK_CALL__('setMode', [{json.dumps(self._current_mode)}]);")
            self._run_js(f"window.__MARK_CALL__('setSatGroup', [{json.dumps(self._current_sat_group)}]);")
            self._run_js(f"window.__MARK_CALL__('setLayers', [{json.dumps(self._current_layers)}]);")
            if self._last_fly_target is not None:
                lat, lon, z, sat_id, alt_km = self._last_fly_target
                self._run_js(f"window.__MARK_CALL__('flyTo', [{lat}, {lon}, {z}, {json.dumps(sat_id)}, {json.dumps(alt_km)}]);")
            else:
                self._run_js(
                    f"window.__MARK_CALL__('flyTo', [{self._qt_canvas._lat}, {self._qt_canvas._lon}, {self._qt_canvas._zoom}, null, null]);"
                )

        def show_install_progress(self, pct: int, text: str):
            self._install_bar_wrap.setVisible(True)
            self._install_lbl.setText(text)
            if pct < 0:
                self._install_prog.setRange(0, 0)
            else:
                if self._install_prog.maximum() == 0:
                    self._install_prog.setRange(0, 100)
                self._install_prog.setValue(max(0, min(100, pct)))
            self._status_lbl.setText(text[:64])

        def on_install_finished(self, ok: bool, note: str):
            if ok:
                self._install_prog.setRange(0, 100)
                self._install_prog.setValue(100)
                activated = self._activate_webengine()
                if activated:
                    self._install_lbl.setText("PyQt6-WebEngine active -- 3D Google/Cesium Globe ready")
                    self._status_lbl.setText("3D WebGL Globe activated.")
                else:
                    self._install_lbl.setText("Installed -- 3D Qt Globe active (restart app for WebGL)")
                    self._status_lbl.setText("3D Qt Globe active (restart for WebGL).")
                QTimer.singleShot(2500, self._install_bar_wrap.hide)
            else:
                self._install_prog.setRange(0, 100)
                self._install_lbl.setText(note[:85])
                self._status_lbl.setText(note[:72])
                QTimer.singleShot(3500, self._install_bar_wrap.hide)
            self._qt_canvas.update()

        def _toggle_renderer(self):
            if self._web_view is None:
                return
            show_web = self._view_stack.currentWidget() is not self._web_view
            self._view_stack.setCurrentWidget(self._web_view if show_web else self._qt_canvas)
            self._run_js(f"window.__MARK_CALL__('setActive', [{json.dumps(show_web)}]);")

        def set_active(self, active: bool):
            if active:
                if not self._timer.isActive():
                    self._timer.start()
                if self._web_view is not None and self._view_stack.currentWidget() is self._web_view:
                    self._run_js("window.__MARK_CALL__('setActive', [true]);")
            else:
                self._timer.stop()
                self._run_js("window.__MARK_CALL__('setActive', [false]);")

        def _run_js(self, script: str):
            if self._web_view is None:
                return
            if not self._web_loaded:
                if len(self._pending_js) < 64:
                    self._pending_js.append(script)
                return
            self._web_view.page().runJavaScript(script)

        def _update_hud_badge(self):
            with self._hub._lock:
                eq_n = len(self._hub.earthquakes.get("features", []))
                fl_n = len(self._hub.flights.get("features", []))
                sat_n = len(self._hub.sat_tles)
            self.set_code(f"{sat_n} SATS \u00b7 {fl_n} FLT \u00b7 {eq_n} EQ")
            sel = self._hub.get_selected()
            if sel and sel.get("summary") and not self._install_bar_wrap.isVisible():
                self._status_lbl.setText(str(sel["summary"])[:72])
            if self._view_stack.currentWidget() is self._qt_canvas:
                self._qt_canvas.update()

        def _sync_all(self):
            self._hub.refresh_all()
            self._status_lbl.setText("Refreshing USGS / OpenSky / CelesTrak...")
            self._run_js("window.__MARK_CALL__('refreshAll', []);")

        def _on_mode_changed(self):
            mode = self._mode_box.currentData()
            self._current_mode = mode
            self._run_js(f"window.__MARK_CALL__('setMode', [{json.dumps(mode)}]);")
            self._qt_canvas.set_mode(mode)

        def _on_sat_group_changed(self):
            grp = self._sat_box.currentData()
            self._current_sat_group = grp
            self._run_js(f"window.__MARK_CALL__('setSatGroup', [{json.dumps(grp)}]);")
            self._qt_canvas.set_sat_group(grp)

        def _on_layers_changed(self):
            state = {k: cb.isChecked() for k, cb in self._layer_checks.items()}
            self._current_layers = state
            self._run_js(f"window.__MARK_CALL__('setLayers', [{json.dumps(state)}]);")
            self._qt_canvas.set_layers(state)

        def _on_search(self):
            q = self._search_edit.text().strip()
            if not q:
                return
            self._status_lbl.setText(f"Locating '{q}'...")

            def _job():
                res = self._hub.geocode(q)
                if res:
                    self._hub.set_selected({"type": res.get("kind"), "summary": res.get("label"), "data": res})
                self._search_done_sig.emit(q, res)

            threading.Thread(target=_job, daemon=True).start()

        @pyqtSlot(str, object)
        def _on_search_done(self, q: str, res):
            if res:
                self.fly_to_result(res)
            else:
                self._status_lbl.setText(f"Not found: {q}")

        def fly_to_result(self, res: dict, zoom_override: float | None = None):
            lat = float(res["lat"])
            lon = float(res["lon"])
            z = float(zoom_override if zoom_override is not None else res.get("zoom", 11.0))
            sat_id = res.get("sat_id")
            alt_km = res.get("alt_km")
            self._last_fly_target = (lat, lon, z, sat_id, alt_km)
            self._status_lbl.setText(f"Target: {res.get('label', '')}")
            self._run_js(f"window.__MARK_CALL__('flyTo', [{lat}, {lon}, {z}, {json.dumps(sat_id)}, {json.dumps(alt_km)}]);")
            self._qt_canvas.fly_to(lat, lon, z)

        def apply_command(self, cmd: dict):
            if "map_mode" in cmd and cmd["map_mode"]:
                idx = self._mode_box.findData(cmd["map_mode"])
                if idx >= 0:
                    self._mode_box.setCurrentIndex(idx)
            if "sat_group" in cmd and cmd["sat_group"]:
                idx = self._sat_box.findData(cmd["sat_group"])
                if idx >= 0:
                    self._sat_box.setCurrentIndex(idx)
            if "layers" in cmd and isinstance(cmd["layers"], list):
                wanted = set(cmd["layers"])
                for k, cb in self._layer_checks.items():
                    cb.setChecked(k in wanted)
            if "geo" in cmd and isinstance(cmd["geo"], dict):
                self.fly_to_result(cmd["geo"], cmd.get("zoom_num"))
            elif "zoom_num" in cmd and cmd["zoom_num"] is not None:
                z = float(cmd["zoom_num"])
                with self._hub._lock:
                    lat, lon = self._hub._view_lat, self._hub._view_lon
                self._last_fly_target = (lat, lon, z, None, None)
                self._run_js(f"window.__MARK_CALL__('flyTo', [{lat}, {lon}, {z}, null, null]);")
                self._qt_canvas.fly_to(lat, lon, z)
            if cmd.get("refresh"):
                self._sync_all()

        def _send_to_jarvis(self):
            sel = self._hub.get_selected()
            with self._hub._lock:
                eq_n = len(self._hub.earthquakes.get("features", []))
                fl_n = len(self._hub.flights.get("features", []))
                sat_n = len(self._hub.sat_tles)
                lat, lon = self._hub._view_lat, self._hub._view_lon
            summary = (sel or {}).get("summary") or f"View centered near {lat:.2f}, {lon:.2f}"
            msg = (
                f"[EARTH GLOBE TELEMETRY]\n"
                f"Target: {summary}\n"
                f"Live counts -- Satellites: {sat_n}, Flights: {fl_n}, Earthquakes (24h): {eq_n}."
            )
            if self._win.on_text_command:
                threading.Thread(target=self._win.on_text_command, args=(msg,), daemon=True).start()
                self._status_lbl.setText("Telemetry sent to JARVIS.")

    class _QtBridge(QObject):
        _open_sig = pyqtSignal(str)
        _close_sig = pyqtSignal()
        _install_progress_sig = pyqtSignal(int, str)
        _install_done_sig = pyqtSignal(bool, str)

        def __init__(self, qt_app, main_win):
            super().__init__()
            self._win = main_win
            self._panel = None
            self._globe_ui = None
            self._pending_install_prog = None
            self._pending_install_done = None
            self.moveToThread(qt_app.thread())
            self._open_sig.connect(self._on_open, Qt.ConnectionType.QueuedConnection)
            self._close_sig.connect(self._on_close, Qt.ConnectionType.QueuedConnection)
            self._install_progress_sig.connect(self._on_install_progress, Qt.ConnectionType.QueuedConnection)
            self._install_done_sig.connect(self._on_install_done, Qt.ConnectionType.QueuedConnection)

        @pyqtSlot(str)
        def _on_open(self, cmd_json: str):
            stack = self._win._hud_cam_stack
            if getattr(self._win, "_video_on", False):
                self._win._on_video_close()
            if stack.currentIndex() == 1:
                self._win._cam_stop.set()

            if self._panel is None:
                container = QWidget()
                container.setStyleSheet(f"background: {C.BG};")
                lay = QVBoxLayout(container)
                lay.setContentsMargins(6, 6, 6, 6)
                self._globe_ui = _TopGlobeWidget(self._win, self._on_close)
                lay.addWidget(self._globe_ui)
                self._panel = container
                stack.addWidget(self._panel)

                prog = self._pending_install_prog or _LAST_INSTALL_PROG
                if prog is not None:
                    self._globe_ui.show_install_progress(*prog)
                if self._pending_install_done is not None:
                    done = self._pending_install_done
                    self._pending_install_done = None
                    self._globe_ui.on_install_finished(*done)

            self._globe_ui.set_active(True)
            stack.setCurrentWidget(self._panel)
            if cmd_json:
                try:
                    cmd = json.loads(cmd_json)
                    if cmd:
                        self._globe_ui.apply_command(cmd)
                except Exception:
                    pass

        @pyqtSlot()
        def _on_close(self):
            stack = self._win._hud_cam_stack
            if self._globe_ui is not None:
                self._globe_ui.set_active(False)
            if self._panel is not None and stack.currentWidget() is self._panel:
                stack.setCurrentIndex(0)

        @pyqtSlot(int, str)
        def _on_install_progress(self, pct: int, text: str):
            self._pending_install_prog = (pct, text)
            self._pending_install_done = None
            if self._globe_ui is not None:
                self._globe_ui.show_install_progress(pct, text)

        @pyqtSlot(bool, str)
        def _on_install_done(self, ok: bool, note: str):
            self._pending_install_done = (ok, note)
            self._pending_install_prog = None
            if self._globe_ui is not None:
                self._globe_ui.on_install_finished(ok, note)

    with _LOCK:
        if _BRIDGE is None:
            _BRIDGE = _QtBridge(app, win)
            _ACTIVE_BRIDGE = _BRIDGE
    return _BRIDGE


# -- Helpers for Parsing Tool Parameters --------------------------------------

def _parse_zoom(raw_zoom) -> float | None:
    if raw_zoom is None or raw_zoom == "":
        return None
    s = str(raw_zoom).strip().lower()
    presets = {"globe": 1.8, "earth": 1.8, "region": 4.8, "country": 5.2, "city": 11.5, "street": 16.0}
    if s in presets:
        return presets[s]
    try:
        return max(0.5, min(19.0, float(s)))
    except ValueError:
        return None


# -- Plugin Entry Point (worker thread) ---------------------------------------

def run(parameters: dict, player=None, session_memory=None) -> str:
    p = parameters or {}
    action = str(p.get("action") or "open").strip().lower()
    location = str(p.get("location") or "").strip()
    map_mode = str(p.get("map_mode") or "").strip().lower()
    sat_group = str(p.get("sat_group") or "").strip().lower()
    layers = p.get("layers")
    zoom_num = _parse_zoom(p.get("zoom"))

    try:
        hub = _get_hub()

        if action == "close":
            if _BRIDGE is not None:
                _BRIDGE._close_sig.emit()
            return "Closed the Earth Globe and restored the main display."

        if action == "status":
            with hub._lock:
                eq_n = len(hub.earthquakes.get("features", []))
                fl_n = len(hub.flights.get("features", []))
                sat_n = len(hub.sat_tles)
                vlat, vlon = hub._view_lat, hub._view_lon
            return (
                f"Earth Globe telemetry: {sat_n} tracked satellites, {fl_n} live flights, "
                f"{eq_n} earthquakes (24h). View center: {vlat:.2f}, {vlon:.2f}."
            )

        bridge = _get_bridge(player) if player is not None else None

        hub.ensure_earthquakes(force=(action == "refresh"))
        hub.ensure_flights(force=(action == "refresh"))
        hub.ensure_satellites(force=(action == "refresh"))

        cmd: dict = {}
        if map_mode in ("auto", "city_lights", "street", "satellite", "dark"):
            cmd["map_mode"] = map_mode
        if sat_group in ("all", "stations", "starlink", "nav", "weather"):
            cmd["sat_group"] = sat_group
        if isinstance(layers, str) and layers.strip():
            layers = [x.strip() for x in layers.split(",") if x.strip()]
        if isinstance(layers, list) and layers:
            cmd["layers"] = [str(x).strip().lower() for x in layers]
        if zoom_num is not None:
            cmd["zoom_num"] = zoom_num
        if action == "refresh":
            cmd["refresh"] = True

        geo = None
        if location:
            geo = hub.geocode(location)
            if geo:
                cmd["geo"] = geo
            elif action == "focus":
                return f"Could not find location or target '{location}'."

        if bridge is not None:
            bridge._open_sig.emit(json.dumps(cmd))
            _log_sys(player, "Earth Globe opened in top HUD.")

        install_note = _start_dependency_install(player=player, console=False)

        parts = ["Earth Globe is open on the top HUD"]
        if geo:
            parts.append(f"focused on {geo.get('label', location)}")
        if map_mode:
            parts.append(f"in {map_mode} mode")
        result = ", ".join(parts) + "."
        if install_note:
            result += f" ({install_note})"
        return result

    except Exception as e:
        return f"Earth Globe failed: {e}"


# -- Proactive Startup Dependency & Asset Bootstrap ---------------------------

def _bootstrap_dependencies_on_import() -> None:
    """Install PyQt6-WebEngine & numpy and pre-cache world assets right on startup."""
    global _INSTALL_ATTEMPTED, _INSTALL_IN_PROGRESS
    hub = _get_hub()
    threading.Thread(target=hub.prewarm_assets, daemon=True).start()

    if getattr(sys, "frozen", False):
        return

    need_web = not _is_webengine_installed()
    need_np = not _is_numpy_installed()
    if not need_web and not need_np:
        return

    qt_inst = None
    try:
        from PyQt6.QtCore import QCoreApplication
        qt_inst = QCoreApplication.instance()
    except Exception:
        pass

    if qt_inst is None:
        with _INSTALL_LOCK:
            _INSTALL_ATTEMPTED = True
            _INSTALL_IN_PROGRESS = True
        ok, _ = _install_dependencies_worker(player=None, console=True)
        if ok:
            try:
                importlib.import_module("PyQt6.QtWebEngineWidgets")
                importlib.import_module("PyQt6.QtWebEngineCore")
            except Exception:
                pass
    else:
        _start_dependency_install(player=None, console=True)


_bootstrap_dependencies_on_import()