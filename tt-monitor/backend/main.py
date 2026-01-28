"""TT-Monitor: Tenstorrent Device Monitoring Dashboard Backend with vLLM metrics."""

import asyncio
import glob
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="TT-Monitor", description="Tenstorrent Device Monitoring with vLLM metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MOCK_MODE = os.environ.get("TT_MONITOR_MOCK", "0") == "1"
VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://localhost:8088/metrics")
DATA_DIR = os.environ.get("TT_MONITOR_DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "metrics.db")

# Sysfs paths
SYSFS_TT_CLASS = "/sys/class/tenstorrent"
SYSFS_HWMON_CLASS = "/sys/class/hwmon"

# Prometheus metrics (for /metrics endpoint)
DEVICE_TEMP = Gauge("tt_device_temperature_celsius", "Device temperature", ["device", "board"])
DEVICE_POWER = Gauge("tt_device_power_watts", "Device power consumption", ["device", "board"])
DEVICE_VOLTAGE = Gauge("tt_device_voltage_volts", "Device voltage", ["device", "board"])
DEVICE_CURRENT = Gauge("tt_device_current_amps", "Device current", ["device", "board"])
DEVICE_AICLK = Gauge("tt_device_aiclk_mhz", "AI clock frequency", ["device", "board"])
DEVICE_ARCCLK = Gauge("tt_device_arcclk_mhz", "ARC clock frequency", ["device", "board"])
DEVICES_TOTAL = Gauge("tt_devices_total", "Total number of TT devices")
SCRAPE_ERRORS = Counter("tt_scrape_errors_total", "Total scrape errors")

# Time ranges in seconds
TIME_RANGES = {
    "10m": 10 * 60,
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
}


@dataclass
class DeviceInfo:
    device_id: str
    board_type: str
    pci_index: int
    pci_bdf: str = ""
    temperature: float = 0.0
    power: float = 0.0
    voltage: float = 0.0
    current: float = 0.0
    aiclk: float = 0.0
    arcclk: float = 0.0
    status: str = "unknown"
    firmware: str = ""
    serial: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class VLLMMetrics:
    timestamp: float
    requests_running: float = 0.0
    requests_waiting: float = 0.0
    gpu_cache_usage: float = 0.0
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    avg_ttft: float = 0.0  # Average time to first token
    avg_tpot: float = 0.0  # Average time per output token
    avg_e2e_latency: float = 0.0  # Average end-to-end latency
    model_name: str = ""


# ============== Database Functions ==============

def init_db():
    """Initialize SQLite database with required tables."""
    os.makedirs(DATA_DIR, exist_ok=True)

    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS device_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                device_id TEXT NOT NULL,
                board_type TEXT,
                temperature REAL,
                power REAL,
                voltage REAL,
                current REAL,
                aiclk REAL,
                arcclk REAL
            );

            CREATE TABLE IF NOT EXISTS vllm_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                model_name TEXT,
                requests_running REAL,
                requests_waiting REAL,
                gpu_cache_usage REAL,
                prompt_tokens_total REAL,
                generation_tokens_total REAL,
                avg_ttft REAL,
                avg_tpot REAL,
                avg_e2e_latency REAL
            );

            CREATE INDEX IF NOT EXISTS idx_device_metrics_timestamp ON device_metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_device_metrics_device ON device_metrics(device_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_vllm_metrics_timestamp ON vllm_metrics(timestamp);
        """)

        # Cleanup old data (keep 1 week)
        cutoff = time.time() - TIME_RANGES["1w"]
        conn.execute("DELETE FROM device_metrics WHERE timestamp < ?", (cutoff,))
        conn.execute("DELETE FROM vllm_metrics WHERE timestamp < ?", (cutoff,))
        conn.commit()


@contextmanager
def get_db():
    """Get database connection context manager."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def store_device_metrics(devices: list[DeviceInfo]):
    """Store device metrics in database."""
    if not devices:
        return

    with get_db() as conn:
        conn.executemany(
            """INSERT INTO device_metrics
               (timestamp, device_id, board_type, temperature, power, voltage, current, aiclk, arcclk)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(d.timestamp, d.device_id, d.board_type, d.temperature, d.power,
              d.voltage, d.current, d.aiclk, d.arcclk) for d in devices]
        )
        conn.commit()


def store_vllm_metrics(metrics: VLLMMetrics):
    """Store vLLM metrics in database."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO vllm_metrics
               (timestamp, model_name, requests_running, requests_waiting, gpu_cache_usage,
                prompt_tokens_total, generation_tokens_total, avg_ttft, avg_tpot, avg_e2e_latency)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (metrics.timestamp, metrics.model_name, metrics.requests_running,
             metrics.requests_waiting, metrics.gpu_cache_usage, metrics.prompt_tokens_total,
             metrics.generation_tokens_total, metrics.avg_ttft, metrics.avg_tpot,
             metrics.avg_e2e_latency)
        )
        conn.commit()


def get_device_history(device_id: str, time_range: str = "1h") -> list[dict]:
    """Get device metrics history from database."""
    seconds = TIME_RANGES.get(time_range, TIME_RANGES["1h"])
    cutoff = time.time() - seconds

    # Determine sampling interval based on time range
    if seconds <= 600:  # 10 min
        sample_interval = 5  # Every 5 seconds
    elif seconds <= 3600:  # 1 hour
        sample_interval = 15  # Every 15 seconds
    elif seconds <= 21600:  # 6 hours
        sample_interval = 60  # Every minute
    elif seconds <= 86400:  # 24 hours
        sample_interval = 300  # Every 5 minutes
    else:  # 1 week
        sample_interval = 1800  # Every 30 minutes

    with get_db() as conn:
        rows = conn.execute(
            """SELECT timestamp, temperature, power, voltage, current, aiclk, arcclk
               FROM device_metrics
               WHERE device_id = ? AND timestamp > ?
               AND CAST(timestamp / ? AS INTEGER) * ? = timestamp / ? * ?
               ORDER BY timestamp""",
            (device_id, cutoff, sample_interval, sample_interval, sample_interval, sample_interval)
        ).fetchall()

        # Fallback: just get all data and sample in Python
        if not rows:
            rows = conn.execute(
                """SELECT timestamp, temperature, power, voltage, current, aiclk, arcclk
                   FROM device_metrics
                   WHERE device_id = ? AND timestamp > ?
                   ORDER BY timestamp""",
                (device_id, cutoff)
            ).fetchall()

    # Sample the data
    result = []
    last_ts = 0
    for row in rows:
        if row["timestamp"] - last_ts >= sample_interval:
            result.append(dict(row))
            last_ts = row["timestamp"]

    return result


def get_vllm_history(time_range: str = "1h") -> list[dict]:
    """Get vLLM metrics history from database."""
    seconds = TIME_RANGES.get(time_range, TIME_RANGES["1h"])
    cutoff = time.time() - seconds

    # Determine sampling interval
    if seconds <= 600:
        sample_interval = 5
    elif seconds <= 3600:
        sample_interval = 15
    elif seconds <= 21600:
        sample_interval = 60
    elif seconds <= 86400:
        sample_interval = 300
    else:
        sample_interval = 1800

    with get_db() as conn:
        rows = conn.execute(
            """SELECT timestamp, model_name, requests_running, requests_waiting,
                      gpu_cache_usage, prompt_tokens_total, generation_tokens_total,
                      avg_ttft, avg_tpot, avg_e2e_latency
               FROM vllm_metrics
               WHERE timestamp > ?
               ORDER BY timestamp""",
            (cutoff,)
        ).fetchall()

    # Sample the data
    result = []
    last_ts = 0
    for row in rows:
        if row["timestamp"] - last_ts >= sample_interval:
            result.append(dict(row))
            last_ts = row["timestamp"]

    return result


# ============== Sysfs Functions ==============

def read_sysfs_value(path: str, default: str = "") -> str:
    """Read a value from sysfs."""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError, IOError):
        return default


def read_sysfs_int(path: str, default: int = 0) -> int:
    """Read an integer from sysfs."""
    val = read_sysfs_value(path)
    try:
        return int(val) if val else default
    except ValueError:
        return default


def read_sysfs_float(path: str, default: float = 0.0) -> float:
    """Read a float from sysfs."""
    val = read_sysfs_value(path)
    try:
        return float(val) if val else default
    except ValueError:
        return default


def find_hwmon_for_pci_device(pci_bdf: str) -> Optional[str]:
    """Find hwmon path for a PCI device."""
    for hwmon_path in glob.glob(f"{SYSFS_HWMON_CLASS}/hwmon*"):
        try:
            real_path = os.path.realpath(hwmon_path)
            if pci_bdf in real_path:
                return hwmon_path
        except (OSError, IOError):
            continue
    return None


def get_mock_devices() -> list[DeviceInfo]:
    """Generate mock device data."""
    import random
    ts = time.time()
    return [
        DeviceInfo(
            device_id=f"device_{i}",
            board_type="n300" if i < 2 else "n150",
            pci_index=i,
            pci_bdf=f"0000:0{i}:00.0",
            temperature=45.0 + random.uniform(-5, 15),
            power=75.0 + random.uniform(-10, 30),
            voltage=0.85 + random.uniform(-0.05, 0.05),
            current=85.0 + random.uniform(-10, 20),
            aiclk=1000 + random.uniform(-50, 50),
            arcclk=500 + random.uniform(-25, 25),
            status="online",
            firmware="v1.2.3",
            serial="MOCK123456",
            timestamp=ts,
        )
        for i in range(4)
    ]


def collect_device_metrics_sysfs() -> list[DeviceInfo]:
    """Collect device metrics from sysfs/hwmon."""
    devices = []
    ts = time.time()

    if not os.path.exists(SYSFS_TT_CLASS):
        return devices

    for tt_device in sorted(glob.glob(f"{SYSFS_TT_CLASS}/tenstorrent!*")):
        try:
            device_name = os.path.basename(tt_device)
            device_idx = int(device_name.split("!")[-1])

            device_link = os.path.join(tt_device, "device")
            pci_bdf = ""
            if os.path.islink(device_link):
                pci_path = os.path.realpath(device_link)
                pci_bdf = os.path.basename(pci_path)

            board_type = read_sysfs_value(os.path.join(tt_device, "tt_card_type"), "unknown")
            aiclk = read_sysfs_float(os.path.join(tt_device, "tt_aiclk"))
            arcclk = read_sysfs_float(os.path.join(tt_device, "tt_arcclk"))
            firmware = read_sysfs_value(os.path.join(tt_device, "tt_arc_fw_ver"))
            serial = read_sysfs_value(os.path.join(tt_device, "tt_serial"))

            hwmon_path = find_hwmon_for_pci_device(pci_bdf)
            temperature = power = voltage = current = 0.0

            if hwmon_path:
                temperature = read_sysfs_int(os.path.join(hwmon_path, "temp1_input")) / 1000.0
                power = read_sysfs_int(os.path.join(hwmon_path, "power1_input")) / 1_000_000.0
                voltage = read_sysfs_int(os.path.join(hwmon_path, "in0_input")) / 1000.0
                current = read_sysfs_int(os.path.join(hwmon_path, "curr1_input")) / 1000.0

            devices.append(DeviceInfo(
                device_id=f"device_{device_idx}",
                board_type=board_type,
                pci_index=device_idx,
                pci_bdf=pci_bdf,
                temperature=temperature,
                power=power,
                voltage=voltage,
                current=current,
                aiclk=aiclk,
                arcclk=arcclk,
                status="online" if board_type != "unknown" else "error",
                firmware=firmware,
                serial=serial,
                timestamp=ts,
            ))

        except Exception as e:
            print(f"Error reading device {tt_device}: {e}")
            SCRAPE_ERRORS.inc()

    return devices


def collect_device_metrics() -> list[DeviceInfo]:
    """Collect device metrics."""
    if MOCK_MODE:
        return get_mock_devices()
    return collect_device_metrics_sysfs()


# ============== vLLM Metrics Functions ==============

def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus text format into dict of metric values."""
    metrics = {}
    for line in text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        # Match metric_name{labels} value or metric_name value
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([0-9.eE+-]+|NaN|Inf|-Inf)$', line)
        if match:
            name, value = match.groups()
            try:
                metrics[name] = float(value) if value not in ('NaN', 'Inf', '-Inf') else 0.0
            except ValueError:
                pass
    return metrics


def get_histogram_avg(metrics: dict, base_name: str) -> float:
    """Calculate average from histogram sum and count."""
    sum_key = f"{base_name}_sum"
    count_key = f"{base_name}_count"

    # Find keys with any labels
    sum_val = count_val = 0.0
    for key, val in metrics.items():
        if sum_key in key and '{' in key:
            sum_val = val
        elif count_key in key and '{' in key:
            count_val = val

    return sum_val / count_val if count_val > 0 else 0.0


async def collect_vllm_metrics() -> Optional[VLLMMetrics]:
    """Collect metrics from vLLM's Prometheus endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(VLLM_METRICS_URL)
            if response.status_code != 200:
                return None

            metrics = parse_prometheus_metrics(response.text)

            # Extract key metrics
            requests_running = 0.0
            requests_waiting = 0.0
            gpu_cache_usage = 0.0
            prompt_tokens = 0.0
            generation_tokens = 0.0
            model_name = ""

            for key, val in metrics.items():
                if "num_requests_running" in key:
                    requests_running = val
                    # Extract model name from labels
                    match = re.search(r'model_name="([^"]+)"', key)
                    if match:
                        model_name = match.group(1)
                elif "num_requests_waiting" in key:
                    requests_waiting = val
                elif "gpu_cache_usage_perc" in key:
                    gpu_cache_usage = val * 100  # Convert to percentage
                elif "prompt_tokens_total" in key and "bucket" not in key:
                    prompt_tokens = val
                elif "generation_tokens_total" in key and "bucket" not in key:
                    generation_tokens = val

            avg_ttft = get_histogram_avg(metrics, "vllm:time_to_first_token_seconds")
            avg_tpot = get_histogram_avg(metrics, "vllm:time_per_output_token_seconds")
            avg_e2e = get_histogram_avg(metrics, "vllm:e2e_request_latency_seconds")

            return VLLMMetrics(
                timestamp=time.time(),
                requests_running=requests_running,
                requests_waiting=requests_waiting,
                gpu_cache_usage=gpu_cache_usage,
                prompt_tokens_total=prompt_tokens,
                generation_tokens_total=generation_tokens,
                avg_ttft=avg_ttft,
                avg_tpot=avg_tpot,
                avg_e2e_latency=avg_e2e,
                model_name=model_name,
            )
    except Exception as e:
        print(f"Error collecting vLLM metrics: {e}")
        return None


# ============== Background Tasks ==============

async def periodic_collection():
    """Background task to collect all metrics every 5 seconds."""
    while True:
        try:
            # Collect device metrics
            devices = await asyncio.to_thread(collect_device_metrics)
            if devices:
                update_prometheus_metrics(devices)
                await asyncio.to_thread(store_device_metrics, devices)

            # Collect vLLM metrics
            vllm_metrics = await collect_vllm_metrics()
            if vllm_metrics:
                await asyncio.to_thread(store_vllm_metrics, vllm_metrics)

        except Exception as e:
            print(f"Error in periodic collection: {e}")

        await asyncio.sleep(5)


def update_prometheus_metrics(devices: list[DeviceInfo]):
    """Update Prometheus metrics."""
    DEVICES_TOTAL.set(len(devices))
    for dev in devices:
        labels = {"device": dev.device_id, "board": dev.board_type}
        DEVICE_TEMP.labels(**labels).set(dev.temperature)
        DEVICE_POWER.labels(**labels).set(dev.power)
        DEVICE_VOLTAGE.labels(**labels).set(dev.voltage)
        DEVICE_CURRENT.labels(**labels).set(dev.current)
        DEVICE_AICLK.labels(**labels).set(dev.aiclk)
        DEVICE_ARCCLK.labels(**labels).set(dev.arcclk)


# ============== API Endpoints ==============

@app.on_event("startup")
async def startup():
    """Initialize database and start background collection."""
    init_db()
    asyncio.create_task(periodic_collection())


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/devices")
async def get_devices():
    """Get current device status."""
    devices = collect_device_metrics()
    return {
        "timestamp": datetime.now().isoformat(),
        "device_count": len(devices),
        "devices": [
            {
                "id": d.device_id,
                "board_type": d.board_type,
                "pci_index": d.pci_index,
                "pci_bdf": d.pci_bdf,
                "status": d.status,
                "temperature": d.temperature,
                "power": d.power,
                "voltage": d.voltage,
                "current": d.current,
                "aiclk": d.aiclk,
                "arcclk": d.arcclk,
                "firmware": d.firmware,
                "serial": d.serial,
            }
            for d in devices
        ],
        "totals": {
            "power": sum(d.power for d in devices),
            "avg_temperature": sum(d.temperature for d in devices) / len(devices) if devices else 0,
        }
    }


@app.get("/api/vllm")
async def get_vllm():
    """Get current vLLM metrics."""
    metrics = await collect_vllm_metrics()
    if not metrics:
        return {"error": "Unable to fetch vLLM metrics", "available": False}

    return {
        "timestamp": datetime.now().isoformat(),
        "available": True,
        "model_name": metrics.model_name,
        "requests_running": metrics.requests_running,
        "requests_waiting": metrics.requests_waiting,
        "gpu_cache_usage_percent": metrics.gpu_cache_usage,
        "prompt_tokens_total": metrics.prompt_tokens_total,
        "generation_tokens_total": metrics.generation_tokens_total,
        "avg_time_to_first_token_ms": metrics.avg_ttft * 1000,
        "avg_time_per_output_token_ms": metrics.avg_tpot * 1000,
        "avg_e2e_latency_s": metrics.avg_e2e_latency,
    }


@app.get("/api/history/devices/{device_id}")
async def api_device_history(device_id: str, range: str = Query("1h", regex="^(10m|1h|6h|24h|1w)$")):
    """Get device metrics history."""
    history = await asyncio.to_thread(get_device_history, device_id, range)
    return {
        "device_id": device_id,
        "time_range": range,
        "data_points": len(history),
        "history": history,
    }


@app.get("/api/history/devices")
async def api_all_device_history(range: str = Query("1h", regex="^(10m|1h|6h|24h|1w)$")):
    """Get history for all devices."""
    devices = collect_device_metrics()
    result = {}
    for d in devices:
        result[d.device_id] = await asyncio.to_thread(get_device_history, d.device_id, range)
    return {
        "time_range": range,
        "devices": result,
    }


@app.get("/api/history/vllm")
async def api_vllm_history(range: str = Query("1h", regex="^(10m|1h|6h|24h|1w)$")):
    """Get vLLM metrics history."""
    history = await asyncio.to_thread(get_vllm_history, range)
    return {
        "time_range": range,
        "data_points": len(history),
        "history": history,
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "mock_mode": MOCK_MODE, "db_path": DB_PATH}


# ============== Static Files ==============

static_path = os.environ.get("FRONTEND_PATH", "/app/frontend/dist")
if os.path.exists(static_path):
    app.mount("/assets", StaticFiles(directory=f"{static_path}/assets"), name="assets")

    @app.get("/")
    async def serve_root():
        return FileResponse(f"{static_path}/index.html")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = f"{static_path}/{path}"
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(f"{static_path}/index.html")
