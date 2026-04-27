from __future__ import annotations

from dataclasses import dataclass
import platform
import re
import shutil
import subprocess
import threading

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass(frozen=True)
class SystemStats:
    cpu_percent: float | None
    ram_percent: float | None
    gpu_percent: float | None


class SystemMonitor(QObject):
    stats_updated = Signal(object)

    def __init__(self, interval_ms: int = 1000, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._sample)
        self._tick = 0
        self._cached_gpu_percent = None
        self._gpu_probe_running = False

    def start(self):
        self._sample()
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def _sample(self):
        self._tick += 1
        cpu_percent, ram_percent = self._sample_cpu_ram()

        # powermetrics can be slow and may require privileges, so probe in the
        # background and keep the latest known value for regular UI updates.
        if (self._tick == 1 or self._tick % 5 == 0) and not self._gpu_probe_running:
            self._gpu_probe_running = True
            threading.Thread(target=self._probe_gpu, daemon=True).start()

        self.stats_updated.emit(
            SystemStats(
                cpu_percent=cpu_percent,
                ram_percent=ram_percent,
                gpu_percent=self._cached_gpu_percent,
            )
        )

    def _probe_gpu(self):
        try:
            self._cached_gpu_percent = self._sample_apple_gpu()
        finally:
            self._gpu_probe_running = False

    @staticmethod
    def _sample_cpu_ram():
        try:
            import psutil
        except ImportError:
            return None, None

        return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent

    @staticmethod
    def _sample_apple_gpu():
        if platform.system() != "Darwin" or shutil.which("powermetrics") is None:
            return None

        try:
            result = subprocess.run(
                [
                    "powermetrics",
                    "--samplers",
                    "gpu_power",
                    "-i",
                    "200",
                    "-n",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0:
            return None

        patterns = [
            r"GPU(?:\s+HW)?\s+active\s+residency:\s*([\d.]+)%",
            r"GPU\s+active\s+residency:\s*([\d.]+)%",
            r"GPU\s+Busy:\s*([\d.]+)%",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return max(0.0, min(100.0, float(match.group(1))))

        return None
