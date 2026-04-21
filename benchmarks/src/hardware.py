"""
Hardware Introspection
Captures system specs to contextualize benchmark results across machines.
"""

import platform
import subprocess
import sys


def _run(cmd):
    """Run a shell command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_macos_specs():
    """macOS-specific spec collection via sysctl."""
    chip = _run(['sysctl', '-n', 'machdep.cpu.brand_string'])
    machine_id = _run(['sysctl', '-n', 'hw.model'])
    ram_bytes = _run(['sysctl', '-n', 'hw.memsize'])
    ram_gb = round(int(ram_bytes) / (1024 ** 3)) if ram_bytes and ram_bytes.isdigit() else None

    return {
        'chip': chip,
        'machine_id': machine_id,
        'ram_gb': ram_gb,
    }


def _get_ollama_version():
    """Try to get the running Ollama version."""
    output = _run(['ollama', '--version'])
    return output if output else None


def get_hardware_specs():
    """
    Collect hardware and runtime specs for the current machine.

    Returns:
        dict with platform info, chip, RAM, OS version, Python version,
        and Ollama version where available. Fields are None if detection fails.
    """
    specs = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'python_version': sys.version.split()[0],
        'ollama_version': _get_ollama_version(),
    }

    if platform.system() == 'Darwin':
        mac_version = platform.mac_ver()[0]
        specs['os_version'] = f"macOS {mac_version}" if mac_version else None
        specs.update(_get_macos_specs())
    else:
        # Fallback for non-macOS systems — values may be None
        specs['os_version'] = platform.platform()
        specs['chip'] = platform.processor() or None
        specs['machine_id'] = platform.machine() or None
        specs['ram_gb'] = None

    return specs