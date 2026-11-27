"""Run manifest: the machine, libraries and code version behind every result file."""
import hashlib, os, platform, subprocess, datetime


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def _cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _ram_gb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemTotal'):
                return round(int(line.split()[1]) / 1e6, 1)
    except Exception:
        return None


def manifest(backend):
    here = os.path.dirname(os.path.abspath(__file__))
    libs = {}
    for name in ('numpy', 'scipy', 'qiskit', 'qiskit_aer', 'cupy'):
        try:
            libs[name] = __import__(name).__version__
        except Exception:
            libs[name] = None
    return dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        backend=backend, python=platform.python_version(), libraries=libs,
        system=platform.system(), machine=platform.machine(), cpu=_cpu_model(),
        cpu_count=os.cpu_count(), ram_gb=_ram_gb(),
        gpu=_run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']),
        git_commit=_run(['git', '-C', here, 'rev-parse', 'HEAD']),
        v10_sha256=hashlib.sha256(open(os.path.join(here, 'v10.py'), 'rb').read()).hexdigest())
