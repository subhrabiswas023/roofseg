import subprocess
import sys
from pathlib import Path

def build_wheel() -> None:
    subprocess.run([sys.executable, "-m", "build", "--wheel"], check=True)


def get_wheel_path(dist_dir: Path) -> Path:
    wheels = list(dist_dir.glob("*.whl"))

    if not wheels:
        raise RuntimeError("No wheel file was produced")

    return wheels[0]

