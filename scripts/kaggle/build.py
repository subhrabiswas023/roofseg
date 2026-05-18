import sys
from pathlib import Path

from .executor import run_command

def build_wheel() -> None:
    print(run_command(sys.executable, "-m", "build", "--wheel"))


def get_wheel_path(dist_dir: Path) -> Path:
    wheels = list(dist_dir.glob("*.whl"))

    if not wheels:
        raise RuntimeError("No wheel file was produced")

    return wheels[0]

