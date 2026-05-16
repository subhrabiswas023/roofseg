import subprocess
import sys
from pathlib import Path

def build_wheel() -> None:
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        check=True
    )
    
def get_wheel_path() -> Path:
    wheels = list(Path("dist").glob("*.whl"))
    
    if not wheels:
        raise RuntimeError("No wheel file was produced")
    
    return wheels[0]

if __name__ == "__main__":
    build_wheel()
    print(get_wheel_path())