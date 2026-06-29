from pathlib import Path
import subprocess
import sys

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def train_impl():
    if not CONFIG_PATH.exists():
        print(f"Missing {CONFIG_PATH}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "-c", str(CONFIG_PATH),
        "--train",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
