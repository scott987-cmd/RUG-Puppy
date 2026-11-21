"""Install a user LaunchAgent on the home MacBook (run as its logged-in user)."""

import argparse
import os
import plistlib
import subprocess
from pathlib import Path


LABEL = "local.puppy.companion"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["install", "status", "stop"])
    parser.add_argument("--root", type=Path, default=Path.home() / "RUGPuppyGateway")
    args = parser.parse_args()
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{LABEL}"
    if args.operation == "status":
        subprocess.run(["launchctl", "print", service], check=True)
        return
    if args.operation == "stop":
        subprocess.run(["launchctl", "bootout", service], check=True)
        return
    root = args.root.expanduser().resolve()
    python = root / ".venv/bin/python"
    for required in (python, root / "companion/server.py", root / ".device-token", root / ".parent-token"):
        if not required.is_file():
            raise SystemExit(f"Missing required file: {required}")
    log_dir = root / "logs"
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = Path.home() / "Library/LaunchAgents"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{LABEL}.plist"
    data = {"Label": LABEL, "ProgramArguments": [str(python), "-m", "companion.launch", "--root", str(root)],
            "WorkingDirectory": str(root), "RunAtLoad": True, "KeepAlive": True,
            "ThrottleInterval": 15, "ExitTimeOut": 20,
            "StandardOutPath": str(log_dir / "gateway.log"),
            "StandardErrorPath": str(log_dir / "gateway-error.log"),
            "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"}}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        plistlib.dump(data, output)
    temporary.chmod(0o600)
    temporary.replace(path)
    # An existing installed service may be updated; an unrelated process on the
    # port is deliberately never killed by this installer.
    exists = subprocess.run(["launchctl", "print", service], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
    if exists:
        subprocess.run(["launchctl", "bootout", service], check=True)
    subprocess.run(["launchctl", "bootstrap", domain, str(path)], check=True)
    print(f"Installed {LABEL}; starts after this macOS user signs in")


if __name__ == "__main__":
    main()
