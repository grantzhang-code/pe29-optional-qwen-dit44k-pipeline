"""Run an independently resumable partition, then preserve its reproduction inputs."""
from pathlib import Path
import argparse, fcntl, json, subprocess, sys

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--node', required=True)
ap.add_argument('--canary', action='store_true')
args = ap.parse_args()
lock = open(root / 'driver.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
command = [sys.executable, '-u', str(root / 'worker.py'), '--node', args.node]
if args.canary:
    command.append('--canary')
subprocess.run(command, check=True)
if not args.canary:
    subprocess.run([sys.executable, '-u', str(root / 'archive_inputs.py'), '--node', args.node], check=True)
