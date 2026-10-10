"""Only start the caption worker; no editor or scoring is launched."""
from pathlib import Path
import fcntl,subprocess,sys
root=Path(__file__).resolve().parent
lock=(root/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
subprocess.run([sys.executable,'-u',str(root/'worker.py'),*sys.argv[1:]],check=True)
