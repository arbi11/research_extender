"""Print what pyFEMM for femm-qt provides: run in the FEMM image by run_femm.py inventory.

One line per public function that femm.py defines, with its signature:
  FN name (signature)
plus the Python version, the path and SHA-256 of femm.py, and the engine's version.
"""
import hashlib
import inspect
import subprocess
import sys

sys.path.insert(0, "/opt/femm/share/femm/pyfemm")
import femm  # noqa: E402

print("PYTHON", sys.version.split()[0], sys.implementation.name)
print("PYFEMM", femm.__file__, hashlib.sha256(open(femm.__file__, "rb").read()).hexdigest())
engine = subprocess.run(["/opt/femm/bin/femm_headless", "--version"], capture_output=True, text=True)
print("ENGINE", engine.stdout.strip() or engine.stderr.strip())
names = sorted(name for name, value in vars(femm).items()
               if inspect.isfunction(value) and value.__module__ == "femm" and not name.startswith("_"))
for name in names:
    print("FN", name, str(inspect.signature(getattr(femm, name))))
print("COUNT", len(names))
