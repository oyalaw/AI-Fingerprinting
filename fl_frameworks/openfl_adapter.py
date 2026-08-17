"""OpenFL -- blocked by Python version, not implemented.

Checked directly: `pip install openfl` fails with "No matching
distribution found" -- every published version caps its supported Python
range at <3.13 (the newest, 1.9, requires >=3.10,<3.13). This project's
Python was 3.14 at the time, newer than any OpenFL release supports. Same
class of blocker as frameworks/executorch_adapter.py -- not a hardware/OS
gate like TensorRT/CoreML, just a real Python-version ceiling. Revisit
once OpenFL publishes a release supporting a newer Python.

Re-checked on this project's move to Ubuntu (Python 3.13.9 here, one minor
version older than the original 3.14 check): still fails identically --
`Requires-Python >=3.10,<3.13` on the newest release excludes 3.13.9 too
(it's `<3.13`, not `<=3.13`). Confirms this was correctly diagnosed as a
Python-version ceiling and not an OS-specific issue -- moving to Linux
changes nothing here.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "OpenFL has no published release supporting this Python version -- see this module's docstring."
    )


FL_FRAMEWORKS.register("OpenFL", implemented=False, organization="Intel")(build)
