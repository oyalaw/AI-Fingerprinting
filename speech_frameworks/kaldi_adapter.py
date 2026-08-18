"""Kaldi -- deliberately left a stub, not a gap: never published as a pip
package at all, by design.

Checked directly: `pip install kaldi` returns "No matching distribution
found". Consistent with Kaldi's real nature: it's a C++ speech-processing
toolkit distributed as source (`git clone` + a from-scratch build against
system BLAS/OpenFST), predating the modern Python packaging ecosystem.
Its Python bindings (the separate `pykaldi` project) are also
source-build-only, with no real PyPI wheel -- would hit the same
missing-compiler wall as everything else in this project needing one
(now resolved on Ubuntu, see cv_frameworks/detectron2_adapter.py, but
moot here regardless).

Re-checked on Ubuntu: `pip index versions pykaldi` does list a release
(0.0.1) -- downloaded and inspected it directly rather than assuming it's
the real thing, since a plausible-looking hit is exactly the kind of
claim this project's docstrings don't take on faith. It's a 1.2KB
name-squatted placeholder ("A demo package for PyKaldi" / "Lorem ipsum
dolor sit amet" as its actual description, author unrelated to the real
pykaldi GitHub org), not the real project. The real pykaldi
(github.com/pykaldi/pykaldi) has never published to PyPI at all; this
finding doesn't change.
"""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Kaldi has no pip package -- it's a source-build-only C++ toolkit -- see "
        "this module's docstring."
    )


SPEECH_FRAMEWORKS.register("Kaldi", implemented=False, application="ASR")(build)
