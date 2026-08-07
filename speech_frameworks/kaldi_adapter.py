"""Kaldi -- deliberately left a stub, not a gap: never published as a pip
package at all, by design.

Checked directly: `pip install kaldi` returns "No matching distribution
found". Consistent with Kaldi's real nature: it's a C++ speech-processing
toolkit distributed as source (`git clone` + a from-scratch build against
system BLAS/OpenFST), predating the modern Python packaging ecosystem.
Its Python bindings (the separate `pykaldi` project) are also
source-build-only, with no PyPI wheel -- would hit the same missing-
compiler wall as everything else in this project needing one.
"""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Kaldi has no pip package -- it's a source-build-only C++ toolkit -- see "
        "this module's docstring."
    )


SPEECH_FRAMEWORKS.register("Kaldi", implemented=False, application="ASR")(build)
