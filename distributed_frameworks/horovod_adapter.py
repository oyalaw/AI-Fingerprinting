"""Horovod -- deliberately left a stub after real investigation. The
original "no CMake" finding is resolved, and the predicted "next Windows-
incompatibility layer" is now confirmed concretely, three layers deep,
rather than left as a prediction.

`pip install cmake` (a normal pip package shipping real prebuilt CMake
binaries -- confirmed directly: `cmake --version` reports a real 4.4.2)
resolves the original "no CMake" blocker. Retrying
`pip install --no-build-isolation horovod` then fails one layer deeper:
Horovod's own build script invokes cmake directly (bypassing the
`CMAKE_ARGS` env var entirely -- confirmed directly it's a hand-rolled
`build_ext.build_extensions()` that constructs the cmake command list
itself, not a standard scikit-build backend), and hits
`CMake Error ... third_party/gloo/CMakeLists.txt:1
(cmake_minimum_required): Compatibility with CMake < 3.5 has been
removed` -- Horovod's vendored `gloo` submodule is old enough to predate
modern CMake's policy floor. Worked around with the
`CMAKE_POLICY_VERSION_MINIMUM=3.5` environment variable (a real CMake
3.31+ feature built for exactly this "vendored dependency's
cmake_minimum_required is too old" situation, and unlike `CMAKE_ARGS`,
CMake itself reads this one directly regardless of what Horovod's build
script passes through).

That gets past the version-floor error entirely, and configuration
proceeds real further -- MSVC detected correctly, `USE_LIBUV ON` selected
("Only USE_LIBUV is supported on Windows", gloo's own real constraint) --
before hitting a third, different, genuine wall:
`CMake Error ... Could not find libuv_LIBRARY using the following names:
uv, libuv`. gloo's Windows transport requires the native `libuv` C
library to already be present on the system (via vcpkg, Conan, or a
manual build/install) -- not something `pip install` can provide, and
setting up a native package manager just to get one dependency's library
was judged the same kind of invasive, unprompted-scope-creep fix this
project has consistently avoided elsewhere (PySyft's full Arrow C++
build, apache-tvm's original compiler ask). Confirmed no side effects
from this attempt (torch/torchvision/numpy versions unchanged, horovod
itself never actually installed). Even past libuv, Horovod is
fundamentally MPI/NCCL-based multi-GPU tooling (confirmed directly in
this same build log: `Could NOT find MPI_CXX`, `Could NOT find NVTX`,
no CUDA compiler found) -- getting a Windows CPU-only build this far was
already more than the framework's own design targets. Revisit if libuv
becomes available in this environment for another reason, or on a Linux
machine where gloo's dependency story is simpler.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Horovod's vendored gloo dependency needs the native libuv library on "
        "Windows, not installed here (not a CMake or compiler problem anymore) -- "
        "see this module's docstring."
    )


DISTRIBUTED_FRAMEWORKS.register("Horovod", implemented=False, organization="Uber")(build)
