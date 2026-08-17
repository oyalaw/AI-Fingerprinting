"""Horovod -- deliberately left a stub after real investigation. The
original "no CMake" finding is resolved, and the predicted "next Windows-
incompatibility layer" is now confirmed concretely, three layers deep,
rather than left as a prediction. Re-investigated again on Ubuntu (the
findings below through the libuv wall are all from this project's earlier
Windows dev machine): that third wall is confirmed fully gone on Linux,
exactly as gloo's own CMakeLists.txt predicts -- and a fourth, different,
genuine wall was found underneath it, unrelated to any platform gate.

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
already more than the framework's own design targets.

On Ubuntu, confirmed directly (`cmake` via pip again, same
`CMAKE_POLICY_VERSION_MINIMUM=3.5` workaround, `--no-build-isolation`):
the libuv wall never appears at all -- inspected gloo's own vendored
`third_party/gloo/CMakeLists.txt` directly, and `USE_LIBUV`/the whole
libuv search only happens inside an `if(MSVC)` block; on a real Linux
compiler that block never executes, `USE_LIBUV_DEFAULT` stays `OFF`, and
CMake configuration completes cleanly with no libuv requirement
whatsoever. That's the platform gate fully resolved, exactly as this
project's own reading of gloo's CMake predicted.

Configuration then proceeds into an actual compile, and hits a fourth,
different, genuine wall: `error: 'string' in namespace 'std' does not
name a type` in Horovod's own `horovod/common/group_table.h`, plus the
same error cascading through `group_table.cc`. Confirmed directly by
inspecting the file: it uses `std::string`/`std::unordered_map<std::string,
...>` throughout but only `#include`s `<mutex>`, `<queue>`,
`<unordered_map>`, `<vector>` -- never `<string>`. This code (dated 2020,
NVIDIA-authored) relied on `<string>` being pulled in transitively through
one of those other standard headers, which older/looser libstdc++ builds
tolerated; this machine's GCC 13.3 (Ubuntu 24.04's default) enforces
stricter standard-library header hygiene and no longer does that transitive
include. A genuine bug in Horovod's own unmaintained source (last real
release 0.28.1, no update since), not a platform gate, not a CMake/compiler
availability issue, and not something a pip flag or environment variable
can route around -- same class of "old package, newer toolchain" wall as
distributed_frameworks/byteps_adapter.py's `include_paths(cuda=...)`
finding. Confirmed no side effects (horovod itself not left installed).
Revisit only by patching Horovod's own vendored source to add the missing
`#include <string>` (out of scope for an adapter investigation), or if
Horovod ever ships a release that fixes this upstream.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Horovod's libuv/Windows wall is gone on Linux, but its own "
        "group_table.h is missing a #include <string> that a modern GCC no "
        "longer tolerates -- see this module's docstring."
    )


DISTRIBUTED_FRAMEWORKS.register("Horovod", implemented=False, organization="Uber")(build)
