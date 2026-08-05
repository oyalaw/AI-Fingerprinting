# AI Fingerprinting Research Testbed

One codebase that generates real, labeled AI-workload network traffic for
encrypted-traffic fingerprinting research. It walks Paradigm -> Role ->
Device -> Framework -> Family -> Architecture -> Application -> Dataset ->
Transport (via config file or an interactive prompt), executes the real
workload, captures the resulting traffic with scapy, and auto-saves ground
truth labels for a 5-level classification taxonomy (Framework / Family /
Architecture / Application / Device).

Every option from the project's framework/architecture/dataset/device
tables is registered and visible via `--list`, even before it has a real
implementation behind it -- unimplemented entries are clearly marked
`[stub]` and raise a clear error if you try to run them. Currently
implemented end-to-end:

- **Inference**: PyTorch -> CNN -> ResNet18 -> Image Classification -> CIFAR10
- **Inference**: PyTorch -> CNN -> MobileNetV2 / ResNet50 -> Image Classification -> CIFAR10, same pattern as ResNet18 (random init, no pretrained download). Verified directly: load/predict/serialize/deserialize for both, plus cross-checked that MobileNetV2 also runs correctly through the OpenVINO adapter unchanged -- confirming these unlock real use across every already-implemented framework's generic PyTorch-conversion path, not just PyTorch itself.
- **Inference**: PyTorch -> Transformer -> ViT -> Image Classification -> CIFAR10/Synthetic, the first Transformer-family architecture; also reuses the existing Image Classification pipeline (`torchvision.models.vit_b_16` with an explicit `image_size=32` override -- its default 224x224 positional embeddings don't fit CIFAR10's 32x32 images otherwise). Verified end-to-end through the PyTorch adapter. Explicitly does NOT carry OpenVINO support the way MobileNetV2/ResNet50 do: tried it directly and OpenVINO's conversion fails on ViT specifically, due to a trace-consistency false positive on Dropout's internal RNG bookkeeping (shifted temporary-variable IDs between two internal retrace passes, not an actual numerical difference) -- see architectures/vit.py's docstring for the full detail and OpenVINO's own suggested workaround, untested here.
- **Inference**: PyTorch -> Transformer -> BERT -> Sentiment Analysis -> IMDB, the first NLP vertical slice (small random-init `BertConfig`, only the tokenizer's fixed vocabulary is downloaded, not model weights). This project's wire protocol is one-tensor-in/one-tensor-out, but BERT needs three input tensors -- `applications/sentiment_analysis.py` stacks input_ids/token_type_ids/attention_mask into a single `(3, 32)` tensor, and a small wrapper in `architectures/bert.py` unstacks them again before the real forward pass, mirroring how YOLOv8's wrapper unwraps its output tuple, just on the input side. `datasets/imdb.py` downloads and parses the real Stanford IMDB archive with the standard library rather than HuggingFace's `datasets` pip package: that package's own top-level import name collides with this project's own `datasets/` package (confirmed directly), so it can't be used from inside this project at all. Verified end-to-end including a real client/server roundtrip with correct ground truth.
- **Inference**: PyTorch -> Transformer -> DistilBERT -> Sentiment Analysis -> IMDB, reusing that exact same pipeline unchanged. The one real difference, confirmed directly: DistilBERT's `forward()` takes only `input_ids`/`attention_mask`, no `token_type_ids` (no next-sentence-prediction pretraining objective, so no segment embeddings) -- its wrapper in `architectures/distilbert.py` just uses two of the three stacked rows `applications/sentiment_analysis.py` produces and ignores the third. Verified end-to-end through the PyTorch adapter with real downloaded IMDB text.
- **Inference**: PyTorch -> RNN -> LSTM / GRU -> Activity Recognition -> UCI HAR, the first RNN-family vertical slice and the first non-vision/non-text sensor-data one. `datasets/uci_har.py` downloads and parses the real UCI HAR archive directly with the standard library (a nested zip, confirmed by inspection: 9 "Inertial Signals" channel files, 128-timestep windows, 6 activity classes) rather than assuming its format. `families/rnn`'s shared helper transposes UCI HAR's natural (channels, timesteps) layout to PyTorch's (timesteps, channels) `batch_first=True` convention, reused by both architectures unchanged. Verified end-to-end for both LSTM and GRU through the PyTorch adapter with real downloaded sensor data, plus a real client/server roundtrip (LSTM) with correct ground truth.
- **Inference**: PyTorch -> Diffusion -> DDPM -> Image Generation -> Synthetic, the first Diffusion-family vertical slice. Unlike every other architecture here, "inference" is a T-step reverse-diffusion loop (T=20, not the original paper's 1000 -- a real, common practical tradeoff), but that loop lives entirely inside the model's own `forward()`, invisible to `frameworks/pytorch_adapter.py`: one `predict()` call in, one generated image out, over the same wire protocol as everything else -- diffusion *sampling* (as opposed to training) genuinely is request/response shaped. `applications/image_generation.py`'s `preprocess` generates fresh Gaussian noise itself rather than using the dataset sample's actual pixel content; `Synthetic` just drives the request count. Verified end-to-end: confirmed no NaN/Inf across the full 20-step reverse process, plus a real client/server roundtrip with correct ground truth. **`diffusion_frameworks/diffusers_adapter.py`** now provides a second, real, verified DDPM sampler built with HuggingFace's real `UNet2DModel` + `DDPMScheduler` instead of the hand-rolled noise predictor/formula -- surfaced and fixed a real bug along the way (`UNet2DModel`'s default `norm_num_groups=32` doesn't divide this project's deliberately small 16/32-channel blocks; fixed with an explicit `norm_num_groups=8`), then confirmed the same T=20-step loop produces a correctly-shaped, NaN/Inf-free output. Same real dispatch as `graph_frameworks/pytorch_geometric_adapter.py` (see the GCN/PyTorch Geometric bullet above for how the threading works): `architectures/ddpm.py`'s `build()` checks `config.diffusion_framework` and dispatches to `DiffusersAdapter.build_ddpm()` when it's set to `Diffusers` -- verified directly (`type(model).__name__` is `_DiffusersDDPMSampler`, not the default `_DDPMSampler`).
- **Inference**: PyTorch -> GNN -> GCN -> Node Classification -> Karate Club, the first GNN-family vertical slice (this bullet was missing from this list until now, despite the slice existing -- fixed alongside the PyTorch Geometric entry below). `datasets/karate_club.py` loads the real Zachary's Karate Club graph via `networkx.karate_club_graph()`. GCN is transductive -- it classifies every node in one forward pass over the whole graph, so unlike every other dataset here `samples(n)` yields the same graph-wide input n times rather than n independent examples, a legitimate modeling choice (and a realistic serving pattern: repeated inference against a static graph). `architectures/gcn.py` hand-rolls the Kipf & Welling propagation (`adjacency @ (features @ W)`) directly in PyTorch rather than depending on torch_geometric/DGL. Verified end-to-end through the PyTorch adapter with the real graph, plus a real client/server roundtrip with correct ground truth. **`graph_frameworks/pytorch_geometric_adapter.py`** now provides a second, real, verified GCN built with `torch_geometric.nn.GCNConv` instead -- confirmed to require `normalize=False`/`add_self_loops=False` with explicit edge weights recovered via `dense_to_sparse`, since Karate Club's adjacency channel is already fully normalized and `GCNConv` normalizes by default (would otherwise double-normalize). Produces the same (34, 2) output shape as the hand-rolled version. **Sub-framework dispatch is now real**: `ExperimentConfig` is threaded through `FrameworkAdapter.load_model()` and every architecture's `build()` (all 22 framework adapters, all 13 architectures), and `architectures/gcn.py`'s `build()` checks `config.graph_framework`, dispatching to `PyTorchGeometricAdapter.build_gcn()` when it's set to `PyTorch Geometric` -- verified directly (`type(model).__name__` is `_PyGGCNClassifier`, not the default `_GCNClassifier`) plus a real client/server roundtrip with `graph_framework: PyTorch Geometric` correctly recorded in ground truth.
- **Inference**: PyTorch -> Autoencoder -> Autoencoder -> Image Reconstruction -> CIFAR10, completing the family axis (6/6). Both the architecture and application are new additions, not existing stubs: the original taxonomy listed 6 families but only paired 5 of them with an architecture, and none of the original 8 applications fit an autoencoder's actual task -- same kind of well-justified gap-fill as Karate Club/Node Classification was for GNN. Small conv encoder/decoder with a Sigmoid decoder output; confirmed directly before writing it that the reconstructed output shape exactly matches the (3, 32, 32) input and stays correctly bounded in [0, 1]. Uses its own `families/autoencoder` [0,1] pixel scaling rather than `families/cnn`'s ImageNet normalization, since the Sigmoid output needs a [0,1] target. Verified end-to-end through the PyTorch adapter with real CIFAR10 images, plus a real client/server roundtrip with correct ground truth.
- **Inference**: OpenVINO -> CNN -> ResNet18, same combo -- converts the PyTorch module straight to OpenVINO IR and compiles it for `CPU` by default. Runs on any x86/ARM machine, no special hardware needed (this one's verified end-to-end, including a real client/server roundtrip).
- **Inference (Jetson only)**: TensorRT -> CNN -> ResNet18, same combo -- exports the PyTorch module to ONNX and compiles a TensorRT engine. Needs an actual NVIDIA GPU (`tensorrt` + `pycuda`); validates and lists fine without them, it just can't build/run an engine on non-NVIDIA hardware.
- **Inference**: TensorFlow Lite -> CNN -> ResNet18, same combo -- converts the PyTorch module to a `.tflite` flatbuffer via `litert-torch` (Google's PyTorch->TFLite converter), then runs it through the standard TFLite interpreter API. Needs `litert-torch` (pulls in full `tensorflow`) plus `ai-edge-litert` for inference; both are normal pip installs but don't yet have wheels for every Python version -- validates and lists fine either way, conversion just needs a Python version `tensorflow` supports.
- **Inference**: ONNX Runtime -> CNN -> ResNet18, same combo -- exports the PyTorch module to ONNX (same export step TensorRT uses) and runs it through `onnxruntime.InferenceSession`. Pure Python/C++ wheel, no special hardware needed (this one's verified end-to-end too, including a real client/server roundtrip).
- **Inference**: ONNX Runtime Mobile -> CNN -> ResNet18, same combo -- runs the same ONNX export through ORT's real mobile model-prep pipeline (`onnxruntime.tools.convert_onnx_models_to_ort`, the API behind the official conversion CLI) to produce the same size/latency-optimized `.ort` flatbuffer format a real Android/iOS app would ship, then loads and runs it via `InferenceSession` on `CPUExecutionProvider` as a stand-in for the on-device NNAPI/CoreML execution provider a real mobile build would pick. Verified end-to-end here, including a real client/server roundtrip -- actual on-device NNAPI/CoreML execution needs a native Android/iOS app, out of scope for the same reason noted under Devices below.
- **Inference**: PyTorch Mobile -> CNN -> ResNet18, same combo -- `torch.jit.trace`s the module, runs Meta's real `optimize_for_mobile` pass, saves via `_save_for_lite_interpreter` (the actual `.ptl` mobile format), then loads it back with `_load_for_lite_interpreter` and runs inference on that loaded module -- no extra dependency at all, it's all in `torch`. Verified end-to-end, including a real client/server roundtrip; on a PyTorch build without XNNPACK (e.g. this project's own dev machine), `optimize_for_mobile` isn't available and the adapter falls back to saving the unoptimized traced module with a warning, rather than failing.
- **Inference (not execution-verified here)**: ExecuTorch -> CNN -> ResNet18, same combo -- Meta's newer PyTorch Mobile successor. `torch.export.export`s the module, lowers it with `executorch.exir.to_edge`, produces real `.pte` program bytes via `.to_executorch()`, then loads and runs that `.pte` through ExecuTorch's own documented host runtime (`_load_for_executorch`). Unlike every other adapter above, this one's blocked by dependency availability rather than hardware: `executorch` currently has no PyPI wheel for this project's Python version, so it validates and lists fine but hasn't actually been run -- treat frameworks/executorch_adapter.py as written-to-the-docs rather than verified, and re-check it against current ExecuTorch docs before relying on it.
- **Inference (not execution-verified here)**: TVM -> CNN -> ResNet18, same combo -- Apache's compiler-stack framework. `torch.jit.trace`s the module, imports it into TVM's Relay IR via `relay.frontend.from_pytorch`, compiles for `target="llvm"` (generic CPU codegen), and runs it through `tvm.contrib.graph_executor`. Also blocked by dependency availability, not hardware: `apache-tvm` has no prebuilt wheel here and tries to build its `apache-tvm-ffi` component from source via CMake, which fails with no C/C++ compiler toolchain configured on this machine -- installing one just to build this package was judged too invasive to do unprompted. Validates and lists fine; treat frameworks/tvm_adapter.py the same as executorch_adapter.py -- written-to-the-docs, not verified, and note TVM has been mid-transition from Relay IR (used here) to a newer Relax IR across recent releases.
- **Inference (conversion verified, execution is macOS/iOS-only)**: CoreML -> CNN -> ResNet18, same combo -- `torch.jit.trace`s the module and converts it via `coremltools.convert(..., convert_to="neuralnetwork")` (the older but still-supported CoreML format; the modern `"mlprogram"` default needs a native blob-writer extension coremltools only ships for macOS, confirmed here: it fails even to *convert* with `RuntimeError: BlobWriter not loaded`). Conversion verified end-to-end on this Windows machine; running the converted model was then tried and confirmed to fail with coremltools' own real error, `Model prediction is only supported on macOS version 10.13 or later` -- not a gap in this adapter, structurally the same situation as TensorRT needing an NVIDIA GPU.
- **Inference (Apple Silicon only)**: MPS Backend -> CNN -> ResNet18, same combo -- Apple's Metal Performance Shaders backend for PyTorch (`torch.device("mps")`), not a separate package: unlike every converter framework above, it's PyTorch's own execution backend, so this adapter just moves the unconverted PyTorch module onto `mps` instead of `cpu`/`cuda`. `torch.backends.mps.is_available()` is checked at `.build()` time rather than letting an op fail deep inside `predict`; verified directly on this Windows machine, where `is_built()`/`is_available()` both correctly report `False` (Windows/Linux torch builds don't compile the Metal backend in), and the adapter raises a clear error immediately instead of silently falling back to CPU, which would defeat the point of a device-specific label.
- **Inference (Android only)**: NNAPI -> CNN -> ResNet18, same combo -- NNAPI has no standalone Python binding; the real documented path is ONNX Runtime's own `NNAPIExecutionProvider` (same ONNX export step the ONNX Runtime adapter uses), compiled into `onnxruntime-android`/ORT Mobile builds, not the plain `onnxruntime` wheel this project depends on. ORT's `providers=[...]` is a *priority* list -- if the first provider isn't compiled in, ORT silently falls back to the next one rather than erroring, so naively requesting NNAPI here would silently run on CPU while still labeling the traffic "NNAPI". This adapter checks `"NNAPIExecutionProvider" in onnxruntime.get_available_providers()` itself and raises a clear error instead of trusting that silent fallback -- verified directly here: it correctly reports unavailable and fails at `load_model()` before attempting anything, the same "check before silently degrading" pattern as MPS Backend.
- **Inference**: PyTorch -> CNN -> YOLOv8 -> Object Detection -> Synthetic/COCO -- the first non-ResNet18 vertical slice. `architectures/yolov8.py` loads YOLOv8 via Ultralytics itself (`ultralytics.YOLO("yolov8n.pt")`, auto-downloaded pretrained weights, ~6MB) and returns a plain `nn.Module` (a small local wrapper unwraps YOLOv8's `(detections, features)` eval-mode output tuple down to just the detections tensor), so the existing PyTorch adapter handles it completely unchanged. `applications/object_detection.py`'s postprocess runs Ultralytics' own `non_max_suppression` on the raw `(1, 84, 8400)` detection tensor -- verified directly against Ultralytics' own high-level `YOLO.predict()` pipeline on the same image (same box region/class, confirming the decode logic is correct, not a hand-rolled reimplementation). Verified end-to-end including a real client/server roundtrip. Note: `cv_framework` (e.g. `Ultralytics`) has no alternative CV framework implemented to dispatch to -- see cv_frameworks/ultralytics_adapter.py's docstring; selecting `framework=PyTorch, architecture=YOLOv8` already runs it via Ultralytics regardless of whether `cv_framework` is separately set. (Sub-framework dispatch itself is real, not decorative -- see the GCN/PyTorch Geometric and DDPM/Diffusers entries below.)
- **Inference**: PyTorch -> CNN -> YOLOv8-Seg -> Segmentation -> Synthetic, a new architecture entry beyond the original 13 (Ultralytics' segmentation checkpoint, `yolov8n-seg.pt`, is a genuinely different head/output shape from plain YOLOv8, the same well-justified gap-fill precedent as Autoencoder). Confirmed directly: eval-mode output is `((detections, mask_prototypes), features)`, detections `(1, 116, 8400)` (4 box + 80 class + 32 mask-coefficient columns per anchor) and prototypes `(1, 32, 160, 160)`. Two structurally different output tensors don't fit this project's one-tensor wire protocol, so `_YOLOv8SegWrapper` flattens and concatenates both at a fixed, shared split point (`DETECTIONS_NUMEL` etc., imported directly by `applications/segmentation.py` so the two files can't drift out of sync) rather than inventing a second stacking convention. `postprocess` runs Ultralytics' real `non_max_suppression(..., nc=80)` (the `nc` is what tells it 32 of the remaining columns are mask coefficients, not classes) followed by `process_mask` to build actual per-object masks -- cross-checked directly against Ultralytics' own `YOLO.predict()` on its bundled bus.jpg test image before writing this file: same detection count/classes/confidences, same relative mask-area proportions. Verified end-to-end through the PyTorch adapter.

Written to each framework's documented API but **not execution-verified** in this environment (same disclosure as ExecuTorch/TVM above -- every one of these was checked against real pip/wheel availability first, and landed here specifically because that check failed):

- **TensorFlow** -> CNN -> ResNet18, same combo -- ONNX export (same step TensorRT/ONNX Runtime use) then `onnx2tf.convert(...)` to a SavedModel, run via `tf.saved_model.load(...)`'s serving signature. `tensorflow` has no wheel for this project's Python version at all (confirmed).
- **TensorFlow Lite Micro** -> CNN -> ResNet18, same combo -- the same `.tflite` litert-torch conversion TensorFlow Lite uses, then simulated host-side via TFLite Micro's own `runtime.Interpreter` Python binding. No PyPI package exists at all for this one (confirmed) -- it's a from-source Bazel build against the tflite-micro GitHub repo, a stronger gap than ExecuTorch/TVM's "wheel not published yet."
- **MediaPipe** -> CNN -> ResNet18, same combo -- the same `.tflite` conversion, with TFLite Metadata (labels + tensor descriptions) attached via MediaPipe's own `MetadataWriter`, then run through `ImageClassifier.create_from_options(...)`. `mediapipe` itself installs and imports cleanly here (confirmed) -- it's the underlying litert-torch/tensorflow model-prep step that's blocked, same gap as TensorFlow Lite.
- **NCNN** -> CNN -> ResNet18, same combo -- `pnnx.export(...)` traces the module straight to NCNN's `.ncnn.param`/`.ncnn.bin` format, then runs via `ncnn.Net()` + `ncnn.Extractor`. Deliberately not run here: `pnnx.export` shells out to a compiled binary bundled in its PyPI wheel, the same shape of risk this project already had to correct once for a different framework's bundled tool -- written to the documented API, left unexecuted by explicit decision rather than dependency failure.
- **SNPE** -> CNN -> ResNet18, same combo -- ONNX export, then Qualcomm's `snpe-onnx-to-dlc`/`snpe-net-run` CLI tools (subprocess calls). Not a pip package at all -- requires manually downloading the Qualcomm Neural Processing SDK with a developer account; `load_model` raises a clear, actionable error when `SNPE_ROOT` isn't set rather than failing obscurely mid-subprocess.
- **RKNN** -> CNN -> ResNet18, same combo -- ONNX export into Rockchip's `rknn.api.RKNN` (`load_onnx`/`build`/`init_runtime()` with no `target=`, which is RKNN's own host-side x86 CPU simulator -- no real Rockchip NPU needed for that path specifically). `rknn-toolkit2` has no wheel for this platform/Python version (confirmed).
- **Arm NN** -> CNN -> ResNet18, same combo -- ONNX export into `pyarmnn.ICreateOnnxParser`, optimized for `CpuAcc`/`CpuRef` backends, run via `IRuntime.EnqueueWorkload`. `pyarmnn` isn't a normal PyPI package on x86_64 Windows -- it targets Arm platforms specifically (Android, Arm Linux, Jetson's Arm CPU).
- **DeepStream** -> CNN -> ResNet18, same combo, and the most speculative adapter in this project: DeepStream is a GStreamer video-*streaming* SDK, not a load-model/predict API, so this adapter pushes one frame through an `appsrc` -> `nvinfer` -> `appsink` pipeline and reads results back from `pyds` buffer metadata -- a real but unusual way to force a streaming pipeline into this project's synchronous per-request shape. `pyds` ships only inside the DeepStream SDK container (Linux + NVIDIA GPU), not on PyPI.
- **Edge Impulse** -> CNN -> ResNet18 -- architecturally different from the ten converters above: Edge Impulse has no local "convert my own PyTorch architecture" path at all, models are trained through Edge Impulse Studio (cloud, account required) and exported as a `.eim` file. `load_model` implements the real `edge_impulse_linux.image.ImageImpulseRunner` API faithfully but raises a clear error for ResNet18 specifically, since this project has no Edge Impulse export for it to load.
- **MNN** -> CNN -> ResNet18, same combo -- ONNX export, then the `mnnconvert` CLI (`--framework ONNX --modelFile ... --MNNModel ...`) bundled in the `mnn` PyPI package, then `MNN.Interpreter`/session-based inference. Blocked by something stronger than the dependency/hardware gaps above: directly testing `mnnconvert --help` in this environment triggered an unprompted `pip install aliyun-log-python-sdk` plus a chain of unrelated network dependencies, with no `--help` output produced at all -- not normal behavior for a --help flag under any framing. Nothing from that install completed here (a downstream `head` cut the pipe first), and `mnn` has been uninstalled from this project's environment as a precaution. Written to MNN's documented CLI/Interpreter API, but re-verify that behavior yourself and decide deliberately before ever actually running the conversion step -- see frameworks/mnn_adapter.py's docstring.

- **Federated learning**: Flower (same PyTorch/ResNet18/CIFAR10 combo, partitioned across simulated clients) -- verified end-to-end.
- **Federated learning (not execution-verified here)**: FedLab -- same combo, communicates over `torch.distributed`'s gloo/TCP rendezvous (same mechanism as the DistributedDataParallel adapter below, verified working in isolation on this machine). A real bug was found and fixed (`SyncServerHandler` was imported from the wrong module). But the full client/server pair hangs indefinitely right after connecting: an isolated test of just FedLab's own post-connect handshake shows the client's `setup()` call returns normally while the server's matching `setup()` call -- blocked on a `recv()` for that same message -- never does. Investigation points at a FedLab/PyTorch version incompatibility, but that's not a confirmed root cause; treat frameworks/fedlab_adapter.py the same as executorch_adapter.py/tvm_adapter.py, and re-run the isolated setup diagnostics in its docstring yourself before relying on it.
- **Federated learning (not execution-verified here, and wouldn't produce real network traffic even where it runs)**: NVFlare -- same combo, via NVFlare's Client API (`FedJob` + `FedAvg` controller + `ScriptRunner` running a real standalone training script). Two separate, serious caveats: (1) `import nvflare` fails outright on this Windows machine -- confirmed directly, its `__init__.py` unconditionally pulls in `nvflare.fuel.f3.cellnet.net_agent`, which does `import resource`, the POSIX-only resource-limits module that doesn't exist on Windows at all; NVFlare requires Linux/macOS. (2) Separately and more fundamentally: this adapter uses `FedJob.simulator_run()`, the simplest API match for "run a federated round from one Python call" -- but simulator mode runs the server and all clients as in-process threads over in-memory queues, not real network sockets, so even on a supported OS this configuration produces no capturable traffic at all. Real traffic needs NVFlare's separate POC/production deployment mode (`nvflare provision` + a real `start.sh` process per site), which is out of scope here. Unlike FedLab/ExecuTorch/TVM, this one wouldn't serve this project's purpose even if it ran successfully in its current form.
- **Distributed training**: PyTorch `DistributedDataParallel` (same combo, multi-process gradient sync)
- **Distributed training**: FairScale (`OSS` optimizer-state sharding + `ShardedDataParallel`, ZeRO stage 1) -- layers on the exact same gloo/TCP process group as the DDP adapter (confirmed directly from `OSS`/`ShardedDataParallel`'s real constructor signatures). The actually-new code -- wrapping a real ResNet18, running real forward/backward/step cycles over a live two-process group -- was verified end-to-end via an isolated two-rank synthetic-data script (both ranks completed cleanly, no hang). The full adapter through `main.py` with real CIFAR10 wasn't run start-to-finish in this session purely because of a slow download in this environment (~40 min, a pre-existing property of the shared `_build_loader` code this adapter reuses unchanged from ddp_adapter.py, not anything new).
- **Transport**: TCP and TLS (self-signed dev cert auto-generated), plus HTTP -- a plain `http.server`/`http.client` implementation of the same `Transport` interface, standard library only. HTTP is request/response, not the persistent bidirectional stream TCP/TLS use, so the server side bridges this project's synchronous send()/recv() calls onto it with a small queue: the HTTP handler (its own thread per request) hands each request body to the main thread via one queue and blocks on a second queue for the response bytes to write back -- correct here because exactly one request is ever in flight (this project's client always waits for the full response before sending the next). Verified end-to-end including a real client/server roundtrip with correct ground truth (`"transport": "HTTP"`).

## Setup

```bash
pip install -r requirements.txt
```

`torch`/`torchvision`/`flwr` are large downloads. If you only want to browse
the registry (`--list`) or work on new adapters, `pyyaml` + `numpy` are
enough to get started.

### Packet capture prerequisites

Capture (`capture: true` in config, on by default) uses scapy and needs a
working packet-capture driver plus elevated privileges:

- **Windows**: install [Npcap](https://npcap.com/), then run your terminal
  as Administrator.
- **Linux / macOS**: libpcap is normally preinstalled; run with `sudo`, or
  grant your Python interpreter `CAP_NET_RAW` once so you don't need root
  every time:
  `sudo setcap cap_net_raw+eip $(readlink -f $(which python3))`

Without capture privileges, set `capture: false` in the config to still run
the workload and generate ground truth (just no `.pcap`/sequence CSV).

## Running the vertical slices

**Inference** (two terminals):

```bash
python main.py --config config.yaml --role server
python main.py --config config.yaml --role client
```

Switch `transport: TCP` to `transport: TLS` in `config.yaml` to compare
plaintext vs. encrypted traffic signatures.

**Federated learning** (one server + 2+ clients, each in its own terminal;
set `paradigm: federated_learning`, `fl_framework: Flower` in the config,
and give each client a distinct `client_index`):

```bash
python main.py --config config.yaml --role server
python main.py --config config.yaml --role client --client-index 0
python main.py --config config.yaml --role client --client-index 1
```

**Distributed training** (coordinator + 1+ workers; set
`paradigm: distributed_training`, `distributed_framework: DistributedDataParallel`):

```bash
python main.py --config config.yaml --role server               # rank 0 / coordinator
python main.py --config config.yaml --role client --worker-rank 1
```

**OpenVINO, TensorRT, TensorFlow Lite, ONNX Runtime, ONNX Runtime Mobile,
PyTorch Mobile, ExecuTorch, TVM, CoreML, MPS Backend, or NNAPI inference**
(set `framework:` to `OpenVINO`, `TensorRT`, `TensorFlow Lite`,
`ONNX Runtime`, `ONNX Runtime Mobile`, `PyTorch Mobile`, `ExecuTorch`,
`TVM`, `CoreML`, `MPS Backend`, or `NNAPI` in the config, everything else
stays the same as the PyTorch inference example -- all but
TensorRT/CoreML/MPS Backend/NNAPI run on any machine; TensorRT needs an
NVIDIA/Jetson GPU, CoreML/MPS Backend need macOS (CoreML conversion works
anywhere, only execution is macOS/iOS-only), and NNAPI needs an
onnxruntime-android build; ExecuTorch and TVM additionally need a
Python/toolchain their respective packages can actually build/install
with):

```bash
python main.py --config config.yaml --role server
python main.py --config config.yaml --role client
```

## Other commands

```bash
python main.py --list          # every registry entry, implemented vs. stub
python main.py --interactive   # guided step-by-step setup (Paradigm -> ... -> Transport)
```

## Output

Each run writes to `experiments/results/<experiment_id>/`:
`ground_truth.json` (the 5-level label + full config snapshot),
`<id>.pcap` (raw capture), `<id>_sequence.csv` (ordered
timestamp/direction/size sequence -- the standard feature representation
for traffic-fingerprinting classifiers), and `events.jsonl` / `experiment.log`.

## Extending

Every axis is a plugin registry (`core/registry.py`). To promote a stub to
a real implementation (e.g. a new mobile app for Android/iPhone, or one of
the other 21 stub frameworks), replace its `raise NotImplementedError` with
a real implementation and flip `implemented=False` to `True` in its
`.register(...)` call -- nothing else needs to change; auto-discovery picks
it up. `frameworks/tensorrt_adapter.py` is a worked example of promoting a
framework whose architectures aren't natively built in it (it consumes the
same PyTorch-built architectures via ONNX export -- see each architecture's
`also_supports` registration metadata and the compatibility check in
`core/config.py`).
