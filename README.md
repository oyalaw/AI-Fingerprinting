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
- **Federated learning**: Flower (same PyTorch/ResNet18/CIFAR10 combo, partitioned across simulated clients)
- **Distributed training**: PyTorch `DistributedDataParallel` (same combo, multi-process gradient sync)
- **Transport**: TCP and TLS (self-signed dev cert auto-generated)

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
a real implementation (e.g. TensorRT, or a new mobile app for
Android/iPhone), replace its `raise NotImplementedError` with a real
implementation and flip `implemented=False` to `True` in its `.register(...)`
call -- nothing else needs to change; auto-discovery picks it up.
