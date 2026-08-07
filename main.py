#!/usr/bin/env python
"""AI Fingerprinting Research Testbed.

One codebase, driven by config or an interactive prompt, that walks:
  Paradigm -> Role -> Device -> Framework -> Family -> Architecture ->
  Application -> Dataset -> Transport
executes the real workload, captures the resulting network traffic, and
auto-saves ground truth labels for downstream traffic-fingerprinting
classifiers.

Usage:
    python main.py --list                          # show every registry entry
    python main.py --interactive                   # guided step-by-step setup
    python main.py --config config.yaml --role server
    python main.py --config config.yaml --role client
"""
import argparse
import sys

from core.config import (
    FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES,
    FL_DISTRIBUTED_DATASET,
    FL_DISTRIBUTED_FRAMEWORK,
    VALID_PARADIGMS,
    VALID_ROLES,
    ExperimentConfig,
    load_config,
)
from core.experiment import Experiment
from core.registry import (
    ALL_REGISTRIES,
    APPLICATIONS,
    ARCHITECTURES,
    CV_FRAMEWORKS,
    DATASETS,
    DEVICES,
    DIFFUSION_FRAMEWORKS,
    DISTRIBUTED_FRAMEWORKS,
    FAMILIES,
    FL_FRAMEWORKS,
    FRAMEWORKS,
    GRAPH_FRAMEWORKS,
    LLM_FRAMEWORKS,
    SPEECH_FRAMEWORKS,
    TRANSPORTS,
    discover_all,
)


def cmd_list(_args):
    discover_all()
    for category, registry in ALL_REGISTRIES.items():
        entries = registry.list()
        implemented_count = sum(1 for e in entries if e.implemented)
        print(f"\n{category} ({implemented_count}/{len(entries)} implemented)")
        print("-" * 64)
        for entry in entries:
            status = "[implemented]" if entry.implemented else "[stub]       "
            meta_bits = ", ".join(f"{k}={v}" for k, v in entry.meta.items() if v not in (None, ""))
            suffix = f"  ({meta_bits})" if meta_bits else ""
            print(f"  {status} {entry.name}{suffix}")


def _prompt_from_options(label, options):
    print(f"\n{label}:")
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")
    while True:
        raw = input(f"Select {label} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid choice, try again.")


def _prompt_registry_choice(label, registry, filter_fn=None):
    entries = registry.list()
    if filter_fn:
        entries = [e for e in entries if filter_fn(e)]
    if not entries:
        raise SystemExit(f"No {label} entries match the previous selections.")
    print(f"\n{label}:")
    for i, entry in enumerate(entries, start=1):
        status = "" if entry.implemented else "  [stub]"
        print(f"  {i}. {entry.name}{status}")
    while True:
        raw = input(f"Select {label} [1-{len(entries)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(entries):
            return entries[int(raw) - 1].name
        print("Invalid choice, try again.")


def _prompt_optional_registry_choice(label, registry):
    """Same as _prompt_registry_choice, but with a leading "(none -- use the
    default)" option, for the five sub-framework fields (llm_framework etc.)
    that core/config.py's validate() treats as optional (required=False)."""
    entries = registry.list()
    print(f"\n{label} (optional):")
    print("  0. (none -- use the architecture's own default)")
    for i, entry in enumerate(entries, start=1):
        status = "" if entry.implemented else "  [stub]"
        print(f"  {i}. {entry.name}{status}")
    while True:
        raw = input(f"Select {label} [0-{len(entries)}]: ").strip()
        if raw.isdigit() and 0 <= int(raw) <= len(entries):
            choice = int(raw)
            return entries[choice - 1].name if choice > 0 else None
        print("Invalid choice, try again.")


def _architecture_compatible_frameworks(entry):
    """The set of framework names (lowercased) an architecture entry can
    run under: its own `framework` plus anything in `also_supports`. Empty
    set means "no restriction declared" -- treat as compatible with any
    framework, same convention `_architecture_matches` below already uses."""
    arch_framework = entry.meta.get("framework")
    also_supports = entry.meta.get("also_supports") or []
    compatible = {arch_framework, *also_supports} - {None}
    return {f.lower() for f in compatible}


def cmd_interactive(_args):
    discover_all()
    print("=== AI Fingerprinting Testbed: guided experiment setup ===")
    print("Every option below is real and selectable; [stub] means it's registered")
    print("but not yet implemented -- picking one will fail validation at the end.")
    print("Each step's choices are narrowed to what's actually compatible with what")
    print("you already picked (device -> framework -> family -> architecture ->")
    print("application -> dataset).\n")

    paradigm = _prompt_from_options("Paradigm", list(VALID_PARADIGMS))
    role = _prompt_from_options("Role", list(VALID_ROLES))

    locks_to_pytorch_cifar10 = paradigm in ("federated_learning", "distributed_training")
    if locks_to_pytorch_cifar10:
        print(
            f"\nNote: every {paradigm} adapter today hardcodes "
            f"{FL_DISTRIBUTED_FRAMEWORK}/{FL_DISTRIBUTED_DATASET} regardless of what's "
            "picked below (none of them read the selected dataset, and other frameworks "
            "return an object their training loop can't use) -- Framework/Architecture/"
            "Dataset are locked to what actually runs; core/config.py's validate() "
            "enforces this same constraint for hand-written config.yaml files too, see "
            "FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES's comment there for the full reasoning."
        )

    device = _prompt_registry_choice("Device", DEVICES)
    device_tags = {t.lower() for t in DEVICES.get(device).meta.get("platform_tags") or ()}

    def _framework_matches_device(entry):
        if locks_to_pytorch_cifar10:
            return entry.name.lower() == FL_DISTRIBUTED_FRAMEWORK.lower()
        platforms = entry.meta.get("platforms")
        if not platforms:
            return True  # no platform restriction declared -- assume compatible
        return bool(device_tags & {p.lower() for p in platforms})

    framework = _prompt_registry_choice("Framework", FRAMEWORKS, filter_fn=_framework_matches_device)

    fl_distributed_compatible_lower = {a.lower() for a in FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES}

    def _architecture_allowed_for_paradigm(entry):
        return not locks_to_pytorch_cifar10 or entry.name.lower() in fl_distributed_compatible_lower

    def _family_has_compatible_architecture(family_entry):
        family_name = family_entry.name.lower()
        for arch_entry in ARCHITECTURES.list():
            arch_family = (arch_entry.meta.get("family") or "").lower()
            if arch_family and arch_family != family_name:
                continue
            if not _architecture_allowed_for_paradigm(arch_entry):
                continue
            compatible = _architecture_compatible_frameworks(arch_entry)
            if not compatible or framework.lower() in compatible:
                return True
        return False

    family = _prompt_registry_choice("Family", FAMILIES, filter_fn=_family_has_compatible_architecture)

    def _architecture_matches(entry):
        arch_family = entry.meta.get("family")
        compatible = _architecture_compatible_frameworks(entry)
        family_ok = not arch_family or arch_family.lower() == family.lower()
        framework_ok = not compatible or framework.lower() in compatible
        return family_ok and framework_ok and _architecture_allowed_for_paradigm(entry)

    architecture = _prompt_registry_choice(
        "Architecture",
        ARCHITECTURES,
        filter_fn=_architecture_matches,
    )

    arch_application = ARCHITECTURES.get(architecture).meta.get("application")

    def _application_matches_architecture(entry):
        return not arch_application or entry.name.lower() == arch_application.lower()

    application = _prompt_registry_choice(
        "Application", APPLICATIONS, filter_fn=_application_matches_architecture
    )

    application_datasets = {
        d.lower() for d in (APPLICATIONS.get(application).meta.get("datasets") or ())
    }

    def _dataset_matches_application(entry):
        if locks_to_pytorch_cifar10:
            return entry.name.lower() == FL_DISTRIBUTED_DATASET.lower()
        return not application_datasets or entry.name.lower() in application_datasets

    dataset = _prompt_registry_choice("Dataset", DATASETS, filter_fn=_dataset_matches_application)
    transport = _prompt_registry_choice("Transport", TRANSPORTS)

    kwargs = dict(
        paradigm=paradigm,
        role=role,
        device=device,
        framework=framework,
        family=family,
        architecture=architecture,
        application=application,
        dataset=dataset,
        transport=transport,
    )

    # Same gating conditions core/config.py's validate() uses to decide
    # whether each sub-framework field is relevant -- only ask when it
    # would actually be checked/used, matching architectures/whisper.py's
    # (etc.) real dispatch conditions.
    application_lower = application.lower()
    family_lower = family.lower()

    if application_lower == "text generation":
        kwargs["llm_framework"] = _prompt_optional_registry_choice("LLM framework", LLM_FRAMEWORKS)
    if application_lower in ("object detection", "segmentation"):
        kwargs["cv_framework"] = _prompt_optional_registry_choice("CV framework", CV_FRAMEWORKS)
    if application_lower == "speech recognition":
        kwargs["speech_framework"] = _prompt_optional_registry_choice(
            "Speech framework", SPEECH_FRAMEWORKS
        )
    if family_lower == "gnn":
        kwargs["graph_framework"] = _prompt_optional_registry_choice("Graph framework", GRAPH_FRAMEWORKS)
    if family_lower == "diffusion":
        kwargs["diffusion_framework"] = _prompt_optional_registry_choice(
            "Diffusion framework", DIFFUSION_FRAMEWORKS
        )

    if paradigm == "federated_learning":
        kwargs["fl_framework"] = _prompt_registry_choice("FL framework", FL_FRAMEWORKS)
    if paradigm == "distributed_training":
        kwargs["distributed_framework"] = _prompt_registry_choice(
            "Distributed training framework", DISTRIBUTED_FRAMEWORKS
        )

    config = ExperimentConfig(**kwargs)
    _validate_and_run(config)


def _validate_and_run(config):
    try:
        config.validate()
    except ValueError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
    Experiment(config).run()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="AI Fingerprinting Research Testbed")
    parser.add_argument("--list", action="store_true", help="List every registry entry (implemented vs stub)")
    parser.add_argument("--interactive", action="store_true", help="Guided step-by-step experiment setup")
    parser.add_argument("--config", help="Path to an experiment config YAML")
    parser.add_argument("--role", choices=list(VALID_ROLES), help="Override role from --config")
    parser.add_argument("--client-index", type=int, help="Override client_index from --config (FL)")
    parser.add_argument("--worker-rank", type=int, help="Override worker_rank from --config (distributed training)")
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.list:
        return cmd_list(args)
    if args.interactive:
        return cmd_interactive(args)
    if args.config:
        config = load_config(args.config)
        if args.role:
            config.role = args.role
        if args.client_index is not None:
            config.client_index = args.client_index
        if args.worker_rank is not None:
            config.worker_rank = args.worker_rank
        return _validate_and_run(config)

    build_arg_parser().print_help()


if __name__ == "__main__":
    main()
