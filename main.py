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

from core.config import VALID_PARADIGMS, VALID_ROLES, ExperimentConfig, load_config
from core.experiment import Experiment
from core.registry import (
    ALL_REGISTRIES,
    APPLICATIONS,
    ARCHITECTURES,
    DATASETS,
    DEVICES,
    DISTRIBUTED_FRAMEWORKS,
    FAMILIES,
    FL_FRAMEWORKS,
    FRAMEWORKS,
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


def cmd_interactive(_args):
    discover_all()
    print("=== AI Fingerprinting Testbed: guided experiment setup ===")
    print("Every option below is real and selectable; [stub] means it's registered")
    print("but not yet implemented -- picking one will fail validation at the end.\n")

    paradigm = _prompt_from_options("Paradigm", list(VALID_PARADIGMS))
    role = _prompt_from_options("Role", list(VALID_ROLES))
    device = _prompt_registry_choice("Device", DEVICES)
    framework = _prompt_registry_choice("Framework", FRAMEWORKS)
    family = _prompt_registry_choice("Family", FAMILIES)
    def _architecture_matches(entry):
        arch_family = entry.meta.get("family")
        arch_framework = entry.meta.get("framework")
        compatible = {arch_framework, *entry.meta.get("also_supports", [])} - {None}
        family_ok = not arch_family or arch_family.lower() == family.lower()
        framework_ok = not compatible or framework.lower() in {f.lower() for f in compatible}
        return family_ok and framework_ok

    architecture = _prompt_registry_choice(
        "Architecture",
        ARCHITECTURES,
        filter_fn=_architecture_matches,
    )
    application = _prompt_registry_choice("Application", APPLICATIONS)
    dataset = _prompt_registry_choice("Dataset", DATASETS)
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
