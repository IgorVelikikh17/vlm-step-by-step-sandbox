from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local VLM inference environment.")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"python version: {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"torch version: {_package_version('torch')}")
    print(f"transformers version: {_package_version('transformers')}")
    print(f"datasets version: {_package_version('datasets')}")
    print(f"accelerate version: {_package_version('accelerate')}")
    print(f"pillow version: {_package_version('Pillow')}")

    torch = _import_torch()
    if torch is None:
        print("device cuda available: unknown (torch import failed)")
        print("device mps available: unknown (torch import failed)")
        print("device cpu available: yes")
    else:
        print(f"device cuda available: {torch.cuda.is_available()}")
        print(f"device mps available: {_mps_available(torch)}")
        print("device cpu available: yes")

    print(f"model name: {args.model_name}")
    _check_processor(args.model_name)
    _check_model(args.model_name)


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _import_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _mps_available(torch) -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _check_processor(model_name: str) -> None:
    print()
    print("AutoProcessor.from_pretrained:")
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        print("success: true")
        print(f"processor class: {processor.__class__.__name__}")
    except Exception as error:
        print("success: false")
        print(f"exception type: {error.__class__.__name__}")
        print(f"exception message: {error}")


def _check_model(model_name: str) -> None:
    print()
    print("AutoModelForImageTextToText.from_pretrained:")
    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_name)
        print("success: true")
        print(f"model class: {model.__class__.__name__}")
    except Exception as error:
        print("success: false")
        print(f"exception type: {error.__class__.__name__}")
        print(f"exception message: {error}")


if __name__ == "__main__":
    main()
