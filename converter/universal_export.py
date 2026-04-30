#!/usr/bin/env python3
import argparse
import os
import sys
import importlib
import pkgutil
import inspect
import torch

# Add current dir to sys.path so 'architectures' module resolves
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architectures.base import ModelArchitecture, ExportError

def load_state_dict(path: str) -> dict:
    if not os.path.isfile(path):
        raise ExportError(f"Checkpoint not found: {path}")
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        blob = blob["state_dict"]
    if not isinstance(blob, dict):
        raise ExportError("Checkpoint is not a PyTorch state_dict.")

    out = {}
    for k, v in blob.items():
        if hasattr(v, "detach"):
            out[k] = v.detach().cpu().numpy().astype(float)
    if not out:
        raise ExportError("No tensors found in checkpoint.")
    return out

def discover_architectures():
    import architectures
    models = {}
    path = architectures.__path__
    prefix = architectures.__name__ + "."
    for _, modname, _ in pkgutil.iter_modules(path, prefix):
        module = importlib.import_module(modname)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, ModelArchitecture) and obj is not ModelArchitecture:
                arch_name = getattr(obj, "name", name)
                models[arch_name] = obj
    return models

def main():
    models = discover_architectures()
    parser = argparse.ArgumentParser(description="Universal ML-to-Verilog-A Exporter")
    parser.add_argument("--checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Output .va file")
    parser.add_argument("--module-name", type=str, required=True, help="Verilog-A module name")
    parser.add_argument("--arch", type=str, required=True, choices=list(models.keys()), help="Model Architecture Plugin")
    args = parser.parse_args()

    try:
        ArchClass = models[args.arch]
        requires_checkpoint = getattr(ArchClass, "requires_checkpoint", True)
        if requires_checkpoint and not args.checkpoint:
            raise ExportError(f"--checkpoint is required for architecture: {args.arch}")

        sd = load_state_dict(args.checkpoint) if requires_checkpoint else {}
        arch_instance = ArchClass()
        arch_instance.parse_weights(sd)
        arch_instance.print_summary()
        
        va_code = arch_instance.emit_model(args.module_name)
        
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(va_code)
        print(f"Successfully generated Verilog-A: {args.out}")
        return 0
    except ExportError as exc:
        print(f"ERROR: {exc}")
        return 2
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"UNEXPECTED ERROR: {exc}")
        return 3

if __name__ == "__main__":
    sys.exit(main())
