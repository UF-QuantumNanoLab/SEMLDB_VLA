from __future__ import annotations
import numpy as np
from typing import Dict, List, Sequence

class ExportError(RuntimeError):
    """Raised for malformed checkpoints or unsupported model variants."""

def fmt(x: float) -> str:
    return f"{float(x):.17g}"

def require(sd: Dict[str, np.ndarray], key: str, shape: tuple) -> np.ndarray:
    if key not in sd:
        raise ExportError(f"Missing key in checkpoint: {key}")
    arr = np.array(sd[key], dtype=float)
    if arr.shape != shape:
        raise ExportError(f"Shape mismatch for {key}: expected {shape}, got {arr.shape}")
    return arr

def gen_names(prefix: str, n: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(n)]

def emit_linear_block(
    lines: List[str],
    out_vars: Sequence[str],
    weight: np.ndarray,
    bias: np.ndarray,
    in_vars: Sequence[str],
) -> None:
    if weight.shape != (len(out_vars), len(in_vars)):
        raise ExportError(
            f"Linear dimension mismatch: weight {weight.shape}, out {len(out_vars)}, in {len(in_vars)}"
        )
    if bias.shape != (len(out_vars),):
        raise ExportError(f"Linear bias mismatch: bias {bias.shape}, out {len(out_vars)}")
    for i, out_name in enumerate(out_vars):
        lines.append(f"    {out_name} = {fmt(bias[i])};")
        for k, in_name in enumerate(in_vars):
            lines.append(f"    {out_name} = {out_name} + ({fmt(weight[i, k])})*{in_name};")

def emit_layernorm(
    lines: List[str],
    in_vars: Sequence[str],
    out_vars: Sequence[str],
    prefix: str,
    eps_name: str,
    gamma: np.ndarray | None,
    beta: np.ndarray | None,
) -> None:
    n = len(in_vars)
    if len(out_vars) != n:
        raise ExportError(f"LayerNorm variable length mismatch in {prefix}: {len(in_vars)} vs {len(out_vars)}")
    if gamma is not None and gamma.shape != (n,):
        raise ExportError(f"LayerNorm gamma shape mismatch in {prefix}: expected {(n,)}, got {gamma.shape}")
    if beta is not None and beta.shape != (n,):
        raise ExportError(f"LayerNorm beta shape mismatch in {prefix}: expected {(n,)}, got {beta.shape}")

    lines.append(f"    {prefix}_mean = (" + " + ".join(in_vars) + f") / {n};")
    var_terms = [f"(({v}) - {prefix}_mean)*(({v}) - {prefix}_mean)" for v in in_vars]
    lines.append(f"    {prefix}_var = (" + " + ".join(var_terms) + f") / {n};")
    for i in range(n):
        if gamma is None:
            lines.append(
                f"    {out_vars[i]} = (({in_vars[i]}) - {prefix}_mean) / sqrt({prefix}_var + {eps_name});"
            )
        else:
            lines.append(
                f"    {out_vars[i]} = ((({in_vars[i]}) - {prefix}_mean) / sqrt({prefix}_var + {eps_name}))"
                f" * ({fmt(gamma[i])}) + ({fmt(beta[i])});"
            )

def emit_gelu(lines: List[str], in_vars: Sequence[str], out_vars: Sequence[str]) -> None:
    if len(in_vars) != len(out_vars):
        raise ExportError("GELU variable length mismatch.")
    for i in range(len(in_vars)):
        lines.append(f"    {out_vars[i]} = gelu_approx({in_vars[i]});")

class ModelArchitecture:
    """Base class that all ML-to-VerilogA model exporters should inherit from."""
    name = "BaseArchitecture"

    def parse_weights(self, state_dict: Dict[str, np.ndarray]) -> None:
        """Extract weight tensors from checkpoint."""
        raise NotImplementedError

    def emit_model(self, module_name: str) -> str:
        """Generate and return the Verilog-A code as a string."""
        raise NotImplementedError

    def print_summary(self) -> None:
        """Optional: Print a model summary."""
        print(f"Model summary for {self.name} architecture.")
