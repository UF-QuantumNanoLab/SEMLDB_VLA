# Universal ML-to-Verilog-A Converter

This tool extracts weights from trained PyTorch models (`.pth` or `.pkl`) and
automatically generates Spectre-compatible **Verilog-A** modules, so that
neural-network-based device compact models can be dropped directly into
Cadence Virtuoso simulations.

---

## Project Structure

```
verilogA_v2/
├── architectures/        # Plugin folder: one .py per device architecture
│   ├── base.py           #   Base class + math helpers (emit_linear_block, ...)
│   ├── sifet.py          #   SiFET plugin       (name = "SiFET")
│   ├── cntfet.py         #   CNTFET plugin      (name = "CNTFET")
│   ├── hfet.py           #   HFET plugin        (name = "HFET")
│   ├── diamondfet.py     #   DiamondFET plugin  (name = "DiamondFET")
│   ├── nmos.py           #   NMOS plugin        (name = "NMOS")
│   └── simple_mlp.py     #   Reference MLP example
│
├── checkpoints/          # Trained PyTorch weights (.pth / .pkl) go here
│   ├── SiFET/
│   │   └── sifet.pth
│   ├── CNTFET/
│   │   └── cntfet.pth
│   ├── HFET/
│   ├── DiamondFET/
│   ├── NMOS/
│   └── ...               #   One subfolder per device architecture
│
├── outputs/              # Generated Verilog-A files are written here
│   ├── SiFET/
│   │   └── sifet.va
│   ├── CNTFET/
│   │   └── cntfet.va
│   ├── HFET/
│   │   ├── HFET_Model.va
│   │   └── *.txt         #   Auxiliary weight files (for large arrays)
│   ├── DiamondFET/
│   ├── NMOS/
│   └── ...               #   One subfolder per device architecture
│
└── universal_export.py   # Main command-line exporter
```

For each device you want to model, you need two things:
1. A **plugin** in [architectures/](architectures/) describing the
   PyTorch→Verilog-A wiring for that architecture.
2. A **trained checkpoint** under `checkpoints/<DeviceName>/` holding the
   learned weights.

The generated `.va` file lands under `outputs/<DeviceName>/` — each device
architecture gets its own subfolder so the output tree stays organized when
you're iterating on several devices (SiFET, CNTFET, HFET, DiamondFET, NMOS,
…) in parallel. The `<DeviceName>` folder name should match the plugin's
`name` attribute for consistency.

---

## Quick Start: Export an Existing Model Type

If you retrained one of the already-supported architectures (e.g. SiFET) and
just want to regenerate the Verilog-A with updated weights:

```bash
python universal_export.py \
    --checkpoint checkpoints/SiFET/new_model.pth \
    --out outputs/SiFET/new_model.va \
    --module-name NewModelVA \
    --arch SiFET
```

> **Note:** `--arch` must exactly match the `name` attribute defined inside
> the architecture plugin class (not the filename). The `checkpoints/` and
> `outputs/` subfolder names should match `--arch` so each device's files
> stay grouped together.

### Argument reference

| Flag | Required | Description |
|---|---|---|
| `--checkpoint` | yes* | Path to the trained `.pth` file, typically under `checkpoints/<DeviceName>/`. |
| `--out` | yes | Path for the generated `.va` file, typically under `outputs/<DeviceName>/`. |
| `--module-name` | yes | Verilog-A `module` name used inside the `.va` file. |
| `--arch` | yes | Architecture plugin `name` (see list below). |

\* Some plugins may declare `requires_checkpoint = False` and can be exported
without weights.

### Currently available architectures

`SiFET`, `CNTFET`, `HFET`, `DiamondFET`, `NMOS`

---

## Adding a New Device Model (e.g. 2D-FET, FeFET, HEMT)

When you train a brand-new neural network for a new device, drop a plugin into
`architectures/` to tell the exporter how to wire the math. The main script
auto-discovers any `ModelArchitecture` subclass — no manual registration
needed.

### Step 1 — Create the plugin file

Create `architectures/my_device.py` and subclass `ModelArchitecture` with a
unique `name`:

```python
from typing import Dict, List
import numpy as np
from architectures.base import ModelArchitecture, require, gen_names, emit_linear_block

class MyDeviceArchitecture(ModelArchitecture):
    name = "MyDevice"
```

### Step 2 — Extract tensor weights (`parse_weights`)

A PyTorch checkpoint is a dict mapping layer names to raw tensors. Use the
`require()` helper — it also verifies the expected shape so mistakes are
caught early:

```python
    def parse_weights(self, sd: Dict[str, np.ndarray]) -> None:
        self.weights = {}
        self.weights["fc1_w"] = require(sd, "fc1.weight", (16, 3))
        self.weights["fc1_b"] = require(sd, "fc1.bias",   (16,))
```

### Step 3 — Write the Verilog-A math (`emit_model`)

The exporter can't read Python `forward()` logic — you must spell out the
Verilog-A math. Use the built-in helpers (e.g. `emit_linear_block`) to keep
it short:

```python
    def emit_model(self, module_name: str) -> str:
        lines: List[str] = [
            f"module {module_name}(d, g, s);",
            "  inout d, g, s;",
            "  electrical d, g, s;",
            "  real vgs, vds, out_0, out_1;",
            "  analog begin",
            "    vgs = V(g, s);",
            "    vds = V(d, s);",
        ]

        emit_linear_block(
            lines=lines,
            out_vars=["out_0", "out_1"],
            weight=self.weights["fc1_w"],
            bias=self.weights["fc1_b"],
            in_vars=["vgs", "vds", "1.0"],
        )

        lines.extend([
            "    I(d,s) <+ out_0;",
            "  end",
            "endmodule",
        ])
        return "\n".join(lines)
```

### Step 4 — Run the exporter

Save the plugin, drop the trained weights into `checkpoints/MyDevice/`, and
run:

```bash
python universal_export.py \
    --checkpoint checkpoints/MyDevice/my_device.pth \
    --out outputs/MyDevice/my_device.va \
    --module-name MyDeviceVA \
    --arch MyDevice
```

# For Example:
## 1.SiFET
python universal_export.py `
    --checkpoint checkpoints/SiFET/sifet.pth `
    --out outputs/SiFET/sifet.va `
    --module-name SiFETVA `
    --arch SiFET

## 2. CNTFET
python universal_export.py `
    --checkpoint checkpoints/CNTFET/cntfet.pth `
    --out outputs/CNTFET/cntfet.va `
    --module-name CNTFETVA `
    --arch CNTFET

## 3.DiamondFET
python universal_export.py `
    --out outputs/DiamondFET/DiamondFET_Model.va `
    --module-name DiamondFETVA `
    --arch DiamondFET
## 4.HFET 
python universal_export.py `
    --out outputs/HFET/HFET.va `
    --module-name HFETVA `
    --arch HFET   
## 5.NMOS
python universal_export.py `
    --out outputs/NMOS/NMOS.va `
    --module-name NMOSVA `
    --arch NMOS


> **Tip:** Start by copying [architectures/simple_mlp.py](architectures/simple_mlp.py)
> — it's the shortest working example and covers the full `parse_weights` /
> `emit_model` pattern end-to-end.

---

## Typical Workflow

1. Train your PyTorch model for the device (I–V / C–V regression, etc.).
2. Save the `state_dict` to `checkpoints/<DeviceName>/<device>.pth`.
3. Make sure a matching plugin exists in `architectures/` (or write one).
4. Run `universal_export.py` with the right `--arch`, pointing `--out` to
   `outputs/<DeviceName>/<device>.va`.
5. Import the generated `.va` from `outputs/<DeviceName>/` into Cadence
   Virtuoso and simulate.
