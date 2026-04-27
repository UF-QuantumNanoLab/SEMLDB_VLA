# Universal ML-to-Verilog-A Converter

This tool extracts weights from trained PyTorch models (`.pth` or `.pkl`) and automatically generates Spectre-compatible **Verilog-A** modules.

## Files
- `universal_export.py`: The main command-line script.
- `architectures/`: A folder containing specific neural network definitions (plugins).

---

##  Quick Start: Export an Existing Model Type
If you retrained an existing model (like SiFET) and just want to generate new Verilog-A code with updated weights:
```bash
python universal_export.py \
    --checkpoint checkpoint/new_model.pth \
    --out out/new_model.va \
    --module-name NewModelVA \
    --arch SiFET
```
*(**Note:** The `--arch` argument must exactly match the `name` variable defined inside the architecture plugin).*

---

##  Adding a New Device Model (e.g., CNTFET, HEMT)

When you build a brand new neural network architecture, you must create a new Python plugin inside the `architectures/` folder to tell the exporter how to wire the math together.

### Step 1: Create your Plugin File
Create a new file like `architectures/cntfet.py` and define a class that inherits from `ModelArchitecture`. Give it a unique `name`.

```python
from typing import Dict, List
import numpy as np
from architectures.base import ModelArchitecture, require, gen_names, emit_linear_block

class CNTFETArchitecture(ModelArchitecture):
    name = "CNTFET"
```

### Step 2: Extract Tensor Weights (`parse_weights`)
A PyTorch file is just a dictionary mapping layer names to raw number matrices. The exporter needs to know exactly which PyTorch tensors to grab.

Implement the `parse_weights` function using the `require()` helper. It checks the shape automatically so mistakes are caught early:
```python
    def parse_weights(self, sd: Dict[str, np.ndarray]) -> None:
        self.weights = {}
        # Grab 'fc1.weight' from PyTorch and verify its shape is (16, 3)
        self.weights["fc1_w"] = require(sd, "fc1.weight", (16, 3))
        self.weights["fc1_b"] = require(sd, "fc1.bias", (16,))
```

### Step 3: Write the Verilog-A Math (`emit_model`)
The exporter cannot read Python `.forward()` logic, so you must write out the sequential Verilog-A math strings inside `emit_model()`.

```python
    def emit_model(self, module_name: str) -> str:
        lines: List[str] = [
            f"module {module_name}(d, g, s);",
            "  inout d, g, s;",
            "  electrical d, g, s;",
            "  real vgs, vds, out_0, out_1;", # Define internal variables
            "  analog begin",
            "    vgs = V(g, s);",
            "    vds = V(d, s);"
        ]

        # Use the built-in math helpers to write the Verilog logic instantly!
        emit_linear_block(
            lines=lines,
            out_vars=["out_0", "out_1"],   # Verilog Output variables
            weight=self.weights["fc1_w"],  # The PyTorch matrix
            bias=self.weights["fc1_b"],    # The PyTorch bias
            in_vars=["vgs", "vds", "1.0"]  # Verilog Input variables
        )

        lines.extend([
            "    I(d,s) <+ out_0;", # Map standard ML output to physics component
            "  end",
            "endmodule"
        ])
        
        return "\n".join(lines)
```

### Step 4: Run the Exporter!
As soon as `architectures/cntfet.py` is saved, the main script auto-discovers it. You don't need to register it anywhere. You can immediately run:

```bash
python universal_export.py \
    --checkpoint checkpoint/new_model.pth \
    --out out/new_model.va \
    --module-name NewModelVA \
    --arch SiFET
```

💡 **Tip:** Peek at `architectures/simple_mlp.py` for a fully working, robust baseline example you can copy-paste!

Few Examples:

### CNTFET

python universal_export.py \
  --checkpoint checkpoint/cntfet.pth \
  --out out/new_model.va \
  --module-name NewModelVA \
  --arch CNTFET

python universal_export.py --checkpoint checkpoint/cntfet.pth --out out/new_model.va --module-name NewModelVA --arch CNTFET






### DiamondFET  
python universal_export.py \
  --checkpoint checkpoint/new_model.pth \
  --out out/new_model.va \
  --module-name NewModelVA \
  --arch DiamondFET

python universal_export.py --checkpoint checkpoint/new_model.pth --out out/new_model.va --module-name NewModelVA --arch DiamondFET
