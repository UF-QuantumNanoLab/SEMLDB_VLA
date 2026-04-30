import numpy as np
from typing import Dict, List
from architectures.base import (
    ModelArchitecture, ExportError, fmt, require, gen_names, emit_linear_block, emit_gelu
)

class SimpleMLPArchitecture(ModelArchitecture):
    name = "SimpleMLP"
    
    # Example layer counts to demonstrate mapping. Change this as needed per your model.
    layers = [3, 16, 16, 2] # Inputs: vgs, vds, bias -> Outputs: ids, q

    def parse_weights(self, sd: Dict[str, np.ndarray]) -> None:
        self.weights = {}
        # Parse 2 hidden layers for the SimpleMLP structure
        self.weights["fc1_w"] = require(sd, "fc1.weight", (self.layers[1], self.layers[0]))
        self.weights["fc1_b"] = require(sd, "fc1.bias", (self.layers[1],))
        
        self.weights["fc2_w"] = require(sd, "fc2.weight", (self.layers[2], self.layers[1]))
        self.weights["fc2_b"] = require(sd, "fc2.bias", (self.layers[2],))
        
        self.weights["fc3_w"] = require(sd, "fc3.weight", (self.layers[3], self.layers[2]))
        self.weights["fc3_b"] = require(sd, "fc3.bias", (self.layers[3],))

    def emit_model(self, module_name: str) -> str:
        lines: List[str] = [
            '`include "constants.vams"',
            '`include "disciplines.vams"',
            "",
            f"module {module_name}(d, g, s);",
            "  inout d, g, s;",
            "  electrical d, g, s;",
            "",
            "  real vgs, vds, out_ids, out_q;",
            "",
            "  analog function real gelu_approx;",
            "    input x;",
            "    real x, c;",
            "    begin",
            "      c = 0.7978845608;",
            "      gelu_approx = 0.5 * x * (1.0 + tanh(c * (x + 0.044715*x*x*x)));",
            "    end",
            "  endfunction",
            "",
        ]

        # Definitions
        for name in gen_names("fc1_l", self.layers[1]): lines.append(f"  real {name};")
        for name in gen_names("fc1_g", self.layers[1]): lines.append(f"  real {name};")
        for name in gen_names("fc2_l", self.layers[2]): lines.append(f"  real {name};")
        for name in gen_names("fc2_g", self.layers[2]): lines.append(f"  real {name};")
        for name in gen_names("fc3_l", self.layers[3]): lines.append(f"  real {name};")

        lines.extend([
            "  analog begin",
            "    vgs = V(g, s);",
            "    vds = V(d, s);",
        ])

        # Layer 1
        emit_linear_block(lines, gen_names("fc1_l", self.layers[1]), self.weights["fc1_w"], self.weights["fc1_b"], ["vgs", "vds", "1.0"])
        emit_gelu(lines, gen_names("fc1_l", self.layers[1]), gen_names("fc1_g", self.layers[1]))
        
        # Layer 2
        emit_linear_block(lines, gen_names("fc2_l", self.layers[2]), self.weights["fc2_w"], self.weights["fc2_b"], gen_names("fc1_g", self.layers[1]))
        emit_gelu(lines, gen_names("fc2_l", self.layers[2]), gen_names("fc2_g", self.layers[2]))
        
        # Layer 3
        emit_linear_block(lines, gen_names("fc3_l", self.layers[3]), self.weights["fc3_w"], self.weights["fc3_b"], gen_names("fc2_g", self.layers[2]))

        lines.extend([
            "    out_ids = fc3_l_0;",
            "    out_q   = fc3_l_1;",
            "",
            "    I(d,s) <+ out_ids;",
            "  end",
            "endmodule",
            ""
        ])
        
        return "\n".join(lines)
