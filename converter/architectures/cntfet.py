import numpy as np
from typing import Dict, List
from architectures.base import (
    ModelArchitecture, ExportError, fmt, require, gen_names, emit_linear_block
)

class CNTFETArchitecture(ModelArchitecture):
    name = "CNTFET"

    def parse_weights(self, sd: Dict[str, np.ndarray]) -> None:
        self.weights = {}
        
        # 1. Embedding / Conv1d
        self.weights["emb_w"] = require(sd, "embedding.weight", (32, 1, 1))
        self.weights["emb_b"] = require(sd, "embedding.bias", (32,))

        # 2. BiGRU layers
        for l in range(3):
            for dir_idx, suffix in enumerate(["", "_reverse"]):
                prefix = f"bigru."
                layer_suffix = f"_l{l}{suffix}"
                
                ih_shape = (96, 32) if l == 0 else (96, 64)
                hh_shape = (96, 32)
                
                self.weights[f"gru_ih_w_{l}{suffix}"] = require(sd, f"{prefix}weight_ih{layer_suffix}", ih_shape)
                self.weights[f"gru_hh_w_{l}{suffix}"] = require(sd, f"{prefix}weight_hh{layer_suffix}", hh_shape)
                self.weights[f"gru_ih_b_{l}{suffix}"] = require(sd, f"{prefix}bias_ih{layer_suffix}", (96,))
                self.weights[f"gru_hh_b_{l}{suffix}"] = require(sd, f"{prefix}bias_hh{layer_suffix}", (96,))

        # 3. Linear out
        self.weights["linear_w"] = require(sd, "linear.weight", (2, 64))
        self.weights["linear_b"] = require(sd, "linear.bias", (2,))

    def emit_model(self, module_name: str) -> str:
        lines: List[str] = [
            '`include "constants.vams"',
            '`include "disciplines.vams"',
            "",
            f"module {module_name}(d, g, s);",
            "  inout d, g, s;",
            "  electrical d, g, s;",
            "",
            "  // Device model parameters",
            "  parameter real tox = 2.0;",
            "  parameter real Lg = 10.0;",
            "  parameter real eps_ox = 3.9;",
            "  parameter real V_th = 0.258;",
            "  parameter real Wid = 1.0 from [0:inf);",
            "",
            "  // Numerical constants",
            "  parameter real smooth_delta = 1e-12;",
            "  parameter real vds_floor = 1e-6;",
            "  parameter real gmin_ds = 1e-12;",
            "",
            "  // Internal scalars",
            "  real vgs, vds, vgs_shift, vds_pos, sign_vds, vds_abs;",
            "  real tox_in, lg_in, eps_in;",
            "  real gate_r, gate_z, gate_n;",
            "  real log_id_lim, ids_mag, ids;",
            "  real qg, qs, qd;",
            "",
        ]

        # Declare signals
        # Setup inputs block
        lines.append("  real seq_in_0, seq_in_1, seq_in_2, seq_in_3, seq_in_4;")

        # Embedding outputs
        for t in range(5):
            for name in gen_names(f"emb_t{t}", 32): lines.append(f"  real {name};")
        
        # BiGRU states
        # l0 -> input 32, out 32+32=64
        # l1 -> input 64, out 32+32=64
        # l2 -> input 64, out 32+32=64
        for l in range(3):
            for t in range(5):
                for name in gen_names(f"gru_{l}_fwd_t{t}_h", 32): lines.append(f"  real {name};")
                for name in gen_names(f"gru_{l}_rev_t{t}_h", 32): lines.append(f"  real {name};")
                for name in gen_names(f"gru_{l}_out_t{t}", 64):   lines.append(f"  real {name};")

        # Linear
        for name in gen_names("out", 2): lines.append(f"  real {name};")

        lines.extend([
            "",
            "  analog function real smooth_pos;",
            "    input x, dlt;",
            "    real x, dlt;",
            "    begin",
            "      smooth_pos = 0.5*(x + sqrt(x*x + dlt*dlt));",
            "    end",
            "  endfunction",
            "",
            "  analog function real sigmoid;",
            "    input x;",
            "    real x;",
            "    begin",
            "      sigmoid = 1.0 / (1.0 + exp(-x));",
            "    end",
            "  endfunction",
            "",
            "  analog function real smooth_log;",
            "    input x, dlt;",
            "    real x, dlt;",
            "    begin",
            "      smooth_log = ln(smooth_pos(x, dlt));",
            "    end",
            "  endfunction",
            "",
            "  analog begin",
            "    vgs = V(g,s);",
            "    vds = V(d,s);",
            "    vgs_shift = vgs - (V_th - 0.258);",
            "    vds_abs = sqrt(vds*vds + vds_floor*vds_floor);",
            "    vds_pos = smooth_pos(vds_abs, vds_floor);",
            "    sign_vds = vds / sqrt(vds*vds + 1e-6);",
            "",
            "    // Static attributes",
            "    tox_in = smooth_pos(tox, smooth_delta);",
            "    lg_in = smooth_pos(Lg, smooth_delta);",
            "    eps_in = smooth_pos(eps_ox, smooth_delta);",
            "",
            "    // The BiGRU is trained on seq_len=5: [tox, Lg, eps_ox, Vg_shift, Vd]",
            "    seq_in_0 = tox_in;",
            "    seq_in_1 = lg_in;",
            "    seq_in_2 = eps_in;",
            "    seq_in_3 = vgs_shift;",
            "    seq_in_4 = vds_pos;",
            "",
        ])

        # 1. Embedding
        lines.append("    // 1. Embedding / Conv1d (in_channels=1, out_channels=32)")
        emb_w = self.weights["emb_w"][:, 0, 0] # shape (32,)
        emb_b = self.weights["emb_b"]          # shape (32,)
        for t in range(5):
            lines.append(f"    // Timestep {t}")
            for i in range(32):
                lines.append(f"    emb_t{t}_{i} = ({fmt(emb_w[i])}) * seq_in_{t} + ({fmt(emb_b[i])});")
            lines.append("")

        # Helper to emit a GRU step
        def emit_gru_step(layer, t, dir_name, prev_h, in_vars, in_dim):
            suffix = "_reverse" if dir_name == "rev" else ""
            ih_w = self.weights[f"gru_ih_w_{layer}{suffix}"]
            hh_w = self.weights[f"gru_hh_w_{layer}{suffix}"]
            ih_b = self.weights[f"gru_ih_b_{layer}{suffix}"]
            hh_b = self.weights[f"gru_hh_b_{layer}{suffix}"]

            for h in range(32):
                r_idx = h
                z_idx = h + 32
                n_idx = h + 64

                # Reset gate r
                r_expr = [f"({fmt(ih_b[r_idx] + hh_b[r_idx])})"]
                for i in range(in_dim): r_expr.append(f"({fmt(ih_w[r_idx, i])}) * {in_vars[i]}")
                if prev_h is not None:
                    for i in range(32): r_expr.append(f"({fmt(hh_w[r_idx, i])}) * {prev_h[i]}")
                lines.append(f"    // h={h}")
                lines.append(f"    gate_r = sigmoid({' + '.join(r_expr)});")

                # Update gate z
                z_expr = [f"({fmt(ih_b[z_idx] + hh_b[z_idx])})"]
                for i in range(in_dim): z_expr.append(f"({fmt(ih_w[z_idx, i])}) * {in_vars[i]}")
                if prev_h is not None:
                    for i in range(32): z_expr.append(f"({fmt(hh_w[z_idx, i])}) * {prev_h[i]}")
                lines.append(f"    gate_z = sigmoid({' + '.join(z_expr)});")

                # New gate n
                n_base = [f"({fmt(ih_b[n_idx])})"]
                for i in range(in_dim): n_base.append(f"({fmt(ih_w[n_idx, i])}) * {in_vars[i]}")
                if prev_h is not None:
                    n_recur = [f"({fmt(hh_b[n_idx])})"]
                    for i in range(32): n_recur.append(f"({fmt(hh_w[n_idx, i])}) * {prev_h[i]}")
                    lines.append(f"    gate_n = tanh({' + '.join(n_base)} + gate_r * ({' + '.join(n_recur)}));")
                else:
                    lines.append(f"    gate_n = tanh({' + '.join(n_base)} + gate_r * ({fmt(hh_b[n_idx])}));")

                # Hidden state h
                if prev_h is not None:
                    lines.append(f"    gru_{layer}_{dir_name}_t{t}_h_{h} = (1.0 - gate_z) * gate_n + gate_z * {prev_h[h]};")
                else:
                    lines.append(f"    gru_{layer}_{dir_name}_t{t}_h_{h} = (1.0 - gate_z) * gate_n;")

        # 2. BiGRU layers
        for l in range(3):
            lines.append(f"    // 2. BiGRU Layer {l}")
            in_dim = 32 if l == 0 else 64

            # Forward pass: t = 0 to 4
            lines.append(f"    // Forward sequence")
            for t in range(5):
                lines.append(f"    // FW step t={t}")
                in_vars = gen_names(f"emb_t{t}", 32) if l == 0 else gen_names(f"gru_{l-1}_out_t{t}", 64)
                prev_h = None if t == 0 else gen_names(f"gru_{l}_fwd_t{t-1}_h", 32)
                emit_gru_step(l, t, "fwd", prev_h, in_vars, in_dim)
                lines.append("")

            # Reverse pass: t = 4 down to 0
            lines.append(f"    // Reverse sequence")
            for t in range(4, -1, -1):
                lines.append(f"    // RV step t={t}")
                in_vars = gen_names(f"emb_t{t}", 32) if l == 0 else gen_names(f"gru_{l-1}_out_t{t}", 64)
                prev_h = None if t == 4 else gen_names(f"gru_{l}_rev_t{t+1}_h", 32)
                emit_gru_step(l, t, "rev", prev_h, in_vars, in_dim)
                lines.append("")
            
            # Concat fwd and rev outputs for this layer
            for t in range(5):
                for h in range(32):
                    lines.append(f"    gru_{l}_out_t{t}_{h} = gru_{l}_fwd_t{t}_h_{h};")
                    lines.append(f"    gru_{l}_out_t{t}_{h+32} = gru_{l}_rev_t{t}_h_{h};")
            lines.append("")

        # 3. Linear
        lines.append("    // 3. Linear output on final sequence step [-1] -> t=4")
        emit_linear_block(lines, gen_names("out", 2), self.weights["linear_w"], self.weights["linear_b"], gen_names("gru_2_out_t4", 64))

        # Output formatting based on the backend ML `run_rnn_sim` metrics
        #   Id_pred = torch.exp(Id_ratio_pred).cpu() * torch.log(Vd_mesh.flatten()*10+1) * 4.049253845214844 * 1e-6
        #   Qg_pred = Qg_pred * 1e-18
        lines.extend([
            "",
            "    // Physics Scaling according to backend target",
            "    // out_0 = Id_ratio_pred, out_1 = Qg_pred",
            "    ids_mag = limexp(out_0) * smooth_log(vds_pos * 10.0 + 1.0, smooth_delta) * 4.049253845214844 * 1.0e-6;",
            "    ids = sign_vds * Wid * ids_mag;",
            "",
            "    qg = Wid * out_1 * 1.0e-18;",
            "    qs = -2.0/3.0 * qg;",
            "    qd = -1.0/3.0 * qg;",
            "",
            "    I(d,s) <+ ids;",
            "    I(d,s) <+ gmin_ds * V(d,s);",
            "    I(g) <+ ddt(qg);",
            "    I(s) <+ ddt(qs);",
            "    I(d) <+ ddt(qd);",
            "  end",
            "endmodule"
        ])

        return "\n".join(lines)
