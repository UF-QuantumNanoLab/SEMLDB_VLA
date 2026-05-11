import numpy as np
from typing import Dict, List
from architectures.base import (
    ModelArchitecture, ExportError, fmt, require, gen_names, emit_linear_block
)


class TwoDFETArchitecture(ModelArchitecture):
    name = "2DFET"
    device_feature_names = ["tox", "Lg", "eps_ox", "meff", "D"]
    device_mean = np.array([1.95006670e-09, 1.44512793e-08, 1.40857444e01, 5.0e-01, 0.0])
    device_scale = np.array([7.07858992e-10, 8.52838872e-09, 7.41677989e00, 1.0, 1.0])
    bias_mean = np.array([2.24948330e-01, 2.51009710e-01])
    bias_scale = np.array([2.20061600e-01, 1.47910610e-01])
    target_mean = np.array([5.72119048e-01, -6.98499729e-11])
    target_scale = np.array([2.41033565e00, 1.16392593e-10])

    def parse_weights(self, sd: Dict[str, np.ndarray]) -> None:
        self.weights = {}
        prefix = "backbone."

        emb = sd.get(prefix + "device_type_embeddings.weight")
        if emb is None:
            raise ExportError("Missing embedding weight: backbone.device_type_embeddings.weight")
        num_device_features, embed = emb.shape
        self.num_device_features = num_device_features
        self.embed_dim = embed

        self.weights["emb"] = emb
        self.weights["ds_w0"] = require(sd, prefix + "device_shared_mlp.0.weight", (2 * embed, 1 + embed))
        self.weights["ds_b0"] = require(sd, prefix + "device_shared_mlp.0.bias", (2 * embed,))
        self.weights["ds_ln0_w"] = require(sd, prefix + "device_shared_mlp.1.weight", (2 * embed,))
        self.weights["ds_ln0_b"] = require(sd, prefix + "device_shared_mlp.1.bias", (2 * embed,))
        self.weights["ds_w1"] = require(sd, prefix + "device_shared_mlp.3.weight", (embed, 2 * embed))
        self.weights["ds_b1"] = require(sd, prefix + "device_shared_mlp.3.bias", (embed,))

        self.weights["bt_w0"] = require(sd, prefix + "bias_mlp.0.weight", (4 * embed, 2))
        self.weights["bt_b0"] = require(sd, prefix + "bias_mlp.0.bias", (4 * embed,))
        self.weights["bt_ln0_w"] = require(sd, prefix + "bias_mlp.1.weight", (4 * embed,))
        self.weights["bt_ln0_b"] = require(sd, prefix + "bias_mlp.1.bias", (4 * embed,))
        self.weights["bt_w1"] = require(sd, prefix + "bias_mlp.3.weight", (4 * embed, 4 * embed))
        self.weights["bt_b1"] = require(sd, prefix + "bias_mlp.3.bias", (4 * embed,))
        self.weights["bt_ln1_w"] = require(sd, prefix + "bias_mlp.4.weight", (4 * embed,))
        self.weights["bt_ln1_b"] = require(sd, prefix + "bias_mlp.4.bias", (4 * embed,))
        self.weights["bt_w2"] = require(sd, prefix + "bias_mlp.6.weight", (embed, 4 * embed))
        self.weights["bt_b2"] = require(sd, prefix + "bias_mlp.6.bias", (embed,))

        self.weights["film_w"] = require(sd, prefix + "film_projection.weight", (2 * embed, embed))
        self.weights["film_b"] = require(sd, prefix + "film_projection.bias", (2 * embed,))

        self.weights["fh_w0"] = require(sd, prefix + "output_head.0.weight", (4 * embed, embed))
        self.weights["fh_b0"] = require(sd, prefix + "output_head.0.bias", (4 * embed,))
        self.weights["fh_ln0_w"] = require(sd, prefix + "output_head.1.weight", (4 * embed,))
        self.weights["fh_ln0_b"] = require(sd, prefix + "output_head.1.bias", (4 * embed,))
        self.weights["fh_w1"] = require(sd, prefix + "output_head.3.weight", (2, 4 * embed))
        self.weights["fh_b1"] = require(sd, prefix + "output_head.3.bias", (2,))

    def print_summary(self) -> None:
        print(
            f"Model summary for {self.name}: "
            f"{self.num_device_features} device features, embed_dim={self.embed_dim}, outputs=(Id,Q)."
        )

    def emit_model(self, module_name: str) -> str:
        embed = self.embed_dim
        d_hidden = 2 * embed
        b_hidden = 4 * embed
        fh_hidden = 4 * embed
        ndev = self.num_device_features
        weights = self.weights

        lines: List[str] = [
            '`include "constants.vams"',
            '`include "disciplines.vams"',
            "",
            f"module {module_name}(d, g, s);",
            "  inout d, g, s;",
            "  electrical d, g, s;",
            "",
            "  // Device/model parameters",
            "  parameter real tox = 3.0e-9;",
            "  parameter real Lg = 12.0e-9;",
            "  parameter real eps_ox = 20.0;",
            "  parameter real meff = 0.5;",
            "  parameter real D = 0.0;",
            "  parameter real V_th = 0.0;",
            "  parameter real Wid = 1.0 from [0:inf);",
            "",
            "  // Numerical constants",
            "  parameter real ln_eps = 1e-5;",
            "  parameter real smooth_delta = 1e-12;",
            "  parameter real vds_floor = 1e-6;",
            "  parameter real vds_sign_smooth = 1e-3;",
            "  parameter real log10_id_min = -20.0;",
            "  parameter real log10_id_max = 20.0;",
            "  parameter real id_log_eps = 1e-20;",
            "  parameter real q_scale = 10.0;",
            "  parameter real gmin_ds = 1e-12;",
            "",
            "  // Internal scalars",
            "  real vgs, vds, vgs_scaled, vds_scaled, vds_abs, sign_vds;",
            "  real tox_scaled, lg_scaled, eps_ox_scaled, meff_scaled, d_scaled;",
            "  real id_scaled, id_log10, id_log10_lim, ids_mag, ids;",
            "  real q_scaled, q_ml_raw, qg, qs, qd;",
            "",
            "  analog function real smooth_pos;",
            "    input x, dlt;",
            "    real x, dlt;",
            "    begin",
            "      smooth_pos = 0.5*(x + sqrt(x*x + dlt*dlt));",
            "    end",
            "  endfunction",
            "",
            "  analog function real gelu_approx;",
            "    input x;",
            "    real x;",
            "    real c;",
            "    begin",
            "      c = 0.79788456080286541;",
            "      gelu_approx = 0.5 * x * (1.0 + tanh(c * (x + 0.044715*x*x*x)));",
            "    end",
            "  endfunction",
            "",
        ]

        for tok in range(ndev):
            lines.append(f"  // Device token {tok}")
            for prefix, n in [
                (f"dt{tok}_l0", d_hidden), (f"dt{tok}_n0", d_hidden),
                (f"dt{tok}_g0", d_hidden), (f"dt{tok}_l1", embed),
            ]:
                for name in gen_names(prefix, n):
                    lines.append(f"  real {name};")
            lines.append(f"  real dt{tok}_mean;")
            lines.append(f"  real dt{tok}_var;")
            lines.append("")

        for name in gen_names("hp", embed):
            lines.append(f"  real {name};")
        for name in gen_names("hpn", embed):
            lines.append(f"  real {name};")
        lines.append("  real hp_mean;")
        lines.append("  real hp_var;")
        lines.append("")

        for prefix, n in [
            ("bt_l0", b_hidden), ("bt_n0", b_hidden), ("bt_g0", b_hidden),
            ("bt_l1", b_hidden), ("bt_n1", b_hidden), ("bt_g1", b_hidden),
            ("hv", embed),
        ]:
            for name in gen_names(prefix, n):
                lines.append(f"  real {name};")
        lines.append("  real bt0_mean;")
        lines.append("  real bt0_var;")
        lines.append("  real bt1_mean;")
        lines.append("  real bt1_var;")
        lines.append("")

        for prefix, n in [
            ("film", 2 * embed), ("hfp", embed), ("hfn", embed),
            ("fh_l0", fh_hidden), ("fh_n0", fh_hidden), ("fh_g0", fh_hidden), ("fh_l1", 2),
        ]:
            for name in gen_names(prefix, n):
                lines.append(f"  real {name};")
        lines.append("  real hf_mean;")
        lines.append("  real hf_var;")
        lines.append("  real fh0_mean;")
        lines.append("  real fh0_var;")
        lines.append("")

        lines.extend([
            "  analog begin",
            "    vgs = V(g,s);",
            "    vds = V(d,s);",
            "    vds_abs = sqrt(vds*vds + vds_floor*vds_floor);",
            "    sign_vds = vds / sqrt(vds*vds + vds_sign_smooth*vds_sign_smooth);",
            "",
            "    // Match the StandardScaler objects saved by the training pipeline.",
            f"    tox_scaled = (tox - ({fmt(self.device_mean[0])})) / ({fmt(self.device_scale[0])});",
            f"    lg_scaled = (Lg - ({fmt(self.device_mean[1])})) / ({fmt(self.device_scale[1])});",
            f"    eps_ox_scaled = (eps_ox - ({fmt(self.device_mean[2])})) / ({fmt(self.device_scale[2])});",
            f"    meff_scaled = (meff - ({fmt(self.device_mean[3])})) / ({fmt(self.device_scale[3])});",
            f"    d_scaled = (D - ({fmt(self.device_mean[4])})) / ({fmt(self.device_scale[4])});",
            f"    vgs_scaled = (vgs - ({fmt(self.bias_mean[0])})) / ({fmt(self.bias_scale[0])});",
            f"    vds_scaled = (vds - ({fmt(self.bias_mean[1])})) / ({fmt(self.bias_scale[1])});",
            "",
        ])

        dev_inputs = ["tox_scaled", "lg_scaled", "eps_ox_scaled", "meff_scaled", "d_scaled"][:ndev]
        for tok in range(ndev):
            lines.append(f"    // Device tower token {tok}")
            eff_bias = weights["ds_b0"] + weights["ds_w0"][:, 1:] @ weights["emb"][tok]
            scalar_w = weights["ds_w0"][:, 0]
            for j in range(d_hidden):
                lines.append(f"    dt{tok}_l0_{j} = ({fmt(eff_bias[j])}) + ({fmt(scalar_w[j])})*{dev_inputs[tok]};")

            ln_in = gen_names(f"dt{tok}_l0", d_hidden)
            lines.append(f"    dt{tok}_mean = (" + " + ".join(ln_in) + f") / {d_hidden};")
            var_terms = [f"(({x}) - dt{tok}_mean)*(({x}) - dt{tok}_mean)" for x in ln_in]
            lines.append(f"    dt{tok}_var = (" + " + ".join(var_terms) + f") / {d_hidden};")
            for j in range(d_hidden):
                lines.append(f"    dt{tok}_n0_{j} = (((dt{tok}_l0_{j}) - dt{tok}_mean) / sqrt(dt{tok}_var + ln_eps)) * ({fmt(weights['ds_ln0_w'][j])}) + ({fmt(weights['ds_ln0_b'][j])});")
                lines.append(f"    dt{tok}_g0_{j} = gelu_approx(dt{tok}_n0_{j});")

            emit_linear_block(
                lines,
                gen_names(f"dt{tok}_l1", embed),
                weights["ds_w1"],
                weights["ds_b1"],
                gen_names(f"dt{tok}_g0", d_hidden),
            )

        lines.append("    // Device token pooling")
        for k in range(embed):
            token_sum = " + ".join(f"dt{tok}_l1_{k}" for tok in range(ndev))
            lines.append(f"    hp_{k} = ({token_sum}) / {float(ndev):.1f};")
        hp_in = gen_names("hp", embed)
        lines.append("    hp_mean = (" + " + ".join(hp_in) + f") / {embed};")
        hp_var_terms = [f"(({x}) - hp_mean)*(({x}) - hp_mean)" for x in hp_in]
        lines.append("    hp_var = (" + " + ".join(hp_var_terms) + f") / {embed};")
        for k in range(embed):
            lines.append(f"    hpn_{k} = ((hp_{k}) - hp_mean) / sqrt(hp_var + ln_eps);")

        lines.append("    // Bias tower")
        emit_linear_block(lines, gen_names("bt_l0", b_hidden), weights["bt_w0"], weights["bt_b0"], ["vgs_scaled", "vds_scaled"])
        bt0_in = gen_names("bt_l0", b_hidden)
        lines.append("    bt0_mean = (" + " + ".join(bt0_in) + f") / {b_hidden};")
        bt0_var_terms = [f"(({x}) - bt0_mean)*(({x}) - bt0_mean)" for x in bt0_in]
        lines.append("    bt0_var = (" + " + ".join(bt0_var_terms) + f") / {b_hidden};")
        for i in range(b_hidden):
            lines.append(f"    bt_n0_{i} = (((bt_l0_{i}) - bt0_mean) / sqrt(bt0_var + ln_eps)) * ({fmt(weights['bt_ln0_w'][i])}) + ({fmt(weights['bt_ln0_b'][i])});")
            lines.append(f"    bt_g0_{i} = gelu_approx(bt_n0_{i});")

        emit_linear_block(lines, gen_names("bt_l1", b_hidden), weights["bt_w1"], weights["bt_b1"], gen_names("bt_g0", b_hidden))
        bt1_in = gen_names("bt_l1", b_hidden)
        lines.append("    bt1_mean = (" + " + ".join(bt1_in) + f") / {b_hidden};")
        bt1_var_terms = [f"(({x}) - bt1_mean)*(({x}) - bt1_mean)" for x in bt1_in]
        lines.append("    bt1_var = (" + " + ".join(bt1_var_terms) + f") / {b_hidden};")
        for i in range(b_hidden):
            lines.append(f"    bt_n1_{i} = (((bt_l1_{i}) - bt1_mean) / sqrt(bt1_var + ln_eps)) * ({fmt(weights['bt_ln1_w'][i])}) + ({fmt(weights['bt_ln1_b'][i])});")
            lines.append(f"    bt_g1_{i} = gelu_approx(bt_n1_{i});")
        emit_linear_block(lines, gen_names("hv", embed), weights["bt_w2"], weights["bt_b2"], gen_names("bt_g1", b_hidden))

        lines.append("    // FiLM fusion + output head")
        emit_linear_block(lines, gen_names("film", 2 * embed), weights["film_w"], weights["film_b"], gen_names("hpn", embed))
        for k in range(embed):
            lines.append(f"    hfp_{k} = hv_{k} * film_{k} + film_{k + embed};")
        hfp_in = gen_names("hfp", embed)
        lines.append("    hf_mean = (" + " + ".join(hfp_in) + f") / {embed};")
        hf_var_terms = [f"(({x}) - hf_mean)*(({x}) - hf_mean)" for x in hfp_in]
        lines.append("    hf_var = (" + " + ".join(hf_var_terms) + f") / {embed};")
        for k in range(embed):
            lines.append(f"    hfn_{k} = ((hfp_{k}) - hf_mean) / sqrt(hf_var + ln_eps);")

        emit_linear_block(lines, gen_names("fh_l0", fh_hidden), weights["fh_w0"], weights["fh_b0"], gen_names("hfn", embed))
        fh0_in = gen_names("fh_l0", fh_hidden)
        lines.append("    fh0_mean = (" + " + ".join(fh0_in) + f") / {fh_hidden};")
        fh0_var_terms = [f"(({x}) - fh0_mean)*(({x}) - fh0_mean)" for x in fh0_in]
        lines.append("    fh0_var = (" + " + ".join(fh0_var_terms) + f") / {fh_hidden};")
        for i in range(fh_hidden):
            lines.append(f"    fh_n0_{i} = (((fh_l0_{i}) - fh0_mean) / sqrt(fh0_var + ln_eps)) * ({fmt(weights['fh_ln0_w'][i])}) + ({fmt(weights['fh_ln0_b'][i])});")
            lines.append(f"    fh_g0_{i} = gelu_approx(fh_n0_{i});")
        emit_linear_block(lines, gen_names("fh_l1", 2), weights["fh_w1"], weights["fh_b1"], gen_names("fh_g0", fh_hidden))

        lines.extend([
            "",
            "    // Inverse target scaling, then inverse Id log10 transform.",
            "    id_scaled = fh_l1_0;",
            f"    id_log10 = id_scaled * ({fmt(self.target_scale[0])}) + ({fmt(self.target_mean[0])});",
            "    id_log10_lim = 0.5*(id_log10 + log10_id_max - sqrt((id_log10 - log10_id_max)*(id_log10 - log10_id_max) + smooth_delta));",
            "    id_log10_lim = 0.5*(id_log10_lim + log10_id_min + sqrt((id_log10_lim - log10_id_min)*(id_log10_lim - log10_id_min) + smooth_delta));",
            "    ids_mag = smooth_pos(limexp(ln(10.0) * id_log10_lim) - id_log_eps, smooth_delta);",
            "    ids = sign_vds * Wid * ids_mag;",
            "",
            "    q_scaled = fh_l1_1;",
            f"    q_ml_raw = q_scaled * ({fmt(self.target_scale[1])}) + ({fmt(self.target_mean[1])});",
            "    qg = Wid * q_scale * q_ml_raw;",
            "    qs = -2.0/3.0 * qg;",
            "    qd = -1.0/3.0 * qg;",
            "",
            "    I(d,s) <+ ids;",
            "    I(d,s) <+ gmin_ds * V(d,s);",
            "    I(g) <+ ddt(qg);",
            "    I(s) <+ ddt(qs);",
            "    I(d) <+ ddt(qd);",
            "  end",
            "endmodule",
        ])

        return "\n".join(lines)
