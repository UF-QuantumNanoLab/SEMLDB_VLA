import numpy as np
from typing import Dict, List
from architectures.base import (
    ModelArchitecture, ExportError, fmt, require, gen_names, emit_linear_block
)

class SiFETArchitecture(ModelArchitecture):
    name = "SiFET"

    def parse_weights(self, sd: Dict[str, np.ndarray]) -> None:
        self.weights = {}
        emb = sd.get("two_tower.device_type_embeddings.weight")
        if emb is None:
            raise ExportError("Missing embedding weight: two_tower.device_type_embeddings.weight")
        num_device_features, embed = emb.shape
        self.embed_dim = embed
        
        self.weights["emb"] = emb
        self.weights["ds_w0"] = require(sd, "two_tower.device_shared_mlp.0.weight", (2 * embed, 1 + embed))
        self.weights["ds_b0"] = require(sd, "two_tower.device_shared_mlp.0.bias", (2 * embed,))
        self.weights["ds_ln0_w"] = require(sd, "two_tower.device_shared_mlp.1.weight", (2 * embed,))
        self.weights["ds_ln0_b"] = require(sd, "two_tower.device_shared_mlp.1.bias", (2 * embed,))
        self.weights["ds_w1"] = require(sd, "two_tower.device_shared_mlp.3.weight", (embed, 2 * embed))
        self.weights["ds_b1"] = require(sd, "two_tower.device_shared_mlp.3.bias", (embed,))
        
        self.weights["bt_w0"] = require(sd, "two_tower.bias_mlp.0.weight", (4 * embed, 2))
        self.weights["bt_b0"] = require(sd, "two_tower.bias_mlp.0.bias", (4 * embed,))
        self.weights["bt_ln0_w"] = require(sd, "two_tower.bias_mlp.1.weight", (4 * embed,))
        self.weights["bt_ln0_b"] = require(sd, "two_tower.bias_mlp.1.bias", (4 * embed,))
        self.weights["bt_w1"] = require(sd, "two_tower.bias_mlp.3.weight", (4 * embed, 4 * embed))
        self.weights["bt_b1"] = require(sd, "two_tower.bias_mlp.3.bias", (4 * embed,))
        self.weights["bt_ln1_w"] = require(sd, "two_tower.bias_mlp.4.weight", (4 * embed,))
        self.weights["bt_ln1_b"] = require(sd, "two_tower.bias_mlp.4.bias", (4 * embed,))
        self.weights["bt_w2"] = require(sd, "two_tower.bias_mlp.6.weight", (embed, 4 * embed))
        self.weights["bt_b2"] = require(sd, "two_tower.bias_mlp.6.bias", (embed,))

        self.weights["film_w"] = require(sd, "two_tower.film_projection.weight", (2 * embed, embed))
        self.weights["film_b"] = require(sd, "two_tower.film_projection.bias", (2 * embed,))

        self.weights["fh_w0"] = require(sd, "two_tower.output_head.0.weight", (2 * embed, embed))
        self.weights["fh_b0"] = require(sd, "two_tower.output_head.0.bias", (2 * embed,))
        self.weights["fh_ln0_w"] = require(sd, "two_tower.output_head.1.weight", (2 * embed,))
        self.weights["fh_ln0_b"] = require(sd, "two_tower.output_head.1.bias", (2 * embed,))
        self.weights["fh_w1"] = require(sd, "two_tower.output_head.3.weight", (3, 2 * embed))
        self.weights["fh_b1"] = require(sd, "two_tower.output_head.3.bias", (3,))

        self.weights["dh_w0"] = require(sd, "device_head.0.weight", (2 * embed, embed))
        self.weights["dh_b0"] = require(sd, "device_head.0.bias", (2 * embed,))
        self.weights["dh_ln0_w"] = require(sd, "device_head.1.weight", (2 * embed,))
        self.weights["dh_ln0_b"] = require(sd, "device_head.1.bias", (2 * embed,))
        self.weights["dh_w1"] = require(sd, "device_head.3.weight", (4, 2 * embed))
        self.weights["dh_b1"] = require(sd, "device_head.3.bias", (4,))

    def emit_model(self, module_name: str) -> str:
        embed = self.embed_dim
        d_hidden = 2 * embed
        b_hidden = 4 * embed
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
            "  parameter real tox = 2.0;",
            "  parameter real Lg = 10.0;",
            "  parameter real eps_ox = 4.0;",
            "  parameter real V_th = 0.20;",
            "",
            "  // Numerical constants",
            "  parameter real ln_eps = 1e-5;",
            "  parameter real smooth_delta = 1e-12;",
            "  parameter real vds_floor = 1e-6;",
            "  parameter real vds_sign_smooth = 1e-3;",
            "  parameter real exp_arg_limit = 80.0;",
            "  parameter real log_id_min = -120.0;",
            "  parameter real log_id_max = 80.0;",
            "  parameter real gmin_ds = 1e-12;",
            "  parameter real Wid = 1.0 from [0:inf);",
            "  parameter real qraw_to_c = 1.0e-10 from [0:inf);",
            "  parameter real qdensity_scale = 1.0 from [0:inf);",
            "",
            "  // Internal scalars",
            "  real vgs, vds, vgs_shift, vds_pos, vds_abs, sign_vds;",
            "  real tox_eff, lg_eff, eps_eff, dev0, dev1, dev2;",
            "  real n_raw, v0_raw, beta_raw, n_par, v0_par, b_par, vt_pred, vdsat_raw, vdsat;",
            "  real cinv, f_softplus, log_vds, log_vdsat, lse, log_id, log_id_lim, ids_mag, ids;",
            "  real tmp_a, tmp_b;",
            "  real q_ml_raw, qg, qs, qd;",
            "  real mean_tmp, var_tmp;",
            "",
            "  analog function real smooth_pos;",
            "    input x, dlt;",
            "    real x, dlt;",
            "    begin",
            "      smooth_pos = 0.5*(x + sqrt(x*x + dlt*dlt));",
            "    end",
            "  endfunction",
            "",
            "  analog function real smooth_clip;",
            "    input x, limit;",
            "    real x, limit;",
            "    begin",
            "      smooth_clip = limit * tanh(x / limit);",
            "    end",
            "  endfunction",
            "",
            "  analog function real sigmoid_lim;",
            "    input x;",
            "    real x;",
            "    begin",
            "      sigmoid_lim = 1.0 / (1.0 + exp(-x));",
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
            "  analog function real logsumexp2_smooth;",
            "    input a, b, dlt;",
            "    real a, b, dlt;",
            "    real m;",
            "    begin",
            "      m = 0.5*(a + b + sqrt((a-b)*(a-b) + dlt));",
            "      logsumexp2_smooth = m + ln(exp(a-m) + exp(b-m));",
            "    end",
            "  endfunction",
            "",
        ]

        # Declare variables for blocks
        # Device tower tokens
        for tok in range(3):
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

        for name in gen_names("hp", embed): lines.append(f"  real {name};")
        for name in gen_names("hpn", embed): lines.append(f"  real {name};")
        lines.append("  real hp_mean;")
        lines.append("  real hp_var;")
        lines.append("")

        for prefix, n in [
            ("bt_l0", b_hidden), ("bt_n0", b_hidden), ("bt_g0", b_hidden),
            ("bt_l1", b_hidden), ("bt_n1", b_hidden), ("bt_g1", b_hidden),
            ("hv", embed),
        ]:
            for name in gen_names(prefix, n): lines.append(f"  real {name};")
        lines.append("  real bt0_mean;\n  real bt0_var;\n  real bt1_mean;\n  real bt1_var;\n")

        for prefix, n in [
            ("dh_l0", d_hidden), ("dh_n0", d_hidden),
            ("dh_g0", d_hidden), ("dh_l1", 4),
        ]:
            for name in gen_names(prefix, n): lines.append(f"  real {name};")
        lines.append("  real dh0_mean;\n  real dh0_var;\n")

        for prefix, n in [
            ("film", 2 * embed), ("hfp", embed), ("hfn", embed),
            ("fh_l0", d_hidden), ("fh_n0", d_hidden), ("fh_g0", d_hidden), ("fh_l1", 3),
        ]:
            for name in gen_names(prefix, n): lines.append(f"  real {name};")
        lines.append("  real hf_mean;\n  real hf_var;\n  real fh0_mean;\n  real fh0_var;\n")

        lines.extend([
            "  analog begin",
            "    vgs = V(g,s);",
            "    vds = V(d,s);",
            "    vgs_shift = vgs - (V_th - 0.20);",
            "    vds_abs = sqrt(vds*vds + vds_floor*vds_floor);",
            "    vds_pos = smooth_pos(vds_abs, vds_floor);",
            "    sign_vds = vds / sqrt(vds*vds + vds_sign_smooth*vds_sign_smooth);",
            "",
            "    tox_eff = smooth_pos(tox, smooth_delta);",
            "    lg_eff = smooth_pos(Lg, smooth_delta);",
            "    eps_eff = smooth_pos(eps_ox, smooth_delta);",
            "    dev0 = tox_eff / 10.0;",
            "    dev1 = lg_eff / 100.0;",
            "    dev2 = eps_eff / 100.0;",
            "",
        ])

        # Logic extracted verbatim to mimic exact outputs
        dev_inputs = ["dev0", "dev1", "dev2"]
        for tok in range(3):
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
            
            emit_linear_block(lines, gen_names(f"dt{tok}_l1", embed), weights["ds_w1"], weights["ds_b1"], gen_names(f"dt{tok}_g0", d_hidden))

        lines.append("    // Device token pooling")
        for k in range(embed):
            lines.append(f"    hp_{k} = (dt0_l1_{k} + dt1_l1_{k} + dt2_l1_{k}) / 3.0;")
        hp_in = gen_names("hp", embed)
        lines.append("    hp_mean = (" + " + ".join(hp_in) + f") / {embed};")
        hp_var_terms = [f"(({x}) - hp_mean)*(({x}) - hp_mean)" for x in hp_in]
        lines.append("    hp_var = (" + " + ".join(hp_var_terms) + f") / {embed};")
        for k in range(embed):
            lines.append(f"    hpn_{k} = ((hp_{k}) - hp_mean) / sqrt(hp_var + ln_eps);")

        lines.append("    // Bias tower")
        emit_linear_block(lines, gen_names("bt_l0", b_hidden), weights["bt_w0"], weights["bt_b0"], ["vgs_shift", "vds_pos"])
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

        lines.append("    // Device-only head: n, v0, beta")
        emit_linear_block(lines, gen_names("dh_l0", d_hidden), weights["dh_w0"], weights["dh_b0"], gen_names("hpn", embed))
        dh0_in = gen_names("dh_l0", d_hidden)
        lines.append("    dh0_mean = (" + " + ".join(dh0_in) + f") / {d_hidden};")
        dh0_var_terms = [f"(({x}) - dh0_mean)*(({x}) - dh0_mean)" for x in dh0_in]
        lines.append("    dh0_var = (" + " + ".join(dh0_var_terms) + f") / {d_hidden};")
        for i in range(d_hidden):
            lines.append(f"    dh_n0_{i} = (((dh_l0_{i}) - dh0_mean) / sqrt(dh0_var + ln_eps)) * ({fmt(weights['dh_ln0_w'][i])}) + ({fmt(weights['dh_ln0_b'][i])});")
            lines.append(f"    dh_g0_{i} = gelu_approx(dh_n0_{i});")
        emit_linear_block(lines, gen_names("dh_l1", 4), weights["dh_w1"], weights["dh_b1"], gen_names("dh_g0", d_hidden))
        
        lines.extend([
            "    n_raw = dh_l1_0;",
            "    v0_raw = dh_l1_1;",
            "    beta_raw = dh_l1_3;",
            "    n_par = limexp(smooth_clip(n_raw, exp_arg_limit));",
            "    v0_par = limexp(smooth_clip(v0_raw, exp_arg_limit));",
            "    b_par = 1.4 + 1.1*sigmoid_lim(beta_raw);",
            "",
        ])

        lines.append("    // FiLM fusion + full head")
        emit_linear_block(lines, gen_names("film", 2 * embed), weights["film_w"], weights["film_b"], gen_names("hpn", embed))
        for k in range(embed):
            lines.append(f"    hfp_{k} = hv_{k} * film_{k} + film_{k + embed};")
        hfp_in = gen_names("hfp", embed)
        lines.append("    hf_mean = (" + " + ".join(hfp_in) + f") / {embed};")
        hf_var_terms = [f"(({x}) - hf_mean)*(({x}) - hf_mean)" for x in hfp_in]
        lines.append("    hf_var = (" + " + ".join(hf_var_terms) + f") / {embed};")
        for k in range(embed):
            lines.append(f"    hfn_{k} = ((hfp_{k}) - hf_mean) / sqrt(hf_var + ln_eps);")

        emit_linear_block(lines, gen_names("fh_l0", d_hidden), weights["fh_w0"], weights["fh_b0"], gen_names("hfn", embed))
        fh0_in = gen_names("fh_l0", d_hidden)
        lines.append("    fh0_mean = (" + " + ".join(fh0_in) + f") / {d_hidden};")
        fh0_var_terms = [f"(({x}) - fh0_mean)*(({x}) - fh0_mean)" for x in fh0_in]
        lines.append("    fh0_var = (" + " + ".join(fh0_var_terms) + f") / {d_hidden};")
        for i in range(d_hidden):
            lines.append(f"    fh_n0_{i} = (((fh_l0_{i}) - fh0_mean) / sqrt(fh0_var + ln_eps)) * ({fmt(weights['fh_ln0_w'][i])}) + ({fmt(weights['fh_ln0_b'][i])});")
            lines.append(f"    fh_g0_{i} = gelu_approx(fh_n0_{i});")
        emit_linear_block(lines, gen_names("fh_l1", 3), weights["fh_w1"], weights["fh_b1"], gen_names("fh_g0", d_hidden))
        
        lines.extend([
            "    vt_pred = fh_l1_1;",
            "    vdsat_raw = fh_l1_2;",
            "    vdsat = 0.01 + 0.39 * sigmoid_lim(vdsat_raw);",
            "    q_ml_raw = fh_l1_0;",
            "    qg = Wid * qdensity_scale * qraw_to_c * q_ml_raw;",
            "    qs = -2.0/3.0 * qg;",
            "    qd = -1.0/3.0 * qg;",
            "",
            "    // Physics / math combo",
            "    cinv = 1.0e12 * eps_eff * 8.854e-12 / tox_eff;",
            "    f_softplus = ln(1.0 + limexp((vgs_shift - vt_pred) / (n_par * 0.026)));",
            "    log_vds = ln(vds_pos);",
            "    log_vdsat = ln(smooth_pos(vdsat, smooth_delta));",
            "    tmp_a = b_par * log_vdsat;",
            "    tmp_b = b_par * log_vds;",
            "    lse = logsumexp2_smooth(tmp_a, tmp_b, smooth_delta);",
            "    log_id = ln(smooth_pos(cinv * n_par * 0.026, smooth_delta))",
            "             + ln(smooth_pos(f_softplus, smooth_delta))",
            "             + ln(smooth_pos(v0_par, smooth_delta))",
            "             + log_vds",
            "             - (1.0 / b_par) * lse;",
            "",
            "    log_id_lim = 0.5*(log_id + log_id_max - sqrt((log_id - log_id_max)*(log_id - log_id_max) + smooth_delta));",
            "    log_id_lim = 0.5*(log_id_lim + log_id_min + sqrt((log_id_lim - log_id_min)*(log_id_lim - log_id_min) + smooth_delta));",
            "",
            "    ids_mag = 1.0e-3 * limexp(log_id_lim);",
            "    ids = sign_vds * Wid * ids_mag;",
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
