import os
import re
import sys
from typing import Dict, List

import joblib
import numpy as np
import torch

from architectures.base import ExportError, ModelArchitecture

try:
    from sklearn.preprocessing import PolynomialFeatures
except ImportError:
    PolynomialFeatures = None


def filter_poly_terms(terms, degree):
    return [term for term in terms if sum(term) <= degree or max(term) == sum(term)]


def get_poly_indices(degree, cross_degree, num_features=4):
    if PolynomialFeatures is None:
        raise ExportError("scikit-learn must be installed in this python environment.")
    poly = PolynomialFeatures(degree, include_bias=True)
    poly.fit(np.zeros((1, num_features)))
    return filter_poly_terms(poly.powers_, cross_degree)


def get_transform_params(scaler, inverse=False):
    """
    Return multiplier/addend so the transform is:
      out = inp * multiplier + addend
    """
    if hasattr(scaler, "mean_"):
        if not inverse:
            return 1.0 / scaler.scale_, -scaler.mean_ / scaler.scale_
        return scaler.scale_, scaler.mean_
    if hasattr(scaler, "min_"):
        if not inverse:
            return scaler.scale_, scaler.min_
        return 1.0 / scaler.scale_, -scaler.min_ / scaler.scale_
    raise ExportError("Unknown scaler format")


class DiamondFETArchitecture(ModelArchitecture):
    name = "DiamondFET"
    requires_checkpoint = False

    IDVD_CONFIG = {
        "degree": 11,
        "cross_degree": 9,
    }
    IDVG_CONFIG = {
        "degree": 8,
        "cross_degree": 8,
    }

    def parse_weights(self, state_dict: Dict[str, np.ndarray]) -> None:
        del state_dict
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        candidate_roots = [
            os.path.join(project_root, "checkpoints", "DiamondFET"),
            os.path.join(project_root, "checkpoints"),
            os.path.join(project_root, "checkpoint", "DiamondFET"),
            os.path.join(project_root, "checkpoint"),
        ]
        ckpt_dir = None
        for candidate in candidate_roots:
            idvd_dir = os.path.join(candidate, "idvd_diamond_9_curves_linear")
            idvg_dir = os.path.join(candidate, "idvg_diamond_6_curves_linear_log")
            if os.path.isdir(idvd_dir) and os.path.isdir(idvg_dir):
                ckpt_dir = candidate
                break
        if ckpt_dir is None:
            raise ExportError(f"DiamondFET checkpoint directory not found under: {candidate_roots}")

        idvd_dir = os.path.join(ckpt_dir, "idvd_diamond_9_curves_linear")
        idvg_dir = os.path.join(ckpt_dir, "idvg_diamond_6_curves_linear_log")

        self.idvd_ae = torch.load(
            os.path.join(idvd_dir, "idvd_diamond_9_curves_linear_scale.pth"),
            map_location="cpu",
        )
        self.idvd_poly_model = torch.load(
            os.path.join(idvd_dir, "idvd_diamond_poly_regression_model.pth"),
            map_location="cpu",
        )
        self.idvd_scaler_x = joblib.load(os.path.join(idvd_dir, "idvd_diamond_scaler_x.pkl"))
        self.idvd_scaler_iv = joblib.load(os.path.join(idvd_dir, "idvd_diamond_scaler_iv.pkl"))
        self.idvd_scaler_ls = joblib.load(os.path.join(idvd_dir, "idvd_diamond_scaler_ls.pkl"))
        self.idvd_poly_indices = get_poly_indices(
            self.IDVD_CONFIG["degree"],
            self.IDVD_CONFIG["cross_degree"],
            len(self.idvd_scaler_x.scale_),
        )

        self.idvg_ae = torch.load(
            os.path.join(idvg_dir, "idvg_diamond_6_curves_linear_log_scale.pth"),
            map_location="cpu",
        )
        self.idvg_poly_model = torch.load(
            os.path.join(idvg_dir, "idvg_diamond_poly_regression_model.pth"),
            map_location="cpu",
        )
        self.idvg_scaler_x = joblib.load(os.path.join(idvg_dir, "idvg_diamond_scaler_x.pkl"))
        self.idvg_scaler_iv = joblib.load(os.path.join(idvg_dir, "idvg_diamond_scaler_iv_linear.pkl"))
        self.idvg_scaler_ls = joblib.load(os.path.join(idvg_dir, "idvg_diamond_scaler_ls.pkl"))
        self.idvg_poly_indices = get_poly_indices(
            self.IDVG_CONFIG["degree"],
            self.IDVG_CONFIG["cross_degree"],
            len(self.idvg_scaler_x.scale_),
        )

    def extract_numpy(self, torch_sd):
        np_weights = {}
        for key, value in torch_sd.items():
            if hasattr(value, "detach"):
                np_weights[key] = value.detach().cpu().numpy()
        return np_weights

    def write_array_to_file(self, out_dir, file_name, array):
        path = os.path.join(out_dir, f"{file_name}.txt")
        flat = np.asarray(array, dtype=float).reshape(-1)
        with open(path, "w", encoding="utf-8") as handle:
            for value in flat:
                handle.write(f"{float(value):.9e}\n")
        return path

    def get_out_dir(self):
        if "--out" in sys.argv:
            out_idx = sys.argv.index("--out") + 1
            return os.path.dirname(os.path.abspath(sys.argv[out_idx]))
        return os.path.abspath(".")

    def sanitize_prefix(self, module_name):
        prefix = re.sub(r"[^0-9A-Za-z_]+", "_", module_name).strip("_").lower()
        return prefix or "diamondfet"

    def append_dense_relu(self, lines, in_size, out_size, in_var, out_var, w_name, b_name):
        lines.append(f"      for(i=0; i<{out_size}; i=i+1) begin")
        lines.append(f"        tmp = {b_name}[i];")
        lines.append(f"        for(j=0; j<{in_size}; j=j+1) tmp = tmp + {w_name}[i*{in_size} + j] * {in_var}[j];")
        lines.append(f"        {out_var}[i] = (tmp > 0.0) ? tmp : 0.0;")
        lines.append("      end")

    def emit_model(self, module_name: str) -> str:
        out_dir = self.get_out_dir()
        os.makedirs(out_dir, exist_ok=True)
        file_prefix = self.sanitize_prefix(module_name)

        idvd_poly_sd = self.extract_numpy(self.idvd_poly_model)
        idvd_ae_sd = self.extract_numpy(self.idvd_ae)
        idvg_poly_sd = self.extract_numpy(self.idvg_poly_model)
        idvg_ae_sd = self.extract_numpy(self.idvg_ae)

        idvd_out_len = int(self.idvd_scaler_iv.scale_.shape[0])
        idvd_latent = int(idvd_poly_sd["linear.bias"].shape[0])
        idvd_h0 = int(idvd_ae_sd["decoder.0.bias"].shape[0])
        idvd_h1 = int(idvd_ae_sd["decoder.2.bias"].shape[0])
        idvd_h2 = int(idvd_ae_sd["decoder.4.bias"].shape[0])
        idvd_poly_len = len(self.idvd_poly_indices)

        idvg_curve_len = int(self.idvg_scaler_iv.scale_.shape[0])
        idvg_latent = int(idvg_poly_sd["linear.bias"].shape[0])
        idvg_h0 = int(idvg_ae_sd["decoder.0.bias"].shape[0])
        idvg_h1 = int(idvg_ae_sd["decoder.2.bias"].shape[0])
        idvg_h2 = int(idvg_ae_sd["decoder.4.bias"].shape[0])
        idvg_poly_len = len(self.idvg_poly_indices)

        idvd_files = {
            "idvd_poly_w": idvd_poly_sd["linear.weight"],
            "idvd_poly_b": idvd_poly_sd["linear.bias"],
            "idvd_ae_w0": idvd_ae_sd["decoder.0.weight"],
            "idvd_ae_b0": idvd_ae_sd["decoder.0.bias"],
            "idvd_ae_w1": idvd_ae_sd["decoder.2.weight"],
            "idvd_ae_b1": idvd_ae_sd["decoder.2.bias"],
            "idvd_ae_w2": idvd_ae_sd["decoder.4.weight"],
            "idvd_ae_b2": idvd_ae_sd["decoder.4.bias"],
            "idvd_ae_w3": idvd_ae_sd["decoder.6.weight"],
            "idvd_ae_b3": idvd_ae_sd["decoder.6.bias"],
        }
        idvg_files = {
            "idvg_poly_w": idvg_poly_sd["linear.weight"],
            "idvg_poly_b": idvg_poly_sd["linear.bias"],
            "idvg_ae_w0": idvg_ae_sd["decoder.0.weight"],
            "idvg_ae_b0": idvg_ae_sd["decoder.0.bias"],
            "idvg_ae_w1": idvg_ae_sd["decoder.2.weight"],
            "idvg_ae_b1": idvg_ae_sd["decoder.2.bias"],
            "idvg_ae_w2": idvg_ae_sd["decoder.4.weight"],
            "idvg_ae_b2": idvg_ae_sd["decoder.4.bias"],
            # Only the linear Id-Vg branch is required for the drain-current model.
            "idvg_ae_w3": idvg_ae_sd["decoder.6.weight"][:idvg_curve_len, :],
            "idvg_ae_b3": idvg_ae_sd["decoder.6.bias"][:idvg_curve_len],
        }

        def add_scaler_arrays(scaler, inverse, prefix):
            mul, add = get_transform_params(scaler, inverse)
            return {
                f"{prefix}_mul": np.asarray(mul, dtype=float).reshape(-1),
                f"{prefix}_add": np.asarray(add, dtype=float).reshape(-1),
            }

        scaler_arrays = {}
        scaler_arrays.update(add_scaler_arrays(self.idvd_scaler_x, False, "idvd_sx"))
        scaler_arrays.update(add_scaler_arrays(self.idvd_scaler_ls, True, "idvd_sl"))
        scaler_arrays.update(add_scaler_arrays(self.idvd_scaler_iv, True, "idvd_siv"))
        scaler_arrays.update(add_scaler_arrays(self.idvg_scaler_x, False, "idvg_sx"))
        scaler_arrays.update(add_scaler_arrays(self.idvg_scaler_ls, True, "idvg_sl"))
        scaler_arrays.update(add_scaler_arrays(self.idvg_scaler_iv, True, "idvg_siv"))

        all_arrays = {**idvd_files, **idvg_files, **scaler_arrays}

        lines: List[str] = [
            "// Automatically generated Verilog-A DiamondFET drain-current model",
            '`include "constants.vams"',
            '`include "disciplines.vams"',
            "",
            f"module {module_name}(d, g, s);",
            "  inout d, g, s;",
            "  electrical d, g, s;",
            "",
            "  // Geometry parameters follow the ML front-end convention:",
            "  // Lg/Lgs/Tox are provided in nm, Rc in ohm*mm.",
            "  // The backend assumes Lgd = Lgs, so only a single gate-to-contact spacing is exposed.",
            "  parameter real Lg = 400.0;",
            "  parameter real Lgs = 900.0;",
            "  parameter real Tox = 17.0;",
            "  parameter real Rc = 0.90;",
            "  // 0: blended current, 1: Id-Vd surface only, 2: Id-Vg surface only",
            "  parameter integer current_mode = 0;",
            "  parameter real gmin_ds = 1e-12 from [0:inf);",
            "  parameter real vds_sign_delta = 1e-4 from [0:inf);",
            "",
        ]

        for array_name, array in all_arrays.items():
            lines.append(f"  real {array_name}[0:{len(np.asarray(array).reshape(-1)) - 1}];")

        lines.extend(
            [
                "",
                "  // Temporary buffers",
                "  real input_x[0:3];",
                f"  real idvd_poly_feat[0:{idvd_poly_len - 1}];",
                f"  real idvd_z[0:{idvd_latent - 1}];",
                f"  real idvd_h0[0:{idvd_h0 - 1}];",
                f"  real idvd_h1[0:{idvd_h1 - 1}];",
                f"  real idvd_h2[0:{idvd_h2 - 1}];",
                f"  real idvd_out[0:{idvd_out_len - 1}];",
                f"  real idvg_poly_feat[0:{idvg_poly_len - 1}];",
                f"  real idvg_z[0:{idvg_latent - 1}];",
                f"  real idvg_h0[0:{idvg_h0 - 1}];",
                f"  real idvg_h1[0:{idvg_h1 - 1}];",
                f"  real idvg_h2[0:{idvg_h2 - 1}];",
                f"  real idvg_out[0:{idvg_curve_len - 1}];",
                "  integer i, j, fd, count_tmp;",
                "  integer idvd_vg0_idx, idvd_vg1_idx, idvd_vd0_idx, idvd_vd1_idx;",
                "  integer idvg_vg0_idx, idvg_vg1_idx, idvg_vd0_idx, idvg_vd1_idx;",
                "  real tmp, val_vg, val_vd, vd_mag, sign_vd, res_id_ref;",
                "  real idvd_vg_pos, idvd_vg_frac, idvd_vd_pos, idvd_vd_frac;",
                "  real idvg_vg_pos, idvg_vg_frac, idvg_vd_pos, idvg_vd_frac;",
                "  real idvd_i00, idvd_i01, idvd_i10, idvd_i11, idvd_interp;",
                "  real idvg_i00, idvg_i01, idvg_i10, idvg_i11, idvg_interp;",
                "  real idvd_vg_dist, idvg_vd_dist, idvd_excess, idvg_excess;",
                "  real w_idvd, w_idvg, w_sum;",
                "",
                "  analog begin",
                "    @(initial_step) begin",
            ]
        )

        for array_name, array in all_arrays.items():
            file_name = f"{file_prefix}_{array_name}"
            read_path = self.write_array_to_file(out_dir, file_name, array).replace("\\", "/")
            array_len = len(np.asarray(array).reshape(-1))
            lines.append(f'      fd = $fopen("{read_path}", "r");')
            lines.append(f'      for(i=0; i<{array_len}; i=i+1) count_tmp = $fscanf(fd, "%e", {array_name}[i]);')
            lines.append("      $fclose(fd);")

        lines.extend(
            [
                "",
                "      // Convert nm -> um before applying the learned input scalers.",
                "      input_x[0] = (1.0e-3 * Lg) * idvd_sx_mul[0] + idvd_sx_add[0];",
                "      input_x[1] = (1.0e-3 * Lgs) * idvd_sx_mul[1] + idvd_sx_add[1];",
                "      input_x[2] = (1.0e-3 * Tox) * idvd_sx_mul[2] + idvd_sx_add[2];",
                "      input_x[3] = Rc * idvd_sx_mul[3] + idvd_sx_add[3];",
            ]
        )

        for idx, powers in enumerate(self.idvd_poly_indices):
            terms = [f"pow(input_x[{dim}], {exp})" for dim, exp in enumerate(powers) if exp > 0]
            lines.append(f"      idvd_poly_feat[{idx}] = {' * '.join(terms) if terms else '1.0'};")

        lines.extend(
            [
                f"      for(i=0; i<{idvd_latent}; i=i+1) begin",
                "        tmp = idvd_poly_b[i];",
                f"        for(j=0; j<{idvd_poly_len}; j=j+1) tmp = tmp + idvd_poly_w[i*{idvd_poly_len} + j] * idvd_poly_feat[j];",
                "        idvd_z[i] = tmp * idvd_sl_mul[i] + idvd_sl_add[i];",
                "      end",
            ]
        )
        self.append_dense_relu(lines, idvd_latent, idvd_h0, "idvd_z", "idvd_h0", "idvd_ae_w0", "idvd_ae_b0")
        self.append_dense_relu(lines, idvd_h0, idvd_h1, "idvd_h0", "idvd_h1", "idvd_ae_w1", "idvd_ae_b1")
        self.append_dense_relu(lines, idvd_h1, idvd_h2, "idvd_h1", "idvd_h2", "idvd_ae_w2", "idvd_ae_b2")
        lines.extend(
            [
                f"      for(i=0; i<{idvd_out_len}; i=i+1) begin",
                "        tmp = idvd_ae_b3[i];",
                f"        for(j=0; j<{idvd_h2}; j=j+1) tmp = tmp + idvd_ae_w3[i*{idvd_h2} + j] * idvd_h2[j];",
                "        tmp = 1.0 / (1.0 + exp(-tmp));",
                "        idvd_out[i] = -1.0 * (tmp * idvd_siv_mul[i] + idvd_siv_add[i]);",
                "      end",
                "",
                "      input_x[0] = (1.0e-3 * Lg) * idvg_sx_mul[0] + idvg_sx_add[0];",
                "      input_x[1] = (1.0e-3 * Lgs) * idvg_sx_mul[1] + idvg_sx_add[1];",
                "      input_x[2] = (1.0e-3 * Tox) * idvg_sx_mul[2] + idvg_sx_add[2];",
                "      input_x[3] = Rc * idvg_sx_mul[3] + idvg_sx_add[3];",
            ]
        )

        for idx, powers in enumerate(self.idvg_poly_indices):
            terms = [f"pow(input_x[{dim}], {exp})" for dim, exp in enumerate(powers) if exp > 0]
            lines.append(f"      idvg_poly_feat[{idx}] = {' * '.join(terms) if terms else '1.0'};")

        lines.extend(
            [
                f"      for(i=0; i<{idvg_latent}; i=i+1) begin",
                "        tmp = idvg_poly_b[i];",
                f"        for(j=0; j<{idvg_poly_len}; j=j+1) tmp = tmp + idvg_poly_w[i*{idvg_poly_len} + j] * idvg_poly_feat[j];",
                "        idvg_z[i] = tmp * idvg_sl_mul[i] + idvg_sl_add[i];",
                "      end",
            ]
        )
        self.append_dense_relu(lines, idvg_latent, idvg_h0, "idvg_z", "idvg_h0", "idvg_ae_w0", "idvg_ae_b0")
        self.append_dense_relu(lines, idvg_h0, idvg_h1, "idvg_h0", "idvg_h1", "idvg_ae_w1", "idvg_ae_b1")
        self.append_dense_relu(lines, idvg_h1, idvg_h2, "idvg_h1", "idvg_h2", "idvg_ae_w2", "idvg_ae_b2")
        lines.extend(
            [
                f"      for(i=0; i<{idvg_curve_len}; i=i+1) begin",
                "        tmp = idvg_ae_b3[i];",
                f"        for(j=0; j<{idvg_h2}; j=j+1) tmp = tmp + idvg_ae_w3[i*{idvg_h2} + j] * idvg_h2[j];",
                "        tmp = 1.0 / (1.0 + exp(-tmp));",
                "        idvg_out[i] = -1.0 * (tmp * idvg_siv_mul[i] + idvg_siv_add[i]);",
                "      end",
                "    end",
                "",
                "    val_vg = V(g, s);",
                "    val_vd = V(d, s);",
                "    vd_mag = sqrt(val_vd*val_vd + vds_sign_delta*vds_sign_delta);",
                "    sign_vd = val_vd / vd_mag;",
                "",
                "    // The ML data was trained for negative drain bias. Use |Vds| for the",
                "    // surface lookup and apply the drain-current sign afterwards.",
                "",
                "    // Id-Vd surface: 9 Vg anchors x 120 Vd samples",
                "    if(val_vg <= -5.0) begin",
                "      idvd_vg0_idx = 0;",
                "      idvd_vg1_idx = 1;",
                "      idvd_vg_frac = 0.0;",
                "      idvd_excess = -5.0 - val_vg;",
                "    end else if(val_vg >= 11.0) begin",
                "      idvd_vg0_idx = 7;",
                "      idvd_vg1_idx = 8;",
                "      idvd_vg_frac = 1.0;",
                "      idvd_excess = val_vg - 11.0;",
                "    end else begin",
                "      idvd_vg_pos = (val_vg + 5.0) / 2.0;",
                "      idvd_vg0_idx = $rtoi(idvd_vg_pos);",
                "      if(idvd_vg0_idx > 7) idvd_vg0_idx = 7;",
                "      idvd_vg1_idx = idvd_vg0_idx + 1;",
                "      idvd_vg_frac = idvd_vg_pos - idvd_vg0_idx;",
                "      idvd_excess = 0.0;",
                "    end",
                "",
                "    if(vd_mag >= 12.0) begin",
                "      idvd_vd0_idx = 0;",
                "      idvd_vd1_idx = 1;",
                "      idvd_vd_frac = 0.0;",
                "    end else if(vd_mag > 0.02) begin",
                "      idvd_vd_pos = (12.0 - vd_mag) / ((12.0 - 0.02) / 118.0);",
                "      idvd_vd0_idx = $rtoi(idvd_vd_pos);",
                "      if(idvd_vd0_idx > 117) idvd_vd0_idx = 117;",
                "      idvd_vd1_idx = idvd_vd0_idx + 1;",
                "      idvd_vd_frac = idvd_vd_pos - idvd_vd0_idx;",
                "    end else begin",
                "      idvd_vd0_idx = 118;",
                "      idvd_vd1_idx = 119;",
                "      idvd_vd_frac = (0.02 - vd_mag) / 0.02;",
                "    end",
                "",
                "    idvd_i00 = idvd_out[idvd_vg0_idx * 120 + idvd_vd0_idx];",
                "    idvd_i01 = idvd_out[idvd_vg0_idx * 120 + idvd_vd1_idx];",
                "    idvd_i10 = idvd_out[idvd_vg1_idx * 120 + idvd_vd0_idx];",
                "    idvd_i11 = idvd_out[idvd_vg1_idx * 120 + idvd_vd1_idx];",
                "    idvd_interp = (1.0 - idvd_vg_frac) * ((1.0 - idvd_vd_frac) * idvd_i00 + idvd_vd_frac * idvd_i01)",
                "                + idvd_vg_frac * ((1.0 - idvd_vd_frac) * idvd_i10 + idvd_vd_frac * idvd_i11);",
                "",
                "    // Id-Vg surface: 6 Vd anchors x 201 Vg samples",
                "    if(val_vg <= -5.0) begin",
                "      idvg_vg0_idx = 0;",
                "      idvg_vg1_idx = 1;",
                "      idvg_vg_frac = 0.0;",
                "    end else if(val_vg >= 12.0) begin",
                "      idvg_vg0_idx = 199;",
                "      idvg_vg1_idx = 200;",
                "      idvg_vg_frac = 1.0;",
                "    end else begin",
                "      idvg_vg_pos = (val_vg + 5.0) / (17.0 / 200.0);",
                "      idvg_vg0_idx = $rtoi(idvg_vg_pos);",
                "      if(idvg_vg0_idx > 199) idvg_vg0_idx = 199;",
                "      idvg_vg1_idx = idvg_vg0_idx + 1;",
                "      idvg_vg_frac = idvg_vg_pos - idvg_vg0_idx;",
                "    end",
                "",
                "    if(vd_mag >= 11.0) begin",
                "      idvg_vd0_idx = 0;",
                "      idvg_vd1_idx = 1;",
                "      idvg_vd_frac = 0.0;",
                "      idvg_excess = vd_mag - 11.0;",
                "    end else if(vd_mag <= 1.0) begin",
                "      idvg_vd0_idx = 4;",
                "      idvg_vd1_idx = 5;",
                "      idvg_vd_frac = 1.0;",
                "      idvg_excess = 1.0 - vd_mag;",
                "    end else begin",
                "      idvg_vd_pos = (11.0 - vd_mag) / 2.0;",
                "      idvg_vd0_idx = $rtoi(idvg_vd_pos);",
                "      if(idvg_vd0_idx > 4) idvg_vd0_idx = 4;",
                "      idvg_vd1_idx = idvg_vd0_idx + 1;",
                "      idvg_vd_frac = idvg_vd_pos - idvg_vd0_idx;",
                "      idvg_excess = 0.0;",
                "    end",
                "",
                "    idvg_i00 = idvg_out[idvg_vd0_idx * 201 + idvg_vg0_idx];",
                "    idvg_i01 = idvg_out[idvg_vd0_idx * 201 + idvg_vg1_idx];",
                "    idvg_i10 = idvg_out[idvg_vd1_idx * 201 + idvg_vg0_idx];",
                "    idvg_i11 = idvg_out[idvg_vd1_idx * 201 + idvg_vg1_idx];",
                "    idvg_interp = (1.0 - idvg_vd_frac) * ((1.0 - idvg_vg_frac) * idvg_i00 + idvg_vg_frac * idvg_i01)",
                "                + idvg_vd_frac * ((1.0 - idvg_vg_frac) * idvg_i10 + idvg_vg_frac * idvg_i11);",
                "",
                "    if(idvd_vg_frac < 0.5) idvd_vg_dist = 2.0 * idvd_vg_frac;",
                "    else idvd_vg_dist = 2.0 * (1.0 - idvd_vg_frac);",
                "    if(idvg_vd_frac < 0.5) idvg_vd_dist = 2.0 * idvg_vd_frac;",
                "    else idvg_vd_dist = 2.0 * (1.0 - idvg_vd_frac);",
                "",
                "    w_idvd = 1.0 / (0.05 + (idvd_vg_dist + 2.0 * idvd_excess) * (idvd_vg_dist + 2.0 * idvd_excess));",
                "    w_idvg = 1.0 / (0.05 + (idvg_vd_dist + 2.0 * idvg_excess) * (idvg_vd_dist + 2.0 * idvg_excess));",
                "",
                "    if(current_mode == 1) begin",
                "      res_id_ref = idvd_interp;",
                "    end else if(current_mode == 2) begin",
                "      res_id_ref = idvg_interp;",
                "    end else begin",
                "      w_sum = w_idvd + w_idvg;",
                "      if(w_sum > 0.0) res_id_ref = (w_idvd * idvd_interp + w_idvg * idvg_interp) / w_sum;",
                "      else res_id_ref = 0.5 * (idvd_interp + idvg_interp);",
                "    end",
                "",
                "    I(d, s) <+ -sign_vd * res_id_ref;",
                "    I(d, s) <+ gmin_ds * V(d, s);",
                "  end",
                "endmodule",
            ]
        )

        return "\n".join(lines)
