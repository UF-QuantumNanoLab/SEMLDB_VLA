# SEMLDB Verilog-A Device Models

A collection of ML-based Verilog-A compact models for use in the **Cadence Virtuoso / Spectre** environment.

Each folder contains one device model ready to be imported as a `veriloga` view into your Cadence library.

| Device       | Folder          | Verilog-A file          | Module name   | External data files? |
|--------------|-----------------|-------------------------|---------------|----------------------|
| SiFET        | [SiFET/](SiFET/)           | [sifet.va](SiFET/sifet.va)                     | `SiFETVA`     | No — standalone |
| CNTFET       | [CNTFET/](CNTFET/)         | [cntfet.va](CNTFET/cntfet.va)                  | `CNTFETVA`    | No — standalone |
| HFET         | [HFET/](HFET/)             | [HFET.va](HFET/HFET.va)                        | `HFETVA`      | **Yes** — 32 `.txt` files |
| DiamondFET   | [DiamondFET/](DiamondFET/) | [DiamondFET_Model.va](DiamondFET/DiamondFET_Model.va) | `DiamondFETVA`| **Yes** — 32 `.txt` files |
| NMOS         | [NMOS/](NMOS/)             | [NMOS.va](NMOS/NMOS.va)                        | `NMOSVA`      | **Yes** — 32 `.txt` files |

All models expose three terminals: **d** (drain), **g** (gate), **s** (source).

---

## 1. Standalone models — SiFET and CNTFET

For **SiFET** and **CNTFET** the weights are baked directly into the Verilog-A file. There is nothing else to configure.

### Steps in Cadence
1. In your library, create a new cellview of type **veriloga**.
2. Open the file ([sifet.va](SiFET/sifet.va) or [cntfet.va](CNTFET/cntfet.va)) in a text editor and copy the entire contents into the new cellview.
3. Save and compile. Cadence will create a `symbol` automatically.
4. Instantiate the symbol in your schematic and connect **d**, **g**, **s**.

That's it — no paths to edit, no extra files to ship.

---

## 2. Data-driven models — HFET, DiamondFET, NMOS

These three models load their trained weights and scaling factors at simulation start from a set of `.txt` files that live next to the `.va` file. You must:

1. Copy the Verilog-A file into Cadence (same as above).
2. Update the file paths inside the `.va` file so Cadence can find the `.txt` files on **your** machine.

### 2.1 Where the `.txt` files live

Keep the `.txt` files together with the `.va` file in the folder shipped here (e.g. [HFET/](HFET/)). You can leave the folder on your local disk or on a network share — as long as Cadence has read access to it.

Each device has **32** companion files, for example for HFET:

- `hfetva_idvd_poly_w.txt`, `hfetva_idvd_poly_b.txt`
- `hfetva_idvd_ae_w0.txt` … `hfetva_idvd_ae_w3.txt`, `hfetva_idvd_ae_b0.txt` … `hfetva_idvd_ae_b3.txt`
- `hfetva_idvd_sx_mul.txt`, `hfetva_idvd_sx_add.txt`, `hfetva_idvd_sl_mul.txt`, `hfetva_idvd_sl_add.txt`, `hfetva_idvd_siv_mul.txt`, `hfetva_idvd_siv_add.txt`
- The same 16 files again with `idvg` in place of `idvd`

The DiamondFET and NMOS folders have the same structure, just with `diamondfetva_…` / `nmosva_…` prefixes.

### 2.2 Fix the paths inside the `.va` file

Open the `.va` file and search for `$fopen`. You will see lines such as:

```verilog
fd = $fopen("C:/Users/ta688/UF Dropbox/PC/VerilogA/verilogA_v2/SEMLDB_veriloga_converter/outputs/HFET/hfetva_idvd_poly_w.txt", "r");
```

Replace the path prefix with the **absolute path to the folder on your machine** that holds the `.txt` files. Keep the filename at the end unchanged.

Example — if you placed the HFET folder at `/home/alice/models/HFET/`, the line becomes:

```verilog
fd = $fopen("/home/alice/models/HFET/hfetva_idvd_poly_w.txt", "r");
```

**Tips**
- Use **forward slashes** `/` in the path, even on Windows.
- Update **every** `$fopen` call (32 per device). A find-and-replace on the path prefix is the easiest way.
- The path must be absolute — relative paths are not reliable across Cadence run directories.

### 2.3 Compile and simulate

Recompile the veriloga cellview after editing. On the first simulation step the model reads all `.txt` files once (`@(initial_step)`); if any path is wrong Spectre will print a file-open error and the model will produce zeros.

---

## 3. Instantiating in a schematic

All five models share the same port order:

```
<ModuleName> inst_name (d, g, s);
```

Set the geometry / bias parameters from the symbol properties — defaults are provided at the top of each `.va` file.

---

## 4. Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Drain current is always 0 for HFET / DiamondFET / NMOS | One or more `$fopen` paths point to a file that doesn't exist. Check the Spectre log for `could not open file`. |
| `Cannot find module HFETVA` | The `.va` file wasn't compiled, or the module name in your schematic doesn't match the one in the table above. |
| Convergence issues | Increase `gmin_ds` (top of the `.va` file) or add a small `.options gmin` in your testbench. |
| Paths with spaces (e.g. `Dropbox`) fail to open | Keep the path inside the existing double quotes — do not add extra escaping. Forward slashes are fine on Windows. |
