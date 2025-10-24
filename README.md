# Fully Convolutional Recurrent Networks for Multitemporal Crop Recognition

This repository implements **fully convolutional recurrent networks (FCRN) built on ConvLSTM** for crop‑type recognition from **multitemporal satellite image sequences**. Models combine multi‑resolution spatial encoders (e.g., U‑Net/Dense variants) with **ConvLSTM** to capture temporal dynamics and output **per‑date (N→N) maps**.

- **Main script:** `networks/convlstm_networks/train_src/main.py`
- **Default example model (in this README):** **`BUnetConvLSTM`**  
  In the source, the model is instantiated through a branch like:
  ```python
  if self.model_type=='BUnetConvLSTM':
      ...
  ```
  (around line ~1369 in `main.py`).

Other models mentioned in the paper and supported in code include:
- `BAtrousGAPConvLSTM`
- `DenseNetTimeDistributed`
- `ConvLSTM_seq2seq`
- `ConvLSTM_seq2seq_bi`

> Note: In the current argument defaults, `--model_type` may be set to `'DenseNet'`. The **examples below use `BUnetConvLSTM`** as the recommended starting point.

---

## Installation

Use Python 3.7–3.9 in a fresh environment. Install Keras/TensorFlow and common scientific deps:

```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install --upgrade pip

# Core
pip install tensorflow keras
# Utilities
pip install numpy scikit-learn opencv-python
# (Optional) plotting / IO helpers
pip install matplotlib
```

If the project provides a `requirements.txt`, prefer:
```bash
pip install -r requirements.txt
```

---

## Data format (expected)

The training pipeline expects **co‑registered time series** of imagery per sample (e.g., Sentinel‑1 VV/VH), together with labels compatible with `class_n` classes. Typical tensor shapes:

- Per sample sequence: `(T, C, H, W)` where
  - `T = t_len` (number of dates),
  - `C = channel_n` (bands per date, e.g., 2 for VV/VH).
- Training proceeds on **patches** of size `patch_len × patch_len`, optionally strided by `patch_step_train` (and `patch_step_test` for eval).

The `--path` argument points to your **data root**. Inside, keep or generate the splits and arrays expected by your dataset loader (e.g., NumPy arrays and split lists).

---

## Command‑line arguments (from `argparse`)

| Flag | Type | Default | Meaning |
|---|---:|:---:|---|
| `--t_len` / `-tl` | int | `7` | Number of timesteps (dates) per sequence. |
| `--class_n` / `-cn` | int | `11` | Number of classes in the segmentation. |
| `--channel_n` / `-chn` | int | `2` | Input channels per date (e.g., VV,VH → 2). |
| `--patch_len` / `-pl` | int | `32` | Patch edge size (square). |
| `--patch_step_train` / `-pstr` | int | `32` | Stride for training patch extraction. |
| `--patch_step_test` / `-psts` | int | `None` | Stride for test/val patch extraction (defaults to no overlap if unset). |
| `--debug` / `-db` | int | `1` | Debug level (enables extra prints/shortcuts depending on code). |
| `--epochs` / `-ep` | int | `8000` | Max training epochs. |
| `--patience` / `-pt` | int | `10` | Early‑stopping patience (epochs). |
| `--batch_size_train` / `-bstr` | int | `32` | Training batch size. |
| `--batch_size_test` / `-bsts` | int | `32` | Test/validation batch size. |
| `--eval_mode` / `-em` | str | `metrics` | Evaluation mode: `metrics` or `predict`. |
| `--im_store` / `-is` | bool | `True` | Save predicted images when evaluating (set to `False` to skip). |
| `--exp_id` / `-eid` | str | `default` | Experiment identifier (used for run folders/filenames). |
| `--path` / `-path` | str | `../data/` | Data root directory. |
| `--model_type` / `-mdl` | str | `DenseNet` | Model family to use (see list below). |

### Accepted values for `--model_type`
- `BUnetConvLSTM`  ← *recommended starting point for N→N mapping*
- `BAtrousGAPConvLSTM`
- `DenseNetTimeDistributed`
- `ConvLSTM_seq2seq`
- `ConvLSTM_seq2seq_bi`
- (and any other strings supported in your codebase, such as `DenseNet` for certain baselines)

> Be sure to match the **exact string** expected in the `main.py` conditionals.

---

## Quick‑start: train **BUnetConvLSTM**

```bash
cd networks/convlstm_networks/train_src/
python main.py   --model_type BUnetConvLSTM   --path /data/crops/   --t_len 7   --channel_n 2   --class_n 11   --patch_len 32   --patch_step_train 32   --batch_size_train 32   --epochs 120   --patience 10   --exp_id bunet_s1_seq7_c2
```

### Notes
- Adjust `--t_len`, `--channel_n`, and `--class_n` to your dataset.
- Increase `--patch_len` (e.g., 64/128) to capture more context if memory allows.

---

## Validation & Testing

To run evaluation with metric reporting:
```bash
cd networks/convlstm_networks/train_src/
python main.py   --model_type BUnetConvLSTM   --path /data/crops/   --t_len 7 --channel_n 2 --class_n 11   --patch_len 32 --patch_step_test 32   --batch_size_test 32   --eval_mode metrics   --im_store False   --exp_id bunet_s1_seq7_c2_eval
```

To generate prediction rasters/tiles (no metrics):
```bash
cd networks/convlstm_networks/train_src/
python main.py   --model_type BUnetConvLSTM   --path /data/crops/   --t_len 7 --channel_n 2 --class_n 11   --patch_len 32 --patch_step_test 32   --batch_size_test 32   --eval_mode predict   --im_store True   --exp_id bunet_s1_seq7_c2_pred
```

---

## Switching to other paper models

Just change the `--model_type` flag:

```bash
# Atrous spatial pyramid + GAP + ConvLSTM
--model_type BAtrousGAPConvLSTM

# DenseNet backbone applied per time step with temporal fusion
--model_type DenseNetTimeDistributed

# Seq2seq ConvLSTM baselines (uni- and bi-directional)
--model_type ConvLSTM_seq2seq
--model_type ConvLSTM_seq2seq_bi
```

Hyperparameters like ConvLSTM hidden size, number of layers, or dropout are defined inside the corresponding model constructors. Tune them there

---

## Tips

- **Sentinel‑1 (VV,VH)**: set `--channel_n 2`. If you add incidence angle or coherence, increase accordingly.
- **Sequence length `t_len`** should match the number of acquisitions after your temporal filtering/gap‑filling.
- For class imbalance, consider weighted losses (e.g., weighted categorical cross‑entropy) already present in the codebase.
- Enable early stopping with `--patience` to avoid overfitting on long runs.

---

## Citation

If you use this code or the BUnetConvLSTM/BAtrousGAPConvLSTM/Dense variants, please cite the ISPRS JPRS article and the earlier workshop paper:

- Chamorro Martinez, J.A., et al. (2021) *Fully convolutional recurrent networks for multidate crop recognition from multitemporal image sequences*. ISPRS JPRS 171, 188–201.
- Chamorro Martinez, J.A., et al. (2019) *A Many‑to‑Many Fully Convolutional Recurrent Network for Multitemporal Crop Recognition*. ISPRS Annals IV‑2/W7.

