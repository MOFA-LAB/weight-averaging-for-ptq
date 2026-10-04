# Weight Averaging for Post-Training Quantization

Code for **Understanding the Weight Averaging Mechanism in LLM Training for Post-Training Quantization**.

Run all commands from the repository root; the examples use OPT-125M.

## 1. Environment

Use Python 3.10–3.12 on Linux with four BF16-capable NVIDIA GPUs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## 2. Configuration



| Stage | Fields to edit |
| --- | --- |
| Training | `project.output_dir`, `data.root`, `data.preparation.raw_dir`, `runtime.gpu_ids` |
| Averaging | `input.stable_dir`, `output.source_run_dir`, `output.run_dir`, `runtime.gpu_id` |
| GPTQ / AWQ | `source.pretrain_runs[*].path`, `project.output_dir`, both dataset `cache_dir` fields, `runtime.gpu_id` |



## 3. Prepare data

Download and tokenize FineWeb-Edu into a shared 20B-token cache.

```bash
python launch.py --config configs/adamw_opt125m_2b.yaml --prepare-data-only
```

## 4. Pretraining

Train OPT from scratch with a shared 1% warmup + 89% stable trajectory, followed by Stable and two cosine-decay continuations.

```bash
python launch.py --config configs/adamw_opt125m_2b.yaml
```

## 5. Weight averaging

After training, average the last 10 Stable checkpoints with LAWA, WMA, and LNWA.

```bash
python tools/build_weighted_endpoints.py --config configs/opt125m_offline_averaging.yaml
```

## 6. Post-training quantization

Prepare shared C4/RefinedWeb caches, then evaluate GPTQ and AWQ.

```bash
python PTQ/GPTQ/launch.py --config PTQ/GPTQ/configs/gptq_opt125m_2b.yaml --prepare-data-only
python PTQ/GPTQ/launch.py --config PTQ/GPTQ/configs/gptq_opt125m_2b.yaml
python PTQ/AWQ/launch.py --config PTQ/AWQ/configs/awq_opt125m_2b.yaml
```


## 7. Other model sizes

Replace `--config` with the matching file below; PTQ filenames use the `gptq_` or `awq_` prefix.

| Model | Training config | Averaging config | GPTQ / AWQ config suffix |
| --- | --- | --- | --- |
| OPT-125M | `configs/adamw_opt125m_2b.yaml` | `configs/opt125m_offline_averaging.yaml` | `opt125m_2b.yaml` |
| OPT-350M | `configs/adamw_opt350m_4b.yaml` | `configs/opt350m_offline_averaging.yaml` | `opt350m_4b.yaml` |
| OPT-1.3B | `configs/adamw_opt1p3b_20b.yaml` | `configs/opt1p3b_offline_averaging.yaml` | `opt1p3b_20b.yaml` |
