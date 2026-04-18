# Towards Empathetic Conversational Recommender System

This project is a fork of [zxd-octopus/ECR](https://github.com/zxd-octopus/ECR) (the official implementation of the RecSys 2024 paper *"Towards Empathetic Conversational Recommender System"*), adapted to run on modern NVIDIA GPUs based on the Blackwell architecture (e.g. RTX 5090, `sm_120`).

The original code targets PyTorch 1.8.1 + CUDA 11.1 and Python 3.8, which are not compatible with the RTX 5090. This fork upgrades the stack to PyTorch built against CUDA 12.8 and Python 3.11, and reorganizes the run instructions into a Jupyter notebook (`run_project.ipynb`) so each pipeline stage can be executed and re-run independently.

## Versions

Tested environment:


| Component             | Version                                 |
| --------------------- | --------------------------------------- |
| OS                    | Linux (kernel 6.19, CachyOS)            |
| Python                | 3.11.14                                 |
| Package manager       | `[uv](https://github.com/astral-sh/uv)` |
| NVIDIA driver         | 595.58.03                               |
| CUDA runtime (driver) | 13.2                                    |
| CUDA build target     | 12.8 (`cu128` wheels)                   |
| GPU                   | NVIDIA GeForce RTX 5090 (32 GB VRAM)    |


Key Python packages (installed in `./.venv/`):


| Package                             | Version           |
| ----------------------------------- | ----------------- |
| `torch`                             | 2.11.0+cu128      |
| `torchvision`                       | 0.26.0+cu128      |
| `transformers`                      | 4.30.0            |
| `accelerate`                        | 1.13.0            |
| `torch-geometric`                   | 2.7.0             |
| `numpy`                             | 1.26.4            |
| `scikit-learn`                      | 1.8.0             |
| `nltk`                              | 3.9.4             |
| `loguru`, `wandb`, `einops`, `tqdm` | latest compatible |


## Download datasets

Two archives must be downloaded from Google Drive (links provided by the original authors):

1. **Pre-processed ECR data** &rarr; unzip into `src_emo/data/emo_data/`
  [https://drive.google.com/file/d/1fb9kDo8uSRLlwc5c4nUw8DZHR5XOY_l_/view?usp=sharing](https://drive.google.com/file/d/1fb9kDo8uSRLlwc5c4nUw8DZHR5XOY_l_/view?usp=sharing)
2. **Released checkpoints** &rarr; unzip into `src_emo/data/saved/`
  [https://drive.google.com/file/d/1uBtcqbQByVrrJ1hEwk2dvsAOxuvEgE19/view?usp=sharing](https://drive.google.com/file/d/1uBtcqbQByVrrJ1hEwk2dvsAOxuvEgE19/view?usp=sharing)

Manual procedure:

1. Download both `.zip` files via your browser.
2. Extract their contents directly inside the two folders listed above.
3. After extraction, `src_emo/data/emo_data/` should contain `entity2id.json`, `train_data_dbpedia_emo.jsonl`, `valid_data_dbpedia_emo.jsonl`, `test_data_dbpedia_emo.jsonl`, `dbpedia_subkg.json`, `relation2id.json`, `relation_set.json`, `stop_words.txt`, `common_template.json`, `movie_reviews_filted_0.1_confi.json`, `conv_unicrs_{train,valid,test}.jsonl`, and the `llama_{train,test}.json` files.

## How to run

### 1. Create the virtual environment

This fork uses `[uv](https://github.com/astral-sh/uv)` for fast, reproducible environments.

```bash
# install uv (one-off, see https://docs.astral.sh/uv/ for other methods)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create a Python 3.11 venv at the repo root
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 2. Install dependencies

All Python dependencies are pinned in `[requirements.txt](./requirements.txt)`. The file already declares the CUDA 12.8 PyTorch index URL, so a single command is enough:

```bash
uv pip install --upgrade pip
uv pip install --index-strategy unsafe-best-match -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

The `--index-strategy unsafe-best-match` flag is required because `requirements.txt` declares both PyPI and the PyTorch CUDA 12.8 index. By default `uv` only resolves a given package from the *first* index that contains it (to prevent dependency-confusion attacks), which causes pinned packages such as `tqdm==4.67.3` to fail when an older version exists on the PyTorch index. `unsafe-best-match` tells `uv` that both indexes are equally trusted and to pick the best matching version across all of them.

(`pip install -r requirements.txt` works as well if you do not use `uv`; plain `pip` does not have this restriction.)

### 3. Run the pipeline

The recommended entry point is the notebook:

```bash
jupyter lab run_project.ipynb
```

It is organized in four sections that mirror the paper:

1. **Preparation** environment & dataset checks, Google Drive download helpers.
2. **Subtask A Emotional Semantic Fusion** (`src_emo/train_pre.py`).
3. **Subtask B Emotion-aware Item Recommendation** (`src_emo/train_rec.py`).
4. **Subtask C Emotion-aligned Response Generation** (`src_emo/train_emp.py` and `src_emo/infer_emp.py`).

Each training step has a **smoke-test** variant (5 steps, batch size 2) right before the full command so you can validate the setup quickly before launching a long run.

If you prefer a pure CLI workflow, the underlying commands are the same as the original `README.md`, e.g.:

```bash
cd src_emo
cp -r data/emo_data/* data/redial/
python data/redial/process.py
accelerate launch train_pre.py \
  --dataset redial \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 64 \
  --num_warmup_steps 1389 \
  --max_length 200 \
  --prompt_max_length 200 \
  --entity_max_length 32 \
  --learning_rate 5e-4 \
  --seed 42 \
  --nei_mer
```

### Hardware notes

The original paper reports experiments on a single 24 GB GPU. The provided batch sizes fit comfortably in the 32 GB of an RTX 5090. If you run on a smaller GPU, lower `per_device_train_batch_size` / `per_device_eval_batch_size` and rescale `gradient_accumulation_steps` accordingly.

## Acknowledgement

Our code is developed based on [UniCRS](https://github.com/RUCAIBox/UniCRS) and forked from the official ECR implementation at [zxd-octopus/ECR](https://github.com/zxd-octopus/ECR).
Any scientific publications that use the original code or dataset should cite the ECR paper as the reference.