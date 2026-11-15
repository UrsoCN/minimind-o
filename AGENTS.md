# AGENTS.md

## Project Overview
- **MiniMind-O** (`~0.1B` / `~0.3B-A0.1B` MoE): End-to-end multimodal Omni model supporting text/audio/image inputs and text + 24 kHz streaming Mimi speech outputs.
- **Core stack:** PyTorch (native Thinker–Talker implementation), Transformers (`4.57.6`), FunASR (`SenseVoiceSmall`), SigLIP2, Mimi codec, CAM++, Silero VAD, Gradio, Flask + WebSocket.

## Setup & Environments
- **Local / Multi-GPU:** `pip install -r requirements.txt` (requires `ffmpeg` on `PATH` for `.mp3`/`.webm` audio).
- **Google Colab / Jupyter:** `pip install -r requirements_colab.txt`. Do **not** overwrite Colab's pre-installed CUDA `torch`/`numpy`, and do **not** pin `Pillow==9.5.0` (breaks Python 3.12+ builds).
- **Pretrained assets:** Encoders live under `model/` (`SenseVoiceSmall/`, `siglip2-base-p32-256-ve/`, `mimi/`, `campplus/`), native checkpoints under `out/`, and parquet datasets under `dataset/`.

## Build, Validation & Run Commands
- **Compile check (all Python modules):**
  `python3 -m py_compile notebook_utils.py eval_omni.py dataset/omni_dataset.py trainer/trainer_utils.py trainer/train_sft_omni.py scripts/convert_omni.py scripts/web_demo_omni.py webui/web_demo.py model/model_minimind.py model/model_omni.py`
- **Notebook JSON & syntax check:** Validate `nbformat == 4` and run `ast.parse` on every code cell in `MiniMind_O_Colab.ipynb` and `notebooks/*.ipynb`.
- **Single/Multi-GPU CLI training (from `trainer/`):** `torchrun --nproc_per_node 1 train_sft_omni.py ...` (see `trainer/train.sh`).
- **Notebook/Python training:** Call `run_training(OmniTrainConfig(...))` from `trainer.train_sft_omni`.
- **Evaluation:** `python eval_omni.py --load_from model --weight sft_omni --mode 0`

## Repository Structure & Guardrails
- `model/model_minimind.py`, `model/model_omni.py`: Pure PyTorch Thinker–Talker architecture and Silero VAD. Attach frozen `audio_encoder`/`vision_encoder` via `object.__setattr__` so they stay excluded from saved `state_dict` checkpoints.
- `dataset/`: **Contains tracked repo files (`omni_dataset.py` and `eval_omni/`)**. Never delete, overwrite, or symlink the `dataset/` directory itself when mounting Google Drive; store external parquet files inside it or pass explicit `data_path`.
- `notebook_utils.py`: Shared Colab/Jupyter helper (hardware check, Drive persistence, HF/ModelScope downloads, inline evaluation, `cloudflared` tunnel). Keep `.ipynb` cells thin by routing shared logic here.
- `scripts/convert_omni.py`, `scripts/web_demo_omni.py`, `webui/web_demo.py`: Anchor default model asset paths to `PROJECT_ROOT` so callers work from both repo root and subdirectories.

## Code & Notebook Conventions
- **Notebook-safe APIs:** Expose dataclass configs (`OmniTrainConfig`, `EvalConfig`) and callable entrypoints (`run_training`, `start_gradio_demo`, `start_realtime_webui`) that never invoke `argparse.parse_args()` on import or assume module-level globals.
- **PyTorch compatibility:** Always load `.pth` checkpoints via `torch_load_compat` (`weights_only=False` fallback for PyTorch >= 2.6). Never call `.half()` on CPU tensors/models.
- **AMP dtype:** Use `resolve_amp_dtype("auto", device)` (`float16` + `GradScaler` on T4/SM < 8.0; `bfloat16` on Ampere+/SM >= 8.0; `float32` on CPU).
- **Notebooks:** Keep cell `outputs: []` and `execution_count: null` before committing; use `#@title` / `#@param` comments for Colab form compatibility without breaking standard Python execution.

## Git & Verification Checklist
- **Commit prefixes:** Follow repository tags (`[feat]`, `[fix]`, `[refactor]`, `[update]`, `[docs]`) with concise, single-purpose commits.
- **Before finishing any change:**
  1. Run `py_compile` on modified `.py` files and AST/JSON checks on modified `.ipynb` files.
  2. Verify no hardcoded `../` path breaks project-root execution and no unrequested weight/cache files are staged (`git status`).
