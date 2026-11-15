"""Colab and Jupyter Notebook utilities for MiniMind-O."""

import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

ENCODER_REPOS: Dict[str, Tuple[str, str, str]] = {
    "SenseVoiceSmall": ("jingyaogong/SenseVoiceSmall", "gongjy/SenseVoiceSmall", os.path.join("model", "SenseVoiceSmall")),
    "siglip2": ("jingyaogong/siglip2-base-p32-256-ve", "gongjy/siglip2-base-p32-256-ve", os.path.join("model", "siglip2-base-p32-256-ve")),
    "mimi": ("jingyaogong/mimi", "gongjy/mimi", os.path.join("model", "mimi")),
    "campplus": ("jingyaogong/campplus", "gongjy/campplus", os.path.join("model", "campplus")),
}

PYTORCH_WEIGHT_REPO = ("jingyaogong/minimind-3o-pytorch", "gongjy/minimind-3o-pytorch")
DATASET_REPO = ("jingyaogong/minimind-o_dataset", "gongjy/minimind-o_dataset")
DATASET_PRESETS = {
    "mini": ["sft_t2a_mini.parquet", "sft_a2a_mini.parquet"],
    "full": ["sft_t2a.parquet", "sft_a2a.parquet", "sft_i2t.parquet"],
    "all": ["sft_t2a_mini.parquet", "sft_a2a_mini.parquet", "sft_t2a.parquet", "sft_a2a.parquet", "sft_i2t.parquet"],
}


def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    return "google.colab" in sys.modules


def ensure_project_root() -> str:
    """Ensure working directory and sys.path point to the MiniMind-O repository root."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if os.path.abspath(os.getcwd()) != os.path.abspath(PROJECT_ROOT):
        os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


def check_environment() -> Dict[str, object]:
    """Inspect runtime hardware and return recommended Colab/Notebook settings."""
    info: Dict[str, object] = {
        "in_colab": is_colab(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gpu_name": None,
        "vram_gb": 0.0,
        "compute_capability": None,
        "bf16_supported": False,
        "recommended_dtype": "float32",
        "recommended_batch_size": 4,
        "recommended_use_compile": 0,
        "recommended_num_workers": min(2, os.cpu_count() or 2),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        vram_gb = props.total_memory / (1024 ** 3)
        try:
            bf16_ok = (major >= 8) and torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            bf16_ok = (major >= 8) and torch.cuda.is_bf16_supported()

        info["gpu_name"] = props.name
        info["vram_gb"] = round(vram_gb, 2)
        info["compute_capability"] = f"{major}.{minor}"
        info["bf16_supported"] = bf16_ok
        info["recommended_dtype"] = "bfloat16" if bf16_ok else "float16"
        if vram_gb >= 30:
            info["recommended_batch_size"] = 40
        elif vram_gb >= 18:
            info["recommended_batch_size"] = 24
        elif vram_gb >= 12:
            info["recommended_batch_size"] = 16
        else:
            info["recommended_batch_size"] = 8
        info["recommended_use_compile"] = 1 if major >= 8 else 0

    print("=== MiniMind-O Runtime Environment ===")
    print(f"Environment   : {'Google Colab' if info['in_colab'] else 'Local Jupyter / Python'}")
    print(f"Python / Torch: {info['python']} / {info['torch']}")
    if info["cuda_available"]:
        print(f"GPU           : {info['gpu_name']} ({info['vram_gb']} GB, SM {info['compute_capability']})")
        print(f"Recommended   : dtype={info['recommended_dtype']}, batch_size={info['recommended_batch_size']}, use_compile={info['recommended_use_compile']}")
    else:
        print("GPU           : None (CPU mode — inference works, training is not recommended)")
    print(f"ffmpeg        : {'available' if info['ffmpeg_available'] else 'NOT FOUND (install with: apt-get install -y ffmpeg)'}")
    return info


def setup_persistent_storage(
    drive_root: str = "/content/drive/MyDrive/minimind-o",
    mount_point: str = "/content/drive",
    link_out_and_checkpoints: bool = True,
) -> Dict[str, str]:
    """Mount Google Drive in Colab and configure persistent folders without overwriting repo files."""
    ensure_project_root()
    if not is_colab():
        paths = {
            "save_dir": os.path.join(PROJECT_ROOT, "out"),
            "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
            "dataset_dir": os.path.join(PROJECT_ROOT, "dataset"),
            "hf_dir": PROJECT_ROOT,
        }
        os.makedirs(paths["save_dir"], exist_ok=True)
        os.makedirs(paths["checkpoint_dir"], exist_ok=True)
        return paths

    from google.colab import drive
    if not os.path.ismount(mount_point):
        drive.mount(mount_point)

    paths = {
        "save_dir": os.path.join(drive_root, "out"),
        "checkpoint_dir": os.path.join(drive_root, "checkpoints"),
        "dataset_dir": os.path.join(drive_root, "dataset"),
        "hf_dir": os.path.join(drive_root, "hf_models"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    if link_out_and_checkpoints:
        for local_name, key in [("out", "save_dir"), ("checkpoints", "checkpoint_dir")]:
            local_path = os.path.join(PROJECT_ROOT, local_name)
            target = paths[key]
            if os.path.islink(local_path):
                continue
            if os.path.isdir(local_path) and not os.listdir(local_path):
                os.rmdir(local_path)
            if not os.path.exists(local_path):
                os.symlink(target, local_path)
                print(f"Linked ./{local_name} -> {target}")

    print(f"Persistent Google Drive paths ready under: {drive_root}")
    return paths


def _download_repo(
    hf_repo: str,
    ms_repo: str,
    local_dir: str,
    source: str = "huggingface",
    filenames: Optional[Sequence[str]] = None,
    repo_type: str = "model",
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    source = source.lower().strip()
    if source in ("hf", "huggingface"):
        from huggingface_hub import hf_hub_download, snapshot_download

        if filenames:
            for fname in filenames:
                print(f"[HF] Downloading {hf_repo}/{fname} -> {local_dir}")
                hf_hub_download(repo_id=hf_repo, filename=fname, repo_type=repo_type, local_dir=local_dir)
        else:
            print(f"[HF] Downloading {hf_repo} -> {local_dir}")
            snapshot_download(repo_id=hf_repo, repo_type=repo_type, local_dir=local_dir)
    elif source in ("ms", "modelscope"):
        if filenames:
            for fname in filenames:
                print(f"[ModelScope] Downloading {ms_repo}/{fname} -> {local_dir}")
                cmd = ["modelscope", "download"]
                if repo_type == "dataset":
                    cmd.extend(["--dataset", ms_repo])
                else:
                    cmd.extend(["--model", ms_repo])
                cmd.extend([fname, "--local_dir", local_dir])
                subprocess.run(cmd, check=True)
        else:
            print(f"[ModelScope] Downloading {ms_repo} -> {local_dir}")
            cmd = ["modelscope", "download"]
            if repo_type == "dataset":
                cmd.extend(["--dataset", ms_repo])
            else:
                cmd.extend(["--model", ms_repo])
            cmd.extend(["--local_dir", local_dir])
            subprocess.run(cmd, check=True)
    else:
        raise ValueError(f"Unsupported download source: {source!r}. Use 'huggingface' or 'modelscope'.")
    return local_dir


def download_encoders(
    source: str = "huggingface",
    include_campplus: bool = True,
    force: bool = False,
) -> Dict[str, str]:
    """Download frozen SenseVoiceSmall, SigLIP2, Mimi, and CAM++ encoders under ./model/."""
    ensure_project_root()
    downloaded = {}
    keys = ["SenseVoiceSmall", "siglip2", "mimi"] + (["campplus"] if include_campplus else [])
    for key in keys:
        hf_repo, ms_repo, rel_dir = ENCODER_REPOS[key]
        target = os.path.join(PROJECT_ROOT, rel_dir)
        has_files = os.path.isdir(target) and any(not f.startswith(".") for f in os.listdir(target))
        if has_files and not force:
            print(f"[Skip] {rel_dir} already exists.")
        else:
            _download_repo(hf_repo, ms_repo, target, source=source, repo_type="model")
        downloaded[key] = target
    return downloaded


def download_pytorch_weights(
    filenames: Optional[Sequence[str]] = None,
    source: str = "huggingface",
    save_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """Download native PyTorch weights (e.g., ['llm_768.pth'] or ['sft_omni_768.pth']) into save_dir."""
    ensure_project_root()
    target = save_dir or os.path.join(PROJECT_ROOT, "out")
    os.makedirs(target, exist_ok=True)
    pending = filenames
    if filenames and not force:
        pending = [f for f in filenames if not os.path.exists(os.path.join(target, f))]
        if not pending:
            print(f"[Skip] Requested weights {list(filenames)} already exist in {target}.")
            return target
    _download_repo(PYTORCH_WEIGHT_REPO[0], PYTORCH_WEIGHT_REPO[1], target, source=source, filenames=pending, repo_type="model")
    return target


def download_hf_model(
    model_name: str = "minimind-3o",
    source: str = "huggingface",
    target_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """Download a Transformers-format MiniMind-O model (minimind-3o or minimind-3o-moe)."""
    ensure_project_root()
    dest = target_dir or os.path.join(PROJECT_ROOT, model_name)
    has_weights = os.path.isdir(dest) and bool(
        set(os.listdir(dest)) & {"pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json", "model.safetensors.index.json"}
    )
    if has_weights and not force:
        print(f"[Skip] {dest} already contains model weights.")
        return dest
    return _download_repo(f"jingyaogong/{model_name}", f"gongjy/{model_name}", dest, source=source, repo_type="model")


def download_datasets(
    preset: str = "mini",
    files: Optional[Sequence[str]] = None,
    source: str = "huggingface",
    dataset_dir: Optional[str] = None,
    force: bool = False,
) -> List[str]:
    """Download MiniMind-O parquet datasets ('mini', 'full', 'all', or explicit filenames)."""
    ensure_project_root()
    target = dataset_dir or os.path.join(PROJECT_ROOT, "dataset")
    os.makedirs(target, exist_ok=True)
    wanted = list(files) if files else list(DATASET_PRESETS[preset])
    pending = wanted if force else [f for f in wanted if not os.path.exists(os.path.join(target, f))]
    if pending:
        _download_repo(DATASET_REPO[0], DATASET_REPO[1], target, source=source, filenames=pending, repo_type="dataset")
    else:
        print(f"[Skip] Dataset files {wanted} already exist in {target}.")
    return [os.path.join(target, f) for f in wanted]


def inspect_parquet_dataset(
    parquet_path: str,
    num_samples: int = 2,
    mimi_path: Optional[str] = None,
    device: str = "cpu",
):
    """Preview parquet dataset rows in Jupyter with inline audio and image rendering."""
    import pyarrow.parquet as pq
    from IPython.display import Audio, Image as IPyImage, display

    pf = pq.ParquetFile(parquet_path)
    print(f"File: {parquet_path}")
    print(f"Total rows: {pf.metadata.num_rows} | Columns: {pf.schema_arrow.names}")

    batch = next(pf.iter_batches(batch_size=max(1, num_samples)))
    table = batch.to_pydict()

    mimi = None
    mimi_dir = mimi_path or os.path.join(PROJECT_ROOT, "model", "mimi")
    if os.path.exists(mimi_dir) and "answer_audios" in table:
        from transformers import MimiModel
        mimi = MimiModel.from_pretrained(mimi_dir).eval().to(device)

    n = min(num_samples, len(table["conversations"]))
    for i in range(n):
        print(f"\n--- Sample {i + 1} ---")
        convs = json.loads(table["conversations"][i])
        for turn in convs:
            print(f"[{turn['role']}]: {turn['content']}")

        if "image_bytes" in table and table["image_bytes"][i]:
            img_raw = table["image_bytes"][i]
            img_bytes = img_raw[0] if isinstance(img_raw, list) else img_raw
            if img_bytes:
                display(IPyImage(data=img_bytes, width=256))

        if "question_audios" in table and table["question_audios"][i]:
            for q_idx, ab in enumerate(table["question_audios"][i]):
                if ab:
                    import soundfile as sf
                    wav, sr = sf.read(io.BytesIO(ab))
                    print(f"Question audio [{q_idx}] (sr={sr}, {len(wav)/sr:.2f}s):")
                    display(Audio(wav, rate=sr))

        if mimi is not None and "answer_audios" in table and table["answer_audios"][i]:
            for a_idx, tokens in enumerate(table["answer_audios"][i]):
                if tokens and len(tokens) >= 8:
                    layers = [tokens[k:len(tokens) - (len(tokens) % 8):8] for k in range(8)]
                    codes = torch.tensor(layers, dtype=torch.long, device=device).unsqueeze(0)
                    codes = torch.where(codes >= 2049, torch.zeros_like(codes), codes)
                    with torch.no_grad():
                        wav = mimi.decode(codes).audio_values.squeeze().float().cpu().numpy()
                    print(f"Answer audio [{a_idx}] (Mimi decoded, {len(wav)/24000:.2f}s):")
                    display(Audio(wav, rate=24000))


def plot_training_history(history: Dict[str, List[float]], title: str = "MiniMind-O SFT Training"):
    """Plot total/text/audio losses and learning rate from run_training() history."""
    if not history or not history.get("steps"):
        print("No logged training history points to plot.")
        return

    import matplotlib.pyplot as plt

    steps = history["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(steps, history["loss"], label="Total Loss", linewidth=2)
    axes[0].plot(steps, history["text_loss"], label="Text Loss", linestyle="--")
    axes[0].plot(steps, history["audio_loss"], label="Audio Loss", linestyle=":")
    axes[0].set_title(title)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, history["lr"], color="tab:orange", linewidth=2)
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("LR")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def record_audio_colab(filename: str = "mic_input.wav", seconds: int = 5) -> str:
    """Record audio from the browser microphone in Google Colab and save as 16kHz mono WAV."""
    if not is_colab():
        raise RuntimeError(
            "Browser JS microphone recording requires Google Colab. "
            "In local Jupyter, pass any existing .wav/.mp3 path directly to session.chat(audio_path=...)."
        )

    from google.colab import output
    from IPython.display import Audio, HTML, display
    from pydub import AudioSegment

    js_code = f"""
    (async () => {{
      const div = document.createElement('div');
      const status = document.createElement('p');
      status.textContent = '🎤 Recording from microphone ({seconds}s)...';
      div.appendChild(status);
      document.body.appendChild(div);

      const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);

      const done = new Promise((resolve) => {{
        recorder.onstop = async () => {{
          stream.getTracks().forEach(t => t.stop());
          const blob = new Blob(chunks, {{ type: recorder.mimeType || 'audio/webm' }});
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result.split(',')[1]);
          reader.readAsDataURL(blob);
        }};
      }});

      recorder.start();
      await new Promise(r => setTimeout(r, {int(seconds * 1000)}));
      recorder.stop();
      const b64 = await done;
      status.textContent = '✅ Recording complete!';
      return b64;
    }})()
    """
    display(HTML(f"<p>🎤 请允许浏览器麦克风权限，正在录音 {seconds} 秒...</p>"))
    b64_data = output.eval_js(js_code)
    raw_bytes = base64.b64decode(b64_data)

    seg = AudioSegment.from_file(io.BytesIO(raw_bytes)).set_frame_rate(16000).set_channels(1).set_sample_width(2)
    out_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seg.export(out_path, format="wav")
    print(f"Saved recorded audio to: {out_path}")
    display(Audio(out_path))
    return out_path


class OmniNotebookSession:
    """High-level interactive multimodal chat, voice cloning, and evaluation session for Notebooks."""

    def __init__(self, eval_config=None):
        ensure_project_root()
        from eval_omni import EvalConfig, init_model

        self.cfg = eval_config or EvalConfig()
        self.model, self.tokenizer = init_model(self.cfg)
        self.device = self.cfg.device
        self.model_dtype = next(self.model.parameters()).dtype
        self.voices_data: Dict[str, dict] = {}
        self.voice_groups: Dict[str, List[str]] = {"builtin": [], "unseen": [], "manual": []}
        self._campplus = None
        self._mel_fn = None
        self.load_voices()

    def load_voices(self) -> Dict[str, List[str]]:
        from trainer.trainer_utils import torch_load_compat

        self.voices_data = {}
        self.voice_groups = {"builtin": [], "unseen": [], "manual": []}
        spk_dir = os.path.join(PROJECT_ROOT, "model", "speaker")
        for fn, group in [("voices.pt", "builtin"), ("voices_unseen.pt", "unseen"), ("voice_clone.pt", "manual")]:
            fp = os.path.join(spk_dir, fn)
            if os.path.exists(fp):
                data = torch_load_compat(fp, map_location=self.device)
                for speaker, val in data.items():
                    self.voices_data[speaker] = val
                    if speaker not in self.voice_groups[group]:
                        self.voice_groups[group].append(speaker)
        return self.voice_groups

    def list_voices(self) -> Dict[str, List[str]]:
        return {k: sorted(v) for k, v in self.voice_groups.items()}

    def _ensure_campplus(self):
        if self._campplus is not None:
            return True
        ckpt_path = os.path.join(PROJECT_ROOT, "model", "campplus", "campplus_cn_common.pt")
        if not os.path.exists(ckpt_path):
            warnings.warn(f"CAM++ checkpoint not found at {ckpt_path}; cloning will use ref_codes only.")
            return False
        try:
            import torchaudio
            from modelscope.models.audio.sv.DTDNN import CAMPPlus
            from trainer.trainer_utils import torch_load_compat

            camp = CAMPPlus(
                feat_dim=80, embedding_size=192, growth_rate=32, bn_size=4,
                init_channels=128, config_str="batchnorm-relu", memory_efficient=True,
            )
            sd = torch_load_compat(ckpt_path, map_location="cpu")
            camp.load_state_dict({k: v.float() for k, v in sd.items()})
            self._campplus = camp.eval().to(self.device)
            self._mel_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                n_mels=80, f_min=20, f_max=7600, norm="slaney", mel_scale="slaney",
            ).to(self.device)
            return True
        except Exception as e:
            warnings.warn(f"Failed to load CAM++ ({e}); cloning will use ref_codes only.")
            return False

    def clone_voice_from_audio(self, ref_audio_path: str, voice_name: str = "custom_clone", save: bool = True) -> str:
        """Encode reference audio into a voice prompt (ref_codes + optional spk_emb) for zero-shot cloning."""
        if getattr(self.model, "mimi_model", None) is None:
            raise RuntimeError("Mimi model is required for voice cloning; download ./model/mimi first.")

        from pydub import AudioSegment

        seg = AudioSegment.from_file(ref_audio_path).set_channels(1).set_sample_width(2)
        if len(seg) < 1000:
            raise ValueError("Reference audio is too short; provide at least 1 second of clear speech.")
        try:
            seg = seg.speedup(playback_speed=1.5, chunk_size=150, crossfade=25)
        except Exception:
            pass

        seg24 = seg.set_frame_rate(24000)
        seg16 = seg.set_frame_rate(16000)
        w24 = torch.tensor(np.frombuffer(seg24.raw_data, dtype=np.int16).astype(np.float32) / 32768.0)
        w16 = torch.tensor(np.frombuffer(seg16.raw_data, dtype=np.int16).astype(np.float32) / 32768.0)

        mimi_dev = next(self.model.mimi_model.parameters()).device
        mimi_dtype = next(self.model.mimi_model.parameters()).dtype
        with torch.inference_mode():
            t = w24.unsqueeze(0).unsqueeze(0).to(device=mimi_dev, dtype=mimi_dtype)
            codes = self.model.mimi_model.encode(t).audio_codes
            nf = math.ceil(w24.shape[-1] / 1920)
            ref_codes = codes[0, :8, :nf].cpu()[:, :min(nf, 300)]

        entry = {"ref_codes": ref_codes}
        if self._ensure_campplus():
            with torch.no_grad():
                mel = self._mel_fn(w16.unsqueeze(0).to(self.device))
                feat = mel.clamp(min=1e-10).log().transpose(1, 2)
                feat = feat - feat.mean(dim=1, keepdim=True)
                entry["spk_emb"] = self._campplus(feat).squeeze(0).cpu()

        self.voices_data[voice_name] = entry
        if voice_name not in self.voice_groups["manual"]:
            self.voice_groups["manual"].append(voice_name)

        if save:
            from trainer.trainer_utils import torch_load_compat
            clone_path = os.path.join(PROJECT_ROOT, "model", "speaker", "voice_clone.pt")
            saved = torch_load_compat(clone_path, map_location="cpu") if os.path.exists(clone_path) else {}
            saved[voice_name] = entry
            torch.save(saved, clone_path)
            print(f"Saved cloned voice '{voice_name}' -> {clone_path}")
        return voice_name

    def chat(
        self,
        text: str = "",
        audio_path: Optional[str] = None,
        image: Optional[Union[str, "Image.Image"]] = None,
        voice_name: str = "default",
        history: Optional[List[dict]] = None,
        open_thinking: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decode_audio: bool = True,
        save_name: Optional[str] = None,
        display_inline: bool = True,
        autoplay: bool = False,
    ) -> dict:
        """Run a multimodal turn in Jupyter with inline image and audio playback."""
        from PIL import Image
        from dataset.omni_dataset import OmniDataset
        from eval_omni import EvalConfig, eval_sample

        run_cfg = EvalConfig(**vars(self.cfg))
        if open_thinking is not None:
            run_cfg.open_thinking = int(bool(open_thinking))
        if temperature is not None:
            run_cfg.temperature = temperature
        if top_p is not None:
            run_cfg.top_p = top_p
        if max_new_tokens is not None:
            run_cfg.max_new_tokens = max_new_tokens
        run_cfg.decode_audio = int(bool(decode_audio))

        prompt = text or ""
        audio_inputs, audio_lens = None, None
        if audio_path:
            if getattr(self.model, "audio_processor", None) is None:
                raise RuntimeError("SenseVoiceSmall audio encoder is not loaded. Download ./model/SenseVoiceSmall first.")
            mel, valid_len = OmniDataset.process_audio(audio_path, self.model.audio_processor)
            audio_inputs = mel.unsqueeze(0).to(self.device)
            audio_lens = torch.tensor([valid_len], device=self.device)
            audio_prompt = self.model.config.audio_special_token * (valid_len or 1)
            prompt = (prompt + "\n\n" + audio_prompt) if prompt else audio_prompt

        pixel_values = None
        pil_img = None
        if image is not None:
            if getattr(self.model, "vision_processor", None) is None:
                raise RuntimeError("SigLIP2 vision encoder is not loaded. Download ./model/siglip2-base-p32-256-ve first.")
            pil_img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
            pixel_values = {k: v.to(self.device) for k, v in self.model.vision_processor(images=pil_img, return_tensors="pt").items()}
            img_prompt = self.model.config.image_special_token * self.model.config.image_token_len
            prompt = (prompt + "\n\n" if prompt else "") + img_prompt

        ref_codes, spk_emb = None, None
        effective_history = list(history or [])
        if voice_name and voice_name != "default":
            if voice_name not in self.voices_data:
                raise KeyError(f"Unknown voice {voice_name!r}. Available: {self.list_voices()}")
            v = self.voices_data[voice_name]
            ref_codes = v["ref_codes"].unsqueeze(0).to(self.device)
            spk_emb = v["spk_emb"].to(dtype=self.model_dtype).unsqueeze(0).to(self.device) if "spk_emb" in v else None
            if not any(m.get("role") == "system" for m in effective_history):
                effective_history.insert(0, {
                    "role": "system",
                    "content": "你是一个专业的语音助手，请用给定的音色风格来回答用户的问题。请尽量详细地回答，给出有价值的信息。",
                })

        if display_inline:
            from IPython.display import Audio, display
            if pil_img is not None:
                display(pil_img.resize((min(320, pil_img.width), int(pil_img.height * min(320, pil_img.width) / max(1, pil_img.width)))))
            if audio_path:
                print(f"🎤 Input audio: {audio_path}")
                display(Audio(audio_path))

        res = eval_sample(
            self.model, self.tokenizer, run_cfg, 0, prompt, audio_inputs, save_name,
            pixel_values=pixel_values, history=effective_history or None,
            audio_lens=audio_lens, ref_codes=ref_codes, spk_emb=spk_emb,
            stream_print=True, save_audio=bool(save_name),
        )

        if display_inline and res and res.get("audio") is not None:
            from IPython.display import Audio, display
            display(Audio(res["audio"], rate=res["sample_rate"], autoplay=autoplay))
        return res

    def run_eval_suite(
        self,
        mode: str = "0",
        prompt_lang: int = 0,
        max_items: int = 2,
        display_inline: bool = True,
    ) -> List[dict]:
        """Run selected evaluation modes from dataset/eval_omni/ and display outputs inline."""
        modes = set(mode.replace(",", "").replace("-1", "012345"))
        audio_dir = self.cfg.audio_dir
        image_dir = self.cfg.image_dir
        results = []

        if "0" in modes:
            print("\n=== Mode 0: Text -> Text + Audio ===")
            prompts_en = ["Tell me an interesting fact about space.", "How do I make a cup of coffee?", "Please introduce yourself."]
            prompts_zh = ["告诉我一个关于太空的有趣事实。", "如何制作一杯咖啡？", "介绍一下你自己"]
            prompts = [prompts_en, prompts_zh, prompts_en + prompts_zh][prompt_lang][:max_items]
            for i, p in enumerate(prompts):
                print(f"\n📝 [{i+1}] {p}")
                results.append(self.chat(text=p, display_inline=display_inline))

        if "1" in modes:
            print("\n=== Mode 1: Multi-Turn Dialogue ===")
            hist_zh = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}]
            hist_en = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello! How can I help you?"}]
            tests = [
                [(hist_en, "I want to find something to do. Do you have any suggestions?")],
                [(hist_zh, "我想找点事做，你有什么建议吗？")],
            ]
            selected = tests[0] if prompt_lang == 0 else (tests[1] if prompt_lang == 1 else tests[0] + tests[1])
            for hist, p in selected[:max_items]:
                print(f"\n💬 History: {hist}\n   User: {p}")
                results.append(self.chat(text=p, history=hist, display_inline=display_inline))

        if "2" in modes:
            print("\n=== Mode 2: Audio -> Text + Audio ===")
            en_files = sorted([f for f in os.listdir(audio_dir) if f.startswith("audio-en-") and f.lower().endswith((".mp3", ".wav"))])
            zh_files = sorted([f for f in os.listdir(audio_dir) if f.startswith("audio-zh-") and f.lower().endswith((".mp3", ".wav"))])
            files = [en_files, zh_files, en_files + zh_files][prompt_lang][:max_items]
            for f in files:
                results.append(self.chat(audio_path=os.path.join(audio_dir, f), display_inline=display_inline))

        if "3" in modes:
            print("\n=== Mode 3: Voice Cloning ===")
            p = "Hello, please introduce yourself." if prompt_lang == 0 else "你好，请介绍一下你自己。"
            speakers = ["default"] + sorted(self.voices_data.keys())[:max_items]
            for spk in speakers:
                print(f"\n🎵 Speaker: {spk}")
                results.append(self.chat(text=p, voice_name=spk, display_inline=display_inline))

        if "4" in modes:
            print("\n=== Mode 4: Image -> Text + Audio ===")
            img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:max_items]
            p = "Please describe this image." if prompt_lang == 0 else "请描述这张图片"
            for f in img_files:
                print(f"\n🖼️ Image: {f}")
                results.append(self.chat(text=p, image=os.path.join(image_dir, f), display_inline=display_inline))

        if "5" in modes:
            print("\n=== Mode 5: Text + Audio + Image -> Text + Audio ===")
            img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:max_items]
            img_audios = sorted([f for f in os.listdir(audio_dir) if f.startswith("img-") and f.lower().endswith((".mp3", ".wav"))])
            if not img_files or not img_audios:
                print(f"Skipping Mode 5: missing image or img-* audio files in {image_dir}")
            else:
                hint = "Please answer me: " if prompt_lang == 0 else "请回答我："
                for i, img_f in enumerate(img_files):
                    aud_f = img_audios[i % len(img_audios)]
                    print(f"\n🌀 Mix: {hint} | {aud_f} | {img_f}")
                    results.append(self.chat(text=hint, audio_path=os.path.join(audio_dir, aud_f), image=os.path.join(image_dir, img_f), display_inline=display_inline))

        return results


def launch_cloudflared_tunnel(port: int = 7860, timeout: int = 30) -> Tuple[subprocess.Popen, str]:
    """Start a free HTTPS/WSS Cloudflare Quick Tunnel for Colab Flask+WebSocket WebUI."""
    bin_path = "/tmp/cloudflared"
    if not os.path.exists(bin_path):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        print(f"Downloading cloudflared -> {bin_path} ...")
        subprocess.run(["curl", "-fsSL", "-o", bin_path, url], check=True)
        os.chmod(bin_path, 0o755)

    log_path = f"/tmp/cloudflared_{port}.log"
    with open(log_path, "w+") as log_f:
        proc = subprocess.Popen(
            [bin_path, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    pattern = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")
    public_url = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        with open(log_path, "r") as r:
            content = r.read()
        match = pattern.search(content)
        if match:
            public_url = match.group(0)
            break
        time.sleep(0.5)

    if not public_url:
        proc.terminate()
        with open(log_path, "r") as r:
            raise RuntimeError(f"Failed to obtain trycloudflare.com URL within {timeout}s:\n{r.read()}")

    print(f"🌐 Public HTTPS/WSS WebUI URL: {public_url}")
    print(f"📞 Phone-mode call page URL : {public_url}/call")
    return proc, public_url
