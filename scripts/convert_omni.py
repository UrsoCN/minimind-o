import os
import sys

__package__ = "scripts"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import argparse
import json
import warnings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.model_omni import MiniMindOmni, OmniConfig
from trainer.trainer_utils import torch_load_compat

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers(
    torch_path,
    transformers_path,
    lm_config,
    dtype=torch.bfloat16,
    tokenizer_path=None,
    audio_encoder_path=None,
    vision_model_path=None,
):
    """Convert native MiniMind-O PyTorch (.pth) weights into HuggingFace Transformers format."""
    tokenizer_path = tokenizer_path or os.path.join(PROJECT_ROOT, 'model')
    audio_encoder_path = audio_encoder_path or os.path.join(PROJECT_ROOT, 'model', 'SenseVoiceSmall')
    vision_model_path = vision_model_path or os.path.join(PROJECT_ROOT, 'model', 'siglip2-base-p32-256-ve')

    OmniConfig.register_for_auto_class()
    MiniMindOmni.register_for_auto_class("AutoModelForCausalLM")
    model = MiniMindOmni(lm_config, audio_encoder_path=audio_encoder_path, vision_model_path=vision_model_path)
    state_dict = torch_load_compat(torch_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(dtype)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'模型参数: {params:.2f}M')
    del model.audio_encoder
    del model.vision_encoder
    os.makedirs(transformers_path, exist_ok=True)
    model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(transformers_path)

    config_path = os.path.join(transformers_path, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    config['tie_word_embeddings'] = True
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tok_cfg = json.load(f)
        tok_cfg.update({"tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}})
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
        config['rope_theta'] = lm_config.rope_theta
        config['rope_scaling'] = None
        config.pop('rope_parameters', None)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    """Convert HuggingFace Transformers MiniMind-O model directory back to native PyTorch .pth."""
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    os.makedirs(os.path.dirname(os.path.abspath(torch_path)), exist_ok=True)
    torch.save(model.state_dict(), torch_path)
    print(f"已保存为 PyTorch 格式: {torch_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert MiniMind-O weights between PyTorch and Transformers formats")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--torch_path', default=None, type=str)
    parser.add_argument('--transformers_path', default='../minimind-3o', type=str)
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    cli_args = parser.parse_args()

    lm_config = OmniConfig(hidden_size=cli_args.hidden_size, num_hidden_layers=cli_args.num_hidden_layers, use_moe=bool(cli_args.use_moe))
    moe_suffix = '_moe' if lm_config.use_moe else ''
    torch_path = cli_args.torch_path or f"../out/sft_omni_{lm_config.hidden_size}{moe_suffix}.pth"
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    convert_torch2transformers(torch_path, cli_args.transformers_path, lm_config, dtype=dtype_map[cli_args.dtype])
