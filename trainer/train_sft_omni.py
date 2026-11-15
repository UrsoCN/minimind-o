import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass

import datasets
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.omni_dataset import OmniDataset
from model.model_omni import OmniConfig
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_omni_model,
    is_main_process,
    log_model_params,
    omni_checkpoint,
    resolve_amp_dtype,
    setup_seed,
)

warnings.filterwarnings('ignore')


@dataclass
class OmniTrainConfig:
    """Notebook-friendly training configuration with project-root default paths."""
    save_dir: str = "./out"
    checkpoint_dir: str = "./checkpoints"
    save_weight: str = "sft_omni"
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 5e-4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "auto"
    num_workers: int = 2
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    hidden_size: int = 768
    num_hidden_layers: int = 8
    max_seq_len: int = 512
    use_moe: int = 0
    data_path: str = "./dataset/sft_t2a_mini.parquet"
    tokenizer_path: str = "./model"
    audio_encoder_dir: str = "./model/SenseVoiceSmall"
    vision_dir: str = "./model/siglip2-base-p32-256-ve"
    from_weight: str = "llm"
    from_resume: int = 0
    freeze_backbone: str = "none"
    mode: str = "all"
    use_wandb: bool = False
    wandb_project: str = "MiniMind-O-SFT"
    use_compile: int = 0
    max_steps: int = 0
    max_samples: int = 0


def omni_collate_fn(batch):
    """自定义collate函数，处理变长audio_inputs和pixel_values"""
    input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = zip(*batch)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    audio_labels = torch.stack(audio_labels)
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)
    valid_audios = [a for a in audio_inputs if a is not None]
    if valid_audios:
        max_t = max(a.size(1) for a in valid_audios)
        padded = [a if a.size(1) == max_t else torch.nn.functional.pad(a, (0, 0, 0, max_t - a.size(1))) for a in valid_audios]
        audio_inputs = torch.cat(padded, dim=0)
    else:
        audio_inputs = None
    valid_images = [p for p in pixel_values if p is not None]
    if valid_images:
        if hasattr(valid_images[0], 'keys'):
            keys = set.intersection(*[set(d.keys()) for d in valid_images])
            pixel_values = {k: torch.cat([d[k] for d in valid_images], dim=0) for k in keys}
        else:
            pixel_values = torch.cat(valid_images, dim=0)
    else:
        pixel_values = None
    spk_emb = torch.stack(spk_emb)
    return input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb


def _save_training_state(model, optimizer, scaler, omni_config, args, epoch, step, wandb=None):
    """Save inference weights (.pth) and full resume checkpoint."""
    model.eval()
    moe_suffix = '_moe' if omni_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    clean_state_dict = {k: v for k, v in raw_model.state_dict().items() if not k.startswith('audio_encoder.')}
    torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp)
    checkpoint_dir = getattr(args, 'checkpoint_dir', '../checkpoints')
    omni_checkpoint(
        omni_config, weight=args.save_weight, model=model, optimizer=optimizer,
        epoch=epoch, step=step, wandb=wandb, save_dir=checkpoint_dir, scaler=scaler
    )
    model.train()


def train_epoch(epoch, loader, iters, model, optimizer, scaler, autocast_ctx, omni_config, args,
                start_step=0, wandb=None, history=None, max_steps=0, global_step=0):
    start_time = time.time()
    stopped_early = False
    for step, (input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        audio_labels = audio_labels.to(args.device)
        audio_lens = audio_lens.to(args.device)
        if audio_inputs is not None:
            audio_inputs = audio_inputs.to(args.device)
        if pixel_values is not None:
            if hasattr(pixel_values, 'keys'):
                pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(args.device)
        spk_emb = spk_emb.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values, spk_emb=spk_emb)
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            
            # Text loss
            text_loss_raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), labels.view(-1))
            text_mask = (labels.view(-1) != -100).float()
            text_loss = (text_loss_raw * text_mask).sum() / (text_mask.sum() + 1e-9)
            
            # Audio loss
            audio_loss = res.audio_logits[0].sum() * 0
            for i, al in enumerate(res.audio_logits):
                al_flat = al.view(-1, al.size(-1))
                target_flat = audio_labels[:, i, :].reshape(-1)
                layer_loss = loss_fct(al_flat, target_flat)
                valid_mask = (target_flat != -100).float()
                stop_mask = (target_flat == 2050).float()
                weighted_loss = layer_loss * valid_mask * (1 + stop_mask * 9)
                msum = valid_mask.sum()
                if msum > 0:
                    audio_loss = audio_loss + weighted_loss.sum() / (msum + 1e-9)
            audio_loss = audio_loss / 8 
            
            loss = (text_loss + audio_loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        hit_max_steps = (max_steps > 0 and global_step >= max_steps)

        if step % args.log_interval == 0 or step == iters or hit_max_steps:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            text_loss_val = text_loss.item() if isinstance(text_loss, torch.Tensor) else 0
            audio_loss_val = audio_loss.item() if isinstance(audio_loss, torch.Tensor) else 0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, text: {text_loss_val:.4f}, audio: {audio_loss_val:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if history is not None and is_main_process():
                history["steps"].append(global_step)
                history["epochs"].append(epoch + step / max(iters, 1))
                history["loss"].append(current_loss)
                history["text_loss"].append(text_loss_val)
                history["audio_loss"].append(audio_loss_val)
                history["lr"].append(current_lr)
            if wandb: 
                wandb.log({"loss": current_loss, "text_loss": text_loss_val, 
                          "audio_loss": audio_loss_val, "lr": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters or hit_max_steps) and is_main_process():
            _save_training_state(model, optimizer, scaler, omni_config, args, epoch, step, wandb=wandb)

        del input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb, res, loss
        if hit_max_steps:
            Logger(f'达到 max_steps={max_steps}，保存权重并结束训练。')
            stopped_early = True
            break

    return global_step, stopped_early


def _make_amp_and_scaler(dtype_arg, device):
    device_type = "cuda" if "cuda" in str(device) and torch.cuda.is_available() else "cpu"
    resolved_dtype = resolve_amp_dtype(dtype_arg, device)
    if device_type == "cpu":
        autocast_ctx = nullcontext()
    else:
        amp_dtype = torch.bfloat16 if resolved_dtype == "bfloat16" else torch.float16
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            autocast_ctx = torch.amp.autocast('cuda', dtype=amp_dtype)
        else:
            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    use_scaler = (device_type == "cuda" and resolved_dtype == "float16")
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    return device_type, resolved_dtype, autocast_ctx, scaler


def run_training(args):
    """Run MiniMind-O SFT training from either a Notebook (OmniTrainConfig) or CLI (Namespace)."""
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_dir = getattr(args, 'checkpoint_dir', '../checkpoints')
    tokenizer_path = getattr(args, 'tokenizer_path', '../model')
    max_steps = getattr(args, 'max_steps', 0)
    max_samples = getattr(args, 'max_samples', 0)
    omni_config = OmniConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    ckp_data = omni_checkpoint(omni_config, weight=args.save_weight, save_dir=checkpoint_dir) if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type, resolved_dtype, autocast_ctx, scaler = _make_amp_and_scaler(args.dtype, args.device)
    Logger(f'Device: {args.device} | AMP dtype: {resolved_dtype}')
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-O-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_omni_model(omni_config, from_weight=args.from_weight,
                                        tokenizer_path=tokenizer_path,
                                        audio_encoder_path=args.audio_encoder_dir,
                                        vision_model_path=args.vision_dir,
                                        save_dir=args.save_dir, device=args.device,
                                        freeze_backbone=args.freeze_backbone, from_resume=args.from_resume)
    
    if args.use_compile == 1:
        try:
            model = torch.compile(model)
        except Exception as e:
            Logger(f'⚠️ torch.compile 不可用，自动回退 eager 模式: {e}')
    
    if model.audio_encoder is not None: model.audio_encoder.to(args.device)
    if model.vision_encoder is not None: model.vision_encoder.to(args.device)
    
    if args.mode == 'audio_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.audio_proj.parameters(): p.requires_grad = True
    elif args.mode == 'vision_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.vision_proj.parameters(): p.requires_grad = True
    log_model_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable: {trainable:.2f}M | Mode: {args.mode} | Freeze: {args.freeze_backbone} | Compile: {"on" if args.use_compile else "off"}')
    
    # scheduled_sampling 现在会自动保护 image/audio token 的连续性
    train_ds = OmniDataset(
        args.data_path, 
        tokenizer, 
        audio_processor=model.audio_processor,
        vision_processor=model.vision_processor,
        max_length=args.max_seq_len,
        image_token_len=model.config.image_token_len,
        max_samples=max_samples
    )
    Logger(f'Dataset samples: {len(train_ds)}')
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        if 'scaler' in ckp_data:
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    history = {"steps": [], "epochs": [], "loss": [], "text_loss": [], "audio_loss": [], "lr": []}
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds, batch_sampler=batch_sampler, collate_fn=omni_collate_fn,
            num_workers=args.num_workers, pin_memory=(device_type == "cuda")
        )
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            global_step, stopped = train_epoch(
                epoch, loader, len(loader) + skip, model, optimizer, scaler, autocast_ctx,
                omni_config, args, start_step=start_step, wandb=wandb, history=history,
                max_steps=max_steps, global_step=global_step
            )
        else:
            global_step, stopped = train_epoch(
                epoch, loader, len(loader), model, optimizer, scaler, autocast_ctx,
                omni_config, args, start_step=0, wandb=wandb, history=history,
                max_steps=max_steps, global_step=global_step
            )
        if stopped:
            break
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
    return history


def build_arg_parser():
    parser = argparse.ArgumentParser(description="MiniMind-O SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="断点续训目录")
    parser.add_argument('--save_weight', default='sft_omni', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="auto", help="混合精度类型 (auto/bfloat16/float16)")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2a_mini.parquet", help="训练数据路径（parquet格式）")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="Tokenizer路径")
    parser.add_argument("--audio_encoder_dir", type=str, default="../model/SenseVoiceSmall", help="音频encoder路径(SenseVoice)")
    parser.add_argument("--vision_dir", type=str, default="../model/siglip2-base-p32-256-ve", help="CLIP视觉模型路径")
    parser.add_argument('--from_weight', default='llm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_backbone', default='none', type=str, choices=['none', 'all', 'last1'], help="冻结主干模型: none=全量训练, all=只训练audio层, last1=只训练最后1层+audio层")
    parser.add_argument('--mode', default='all', type=str, choices=['all', 'audio_proj', 'vision_proj'], help="训练模式: all=全量训练, audio_proj=只训练audio_proj, vision_proj=只训练vision_proj")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-O-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--max_steps", type=int, default=0, help="最大训练步数（0=不限制，适合Notebook快速验证）")
    parser.add_argument("--max_samples", type=int, default=0, help="最多使用多少条样本（0=全量）")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_training(args)
