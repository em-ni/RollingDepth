"""
python train_rollingdepth.py \
    --dataset_dir data/em/train \
    --val_dataset_dir data/em/val \
    --output_dir ./finetuned \
    --batch_size 4 \
    --num_epochs 10 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2 \
    --depth_range 0.1 1000.0 \
    --depth_map_factor 5000.0 \
    --mixed_precision fp32 2>&1 
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import einops
from tqdm.auto import tqdm
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from rollingdepth import RollingDepthPipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDepthDataset(Dataset):
    """
    Dataset for video depth estimation finetuning.
    
    Expects directory structure:
    dataset/
        ├── video_001/
        │   ├── rgb/
        │   │   ├── frame_000.png
        │   │   └── ...
        │   └── depth/
        │       ├── frame_000.npy (or .png)
        │       └── ...
        └── ...
    """
    
    def __init__(
        self,
        dataset_dir: str,
        snippet_len: int = 3,
        max_frames_per_video: int = 0,
        max_total_frames: int = 0,
        image_size: Tuple[int, int] = (512, 512),
        depth_range: Tuple[float, float] = (0.1, 1000.0),
        depth_map_factor: float = 5000.0,
        normalize_depth: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.snippet_len = snippet_len
        self.max_frames_per_video = max_frames_per_video
        self.image_size = image_size
        self.depth_range = depth_range
        self.depth_map_factor = depth_map_factor
        self.normalize_depth = normalize_depth
        
        # Collect all video sequences
        self.sequences = []
        self.total_frames = 0
        
        self._collect_sequences()
        
        if max_total_frames > 0:
            self.total_frames = min(self.total_frames, max_total_frames)
            logger.info(f"Limited to {self.total_frames} total frames")
    
    def _collect_sequences(self):
        """Collect all video sequences from dataset directory"""
        video_dirs = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir()])
        
        for video_dir in video_dirs:
            rgb_dir = video_dir / "rgb"
            depth_dir = video_dir / "depth"
            
            if not (rgb_dir.exists() and depth_dir.exists()):
                logger.warning(f"Skipping {video_dir.name}: missing rgb or depth directory")
                continue
            
            # Get all rgb frames
            rgb_frames = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
            depth_frames = sorted(depth_dir.glob("*.npy")) + sorted(depth_dir.glob("*.png"))
            
            if len(rgb_frames) != len(depth_frames):
                logger.warning(
                    f"Mismatch in {video_dir.name}: "
                    f"{len(rgb_frames)} RGB frames vs {len(depth_frames)} depth frames"
                )
            
            num_frames = min(len(rgb_frames), len(depth_frames))
            if num_frames < self.snippet_len:
                logger.warning(
                    f"Skipping {video_dir.name}: only {num_frames} frames "
                    f"(need at least {self.snippet_len})"
                )
                continue
            
            # Limit frames per video if specified
            if self.max_frames_per_video > 0:
                num_frames = min(num_frames, self.max_frames_per_video)
            
            self.sequences.append({
                'video_dir': video_dir,
                'rgb_frames': rgb_frames[:num_frames],
                'depth_frames': depth_frames[:num_frames],
                'num_frames': num_frames,
            })
            self.total_frames += num_frames
        
        logger.info(
            f"Collected {len(self.sequences)} sequences, "
            f"{self.total_frames} total frames"
        )
    
    def __len__(self):
        # Return number of possible snippets
        return sum(max(0, seq['num_frames'] - self.snippet_len + 1) 
                   for seq in self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which sequence and position this index corresponds to
        current_idx = 0
        for seq in self.sequences:
            num_snippets = max(0, seq['num_frames'] - self.snippet_len + 1)
            if current_idx + num_snippets > idx:
                # This snippet is in this sequence
                snippet_idx = idx - current_idx
                return self._get_snippet(seq, snippet_idx)
            current_idx += num_snippets
        
        raise IndexError(f"Index {idx} out of range")
    
    def _get_snippet(self, seq: Dict, snippet_idx: int) -> Dict[str, torch.Tensor]:
        """Get a snippet of frames from a sequence"""
        # Select frame indices
        start_idx = snippet_idx
        end_idx = start_idx + self.snippet_len
        frame_indices = list(range(start_idx, end_idx))
        
        # Load RGB frames
        rgb_frames = []
        for i in frame_indices:
            rgb_path = seq['rgb_frames'][i]
            rgb = self._load_image(rgb_path)  # [3, H, W] in [0, 1]
            rgb_frames.append(rgb)
        rgb_frames = torch.stack(rgb_frames, dim=0)  # [T, 3, H, W]
        
        # Load depth frames
        depth_frames = []
        for i in frame_indices:
            depth_path = seq['depth_frames'][i]
            depth = self._load_depth(depth_path)  # [1, H, W]
            depth_frames.append(depth)
        depth_frames = torch.stack(depth_frames, dim=0)  # [T, 1, H, W]
        
        # Normalize to [-1, 1] as per RollingDepth convention
        if self.normalize_depth:
            depth_frames = self._normalize_depth(depth_frames)
        
        return {
            'rgb': rgb_frames,  # [T, 3, H, W] in [0, 1]
            'depth': depth_frames,  # [T, 1, H, W] in [-1, 1]
            'video_name': str(seq['video_dir'].name),
        }
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess image"""
        from PIL import Image
        
        image = Image.open(image_path).convert('RGB')
        # Resize to target size
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        # Convert to tensor [0, 1]
        image = torch.from_numpy(np.array(image)).float() / 255.0
        # Normalize to [-1, 1] as per diffusers convention
        image = image * 2.0 - 1.0
        # Rearrange to [C, H, W]
        image = einops.rearrange(image, 'h w c -> c h w')
        
        return image
    
    def _load_depth(self, depth_path: Path) -> torch.Tensor:
        """Load depth map and recover RollingDepth [-1, 1] format
        
        Inverse of create_orb_slam_input.py encoding:
        1. uint16 PNG ÷ depth_map_factor → depth_linear (meters)
        2. Reverse linearization formula
        3. Invert normalized depth
        4. Result: [0, 1] range ready for final normalization
        """
        if depth_path.suffix == '.npy':
            # NPY files are assumed to be already in the correct format
            depth = np.load(depth_path)
        else:  # .png (uint16 saved by simulator via create_orb_slam_input.py)
            from PIL import Image
            depth_img = Image.open(depth_path)
            depth_uint16 = np.array(depth_img).astype(np.float32)
            
            # STEP 1: Recover depth_linear from uint16
            # (inverse of: depth_scaled = depth_linear * depth_map_factor)
            depth_linear = depth_uint16 / self.depth_map_factor
            
            # STEP 2: Reverse the linearization formula
            # Original: depth_linear = (2*np*fp) / (fp+np - (2*depth_normalized-1)*(fp-np))
            # Solve for depth_normalized:
            # depth_linear * (fp+np - (2*depth_normalized-1)*(fp-np)) = 2*np*fp
            # depth_linear * (fp+np) - depth_linear*(2*depth_normalized-1)*(fp-np) = 2*np*fp
            # -depth_linear*(2*depth_normalized-1)*(fp-np) = 2*np*fp - depth_linear*(fp+np)
            # (2*depth_normalized-1) = (2*np*fp - depth_linear*(fp+np)) / (depth_linear*(fp-np))
            # depth_normalized = ((2*np*fp - depth_linear*(fp+np)) / (depth_linear*(fp-np)) + 1) / 2
            near_plane = self.depth_range[0]
            far_plane = self.depth_range[1]
            
            # Avoid division by zero
            depth_linear = np.clip(depth_linear, 1e-6, np.inf)
            
            numerator = 2.0 * near_plane * far_plane - depth_linear * (far_plane + near_plane)
            denominator = depth_linear * (far_plane - near_plane)
            depth_normalized = (numerator / denominator + 1.0) / 2.0
            depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
            
            # STEP 3: Invert back (reverse of: depth_normalized = 1.0 - depth_normalized)
            depth_normalized = 1.0 - depth_normalized
            
            depth = depth_normalized
        
        # Handle different array shapes
        if depth.ndim == 3:
            depth = depth[..., 0]  # Take first channel if RGB
        
        # Resize to target size
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)  # [1, H, W]
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0),
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # [1, H, W]
        
        return depth_tensor
    
    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize depth from [0, 1] to [-1, 1]
        
        After _load_depth(), depth is in [0, 1] range.
        This final step scales to [-1, 1] for RollingDepth training.
        """
        depth = depth * 2.0 - 1.0
        return depth


class RollingDepthTrainer:
    """Trainer for finetuning RollingDepth"""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        num_epochs: int = 10,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = 'fp16',
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        device: str = 'cuda',
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.use_lora = use_lora
        self.device = device
        
        # Setup mixed precision (disable autocast if using GradScaler)
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision == 'fp16' else None
        self.use_autocast = False  # Autocast incompatible with GradScaler for fp16
        
        # Load model
        logger.info(f"Loading model from {model_name}")
        self.pipeline: RollingDepthPipeline = RollingDepthPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if mixed_precision == 'fp16' else torch.float32
        )
        self.pipeline = self.pipeline.to(device)
        
        # Get UNet
        self.unet = self.pipeline.unet
        
        # Apply LoRA if specified
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)
        
        # Freeze VAE and text encoder (only train UNet)
        self._freeze_encoder_and_vae()
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Setup scheduler
        self.lr_scheduler = None
        
        # Encode empty text embedding for conditioning
        self.pipeline.encode_empty_text()
        
        # Training state
        self.global_step = 0
        self.training_history = {
            'loss': [],
            'learning_rate': [],
        }
    
    def _apply_lora(self, rank: int, alpha: int):
        """Apply LoRA to UNet attention layers"""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            logger.error("peft package not found. Install with: pip install peft")
            raise
        
        logger.info(f"Applying LoRA with rank={rank}, alpha={alpha}")
        
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["to_q", "to_v", "to_k"],
            lora_dropout=0.05,
            bias="none",
        )
        
        self.unet = get_peft_model(self.unet, peft_config)
        self.pipeline.unet = self.unet
        
        logger.info(f"LoRA applied. Trainable parameters: {self._count_trainable_params()}")
    
    def _freeze_encoder_and_vae(self):
        """Freeze VAE and text encoder"""
        for param in self.pipeline.vae.parameters():
            param.requires_grad = False
        for param in self.pipeline.text_encoder.parameters():
            param.requires_grad = False
    
    def _count_trainable_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop"""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Trainable parameters: {self._count_trainable_params()}")
        logger.info(f"Batch size: {self.batch_size} (effective: {self.batch_size * self.gradient_accumulation_steps})")
        
        # Setup learning rate scheduler
        total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_loss = self._train_epoch(train_dataloader)
            logger.info(f"Train loss: {train_loss:.6f}")
            
            # Validate
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                logger.info(f"Val loss: {val_loss:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"Best model saved!")
            else:
                self._save_checkpoint(epoch, is_best=False)
            
            # Record history
            self.training_history['loss'].append(train_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save history
            self._save_history()
        
        logger.info(f"\nTraining completed! Best checkpoint saved to {self.output_dir}")
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.unet.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training", leave=True, unit="batch")
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                rgb = batch['rgb'].to(self.device)  # [B, T, 3, H, W]
                depth_gt = batch['depth'].to(self.device)  # [B, T, 1, H, W]
                
                # Normalize RGB to [-1, 1] for diffusers
                rgb_normalized = rgb  # Already normalized in dataset
                
                # Forward pass: compute loss
                loss = self._compute_loss(rgb_normalized, depth_gt)
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    
                    self.global_step += 1
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                avg_batch_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar with more info
                pbar.set_postfix({
                    'loss': f'{avg_batch_loss:.4f}',
                    'step': self.global_step,
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                raise
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def _compute_loss(self, rgb: torch.Tensor, depth_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute depth prediction loss.
        
        This is a simplified loss that compares predicted and ground truth depth.
        In practice, you might want to add:
        - Temporal consistency loss
        - Photometric loss
        - Smoothness loss
        """
        B, T, _, H, W = rgb.shape
        
        # Get the dtype from the model
        dtype = next(self.unet.parameters()).dtype
        
        # Encode RGB frames (keep 5D for encode_rgb)
        with torch.no_grad():
            rgb_latent = self.pipeline.encode_rgb(rgb, max_batch_size=B, verbose=False)
        # rgb_latent is already [B, T, C, h, w]
        
        # Add noise to ground truth depth (simulate diffusion process)
        with torch.no_grad():
            # Sample random timesteps
            timesteps = torch.randint(
                0, self.pipeline.scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()
            
            # Convert depth to 3-channel (repeat across channels) for VAE encoding
            depth_gt_3ch = depth_gt.repeat(1, 1, 3, 1, 1)  # [B, T, 3, H, W]
            
            # Encode ground truth depth (keep 5D)
            depth_gt_latent = self.pipeline.encode_rgb(depth_gt_3ch, max_batch_size=B, verbose=False)
            # depth_gt_latent is already [B, T, C, h, w]
            
            # Sample noise
            noise = torch.randn_like(depth_gt_latent)
            
            # Add noise (forward diffusion)
            noisy_depth_latent = self.pipeline.scheduler.add_noise(
                depth_gt_latent, noise, timesteps[0]
            )
        
        # Cast latents to model dtype
        rgb_latent = rgb_latent.to(dtype)
        noisy_depth_latent = noisy_depth_latent.to(dtype)
        noise = noise.to(dtype)
        
        # Expand timesteps to match flattened batch size (B*T)
        timesteps_expanded = timesteps.repeat_interleave(T)  # (B,) -> (B*T,)
        
        # Flatten latents for UNet call without cross-frame attention
        # Reshape from [B, T, C, h, w] to [B*T, C, h, w]
        B_orig = rgb_latent.shape[0]
        rgb_latent_flat = einops.rearrange(rgb_latent, "b t c h w -> (b t) c h w")
        noisy_depth_latent_flat = einops.rearrange(noisy_depth_latent, "b t c h w -> (b t) c h w")
        
        # Concat rgb and depth latents
        unet_input = torch.cat([rgb_latent_flat, noisy_depth_latent_flat], dim=1)  # [B*T, 8, h, w]
        
        # Prepare encoder hidden states - replicate for B*T
        encoder_hidden_states = self.pipeline.empty_text_embed.to(dtype).to(self.device)
        if encoder_hidden_states is not None:
            # encoder_hidden_states should be [1, seq_len, embedding_dim], expand to [B*T, seq_len, embedding_dim]
            encoder_hidden_states = encoder_hidden_states.expand(B_orig * 3, -1, -1)  # Replicate for all timesteps
        
        # Predict noise (UNet forward pass)
        # Call UNet directly without cross-frame attention (num_view=1)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            noise_pred = self.unet(
                unet_input,
                timesteps_expanded,
                encoder_hidden_states=encoder_hidden_states,
                num_view=1,  # Disable cross-frame attention
            ).sample  # [B*T, 4, h, w]
        
        # Reshape back to [B, T, 4, h, w]
        noise_pred = einops.rearrange(noise_pred, "(b t) c h w -> b t c h w", b=B_orig)
        
        # MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def _validate(self, dataloader: DataLoader) -> float:
        """Validate on validation set"""
        self.unet.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False, unit="batch")
            for batch in pbar:
                rgb = batch['rgb'].to(self.device)
                depth_gt = batch['depth'].to(self.device)
                
                loss = self._compute_loss(rgb, depth_gt)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        save_dir = self.output_dir / (f"checkpoint_best" if is_best else f"checkpoint_epoch_{epoch}")
        save_dir.mkdir(exist_ok=True)
        
        # Save UNet
        self.unet.save_pretrained(str(save_dir / "unet"))
        
        # Save full pipeline (for easier inference)
        self.pipeline.save_pretrained(str(save_dir))
        
        # Save training config
        config = {
            'epoch': epoch,
            'global_step': self.global_step,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'use_lora': self.use_lora,
            'mixed_precision': self.mixed_precision,
        }
        with open(save_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Checkpoint saved to {save_dir}")
    
    def _save_history(self):
        """Save training history"""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Finetune RollingDepth model")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to training dataset directory"
    )
    parser.add_argument(
        "--val_dataset_dir",
        type=str,
        default=None,
        help="Path to validation dataset directory (optional)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="prs-eth/rollingdepth-v1-0",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for finetuned model"
    )
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", choices=['fp16', 'fp32'], default='fp16')
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient training")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Dataset processing arguments
    parser.add_argument("--snippet_len", type=int, default=3)
    parser.add_argument("--max_frames_per_video", type=int, default=0)
    parser.add_argument("--max_total_frames", type=int, default=0)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--depth_range", type=float, nargs=2, default=[0.1, 1000.0], help="Depth range in meters (near, far)")
    parser.add_argument("--depth_map_factor", type=float, default=5000.0, help="Depth scale factor used by simulator (e.g., 5000 means 1m = 5000 uint16 units)")
    
    # Other arguments
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = VideoDepthDataset(
        dataset_dir=args.dataset_dir,
        snippet_len=args.snippet_len,
        max_frames_per_video=args.max_frames_per_video,
        max_total_frames=args.max_total_frames,
        image_size=tuple(args.image_size),
        depth_range=tuple(args.depth_range),
        depth_map_factor=args.depth_map_factor,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_dataloader = None
    if args.val_dataset_dir is not None:
        val_dataset = VideoDepthDataset(
            dataset_dir=args.val_dataset_dir,
            snippet_len=args.snippet_len,
            max_frames_per_video=args.max_frames_per_video,
            image_size=tuple(args.image_size),
            depth_range=tuple(args.depth_range),
            depth_map_factor=args.depth_map_factor,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    
    # Create trainer
    trainer = RollingDepthTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
    )
    
    # Train
    trainer.train(train_dataloader, val_dataloader)
    
    logger.info("Finetuning completed!")


if __name__ == "__main__":
    main()
