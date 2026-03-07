import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional
import yaml
import torch
from tqdm.auto import tqdm
import gc
import warnings

class VLMTrainer:
    """High-level training orchestration"""
    
    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None):
        warnings.filterwarnings('ignore', message='.*Unused or unrecognized kwargs.*')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.trainer = None
        self.results = {}
        self.pbar = None  # Progress bar
    
    def setup(self):
        """Setup model, data, trainer"""
        self._clear_memory()

        from ..models.model_registry import build_model
        from ..data.wad_dataset import build_dataset
        from .callbacks import MemoryOptimizationCallback, ExperimentTrackingCallback
        from ..data.data_collator import VLMDataCollator
        
        config = self.config.copy()
        if self.checkpoint_path:
            print("\n  checkpoint_path detected: Disabling LoRA in config")
            print("   (LoRA will be loaded from checkpoint instead)\n")
            if 'model' in config and 'lora' in config['model']:
                config['model']['lora']['enabled'] = False

        # Tạo progress bar cho setup
        setup_steps = ["Building model", "Building dataset", "Creating trainer"]
        if self.checkpoint_path:
            setup_steps.insert(1, "Loading checkpoint")  # Thêm bước load checkpoint
            
        with tqdm(total=len(setup_steps), desc="Setup Progress") as pbar:
            # Build model
            pbar.set_description("Building model...")
            vlm = build_model(self.config)
            self.model = vlm.model
            pbar.update(1)
            
            # ✨ Load checkpoint nếu có
            if self.checkpoint_path:
                pbar.set_description("Loading checkpoint...")
                self._load_checkpoint(self.checkpoint_path)
                pbar.update(1)
            
            # Build dataset
            pbar.set_description("Building dataset...")
            train_dataset, eval_dataset = build_dataset(
                self.config,
                vlm.processor,
                vlm.tokenizer
            )
            pbar.update(1)
            
            # Training arguments
            pbar.set_description("Creating trainer...")
            training_args = TrainingArguments(
                output_dir=self.config['training']['output_dir'],
                num_train_epochs=self.config['training']['num_epochs'],
                per_device_train_batch_size=self.config['training']['batch_size'],
                gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
                learning_rate=float(self.config['training']['learning_rate']),
                warmup_steps=int(self.config['training']['warmup_steps']),
                weight_decay=float(self.config['training']['weight_decay']),
                fp16=self.config['training']['fp16'],
                bf16=self.config['training'].get('bf16', False),
                bf16_full_eval=self.config['training'].get('bf16_full_eval', self.config['training'].get('bf16', False)),
                gradient_checkpointing=self.config['training']['gradient_checkpointing'],
                logging_steps=self.config['training']['logging_steps'],
                eval_strategy=self.config['training'].get('eval_strategy', 'steps'),
                per_device_eval_batch_size=self.config['training'].get('per_device_eval_batch_size', 1),
                eval_accumulation_steps=self.config['training'].get('eval_accumulation_steps', 1),
                eval_steps=self.config['training']['eval_steps'],
                save_steps=self.config['training']['save_steps'],
                save_total_limit=self.config['training']['save_total_limit'],
                remove_unused_columns=False,
                dataloader_pin_memory=self.config['hardware']['pin_memory'],
                dataloader_num_workers=self.config['hardware']['num_workers'],
                report_to="none",
                optim=self.config['training']['optimizer'],
                disable_tqdm=False,
            )
            
            data_collator = VLMDataCollator(tokenizer=vlm.tokenizer)
                
            # Callbacks
            callbacks = [
                MemoryOptimizationCallback(
                    clear_cache_steps=25,
                    log_memory_steps=10
                ),
            ]
            
            if self.config['tracking']['enabled']:
                callbacks.append(ExperimentTrackingCallback(self.config))
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=vlm.processor,
                data_collator=data_collator,
                callbacks=callbacks
            )
            pbar.update(1)
        
        print("✓ Setup complete!")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """✨ Load checkpoint để train tiếp"""
        from peft import PeftModel
        
        print(f"\n{'='*80}")
        print(f"LOADING CHECKPOINT FROM: {checkpoint_path}")
        print(f"{'='*80}\n")
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path không tồn tại: {checkpoint_path}")
        
        # Kiểm tra các file cần thiết
        required_files = ['adapter_model.safetensors', 'adapter_config.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(checkpoint_path, f))]
        
        if missing_files:
            raise ValueError(f"Thiếu các file: {missing_files}")
        
        # Load adapter vào model
        print("Loading adapter weights...")
        self.model = PeftModel.from_pretrained(
            self.model,
            checkpoint_path,
            is_trainable=True  # Quan trọng: cho phép train tiếp
        )
        
        print("✓ Checkpoint loaded successfully!")
        print(f"  - Adapter weights: ✓")
        print(f"  - Training state: ✓")
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Run training
        
        Args:
            resume_from_checkpoint: Path to checkpoint để resume training state
            (optimizer, scheduler, training steps)
        """
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        self._clear_memory()
        
        # Nếu có checkpoint_path và muốn resume đầy đủ (bao gồm optimizer, scheduler)
        if resume_from_checkpoint:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            self.trainer.train()
        
        print("\n✓ Training complete!")
        self._clear_memory()

    def evaluate(self):
        """Run evaluation"""
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80 + "\n")
        self._clear_memory()
        results = self.trainer.evaluate()
        self.results = results
        
        print(results)
        self._clear_memory() 
        return results
    
    def save(self, output_path: str):
        """Save model"""
        self._clear_memory()
        print(f"\nSaving model to {output_path}...")
        with tqdm(total=1, desc="Saving model") as pbar:
            self.trainer.save_model(output_path)
            pbar.update(1)
        print(f"✓ Model saved to {output_path}")
        self._clear_memory()
        
    def _clear_memory(self):
        """Clear GPU and CPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()