import argparse
import yaml
import os
import json
import torch
import pickle  # Thêm pickle
from collections import defaultdict # Thêm defaultdict
from torch.utils.data import DataLoader, Subset
from peft import PeftModel
import sys
sys.path.append('.')

# Import project modules
from src.models.model_registry import build_model
from src.data.wad_dataset import build_dataset, WADDataset # Import class WADDataset
from src.data.data_collator import VLMDataCollator
from src.evaluation.evaluator import VLMEvaluator
from datasets import load_dataset # Import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation for Navigation VLM")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to config.yaml"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to checkpoint folder. If None, evaluates Base Model."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="eval_results.json", 
        help="Path to save results JSON"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test_alter",
        choices=["train", "valid", "test_alter", "test_QA"],
        help="Dataset split to evaluate on"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Giới hạn số lượng mẫu để test nhanh (ví dụ: 5)"
    )

    return parser.parse_args()

def prepare_auxiliary_data(config):
    """
    Hàm phụ trợ: Load frame_index và bbox giống hệt logic trong build_dataset
    để dùng cho test_alter/test_QA
    """
    print("--- Loading Auxiliary Data for Testing ---")
    
    # 1. Load frame index
    index_file = "./wad_dataset/frame_index.pkl"
    print(f"Loading frame index from {index_file}...")
    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            frame_index = pickle.load(f)
    else:
        raise FileNotFoundError(f"Frame index not found at {index_file}. Please check path.")

    # 2. Load bboxes
    print("Loading bboxes...")
    # Load từ file local nếu có, hoặc từ HF hub theo config
    bbox_file = "all_bboxes_1.jsonl"
    if os.path.exists(bbox_file):
        bbox_dataset = load_dataset("json", data_files=bbox_file, split="train")
    else:
        bbox_dataset = load_dataset(config['data']['name'], data_files="all_bboxes_1.jsonl", split="train")

    bbox_by_folder = defaultdict(lambda: defaultdict(list))
    for bbox_entry in bbox_dataset:
        folder_id = bbox_entry['folder_id']
        frame_id = bbox_entry['frame_id']
        
        bbox_by_folder[folder_id][frame_id].append({
            'label': bbox_entry['label'],
            'confidence': bbox_entry['probs'],
            'bbox': bbox_entry['boxs']
        })
    
    return frame_index, bbox_by_folder

def main():
    args = parse_args()
    
    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # FIX: Tắt LoRA trong config khi có checkpoint
    if args.checkpoint:
        print("\n  Disabling LoRA in config (will load from checkpoint)")
        if 'model' in config and 'lora' in config['model']:
            config['model']['lora']['enabled'] = False
    
    # 2. Build Base Model
    print("Building Base Model...")
    vlm_wrapper = build_model(config)
    
    model = vlm_wrapper.model
    tokenizer = vlm_wrapper.tokenizer
    processor = vlm_wrapper.processor
    
    # 3. Load LoRA từ checkpoint
    if args.checkpoint:
        print("\n" + "="*60)
        print("MODE: FINE-TUNED MODEL (LoRA)")
        print(f"Checkpoint: {args.checkpoint}")
        print("="*60 + "\n")
        
        if not os.path.exists(args.checkpoint):
            raise ValueError(f"Checkpoint not found: {args.checkpoint}")
        
        # Validate files logic (giữ nguyên của bạn)
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        missing_files = [f for f in required_files 
                        if not os.path.exists(os.path.join(args.checkpoint, f))]
        
        if missing_files:
            if 'adapter_model.safetensors' in missing_files:
                if os.path.exists(os.path.join(args.checkpoint, 'adapter_model.bin')):
                    missing_files.remove('adapter_model.safetensors')
            if missing_files:
                raise ValueError(f"Missing files: {missing_files}")
        
        print("Loading LoRA adapter...")
        try:
            model = PeftModel.from_pretrained(
                model,
                args.checkpoint,
                torch_dtype=torch.bfloat16 if config['training'].get('bf16', False) 
                           else torch.float16 if config['training']['fp16'] 
                           else torch.float32,
                is_trainable=False
            )
            print("✓ LoRA Adapter loaded successfully.")
        except Exception as e:
            print(f" Error loading adapter: {e}")
            raise
            
    else:
        print("\n" + "="*60)
        print("MODE: BASE MODEL (Zero-shot)")
        print("="*60 + "\n")
    
    # Set device
    device = config.get('hardware', {}).get('device', 
             'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"✓ Model ready on device: {device}\n")

    # 4. Prepare Dataset
    print(f"Loading dataset split: {args.split}...")
    
    if args.split in ["train", "valid"]:
        # Dùng build_dataset có sẵn cho train/valid
        train_dataset, valid_dataset = build_dataset(config, processor, tokenizer)
        if args.split == "train":
            target_dataset = train_dataset
        else:
            target_dataset = valid_dataset
            
    else:  # test_alter hoặc test_QA
        # --- PHẦN ĐÃ SỬA ---
        
        # A. Load dữ liệu phụ trợ (Bắt buộc cho WADDataset)
        frame_index, bbox_by_folder = prepare_auxiliary_data(config)

        # B. Load file metadata tương ứng
        if args.split == "test_alter":
            data_file = "test_alter.json"
        elif args.split == "test_QA":
            data_file = "test_QA.json"
        
        print(f"Loading metadata from {data_file}...")
        # Load dataset dict: {'test': ...}
        dataset_dict = load_dataset(
            config['data']['name'],
            data_files={
                "test": data_file
            }
        )
        
        # C. Xác định image_size (Logic từ build_dataset)
        architecture = config['model']['architecture']
        if architecture == 'qwen':
            image_size = None
        else:
            image_size = tuple(config['model']['vision']['image_size'])

        # D. Khởi tạo WADDataset (Đúng tham số)
        target_dataset = WADDataset(
            metadata_dataset=dataset_dict,   # Truyền dataset dict vào
            frame_index=frame_index,         # Cần thiết
            bbox_by_folder=bbox_by_folder,   # Cần thiết
            processor=processor,
            tokenizer=tokenizer,
            split='test',                    # Key để lấy data trong metadata_dataset
            num_frames=config['data'].get('num_frames', 1),
            image_size=image_size
        )
    
    print(f"Split: {args.split}")
    print(f"Number of evaluation samples: {len(target_dataset)}")
    
    if args.max_samples is not None:
        if args.max_samples < len(target_dataset):
            print(f"\n[INFO] QUICK TEST MODE ON: Cắt dataset xuống còn {args.max_samples} mẫu đầu tiên...")
            target_dataset = Subset(target_dataset, range(args.max_samples))
        else:
            print(f"\n[INFO] max_samples ({args.max_samples}) lớn hơn tổng data. Sẽ chạy toàn bộ.")

    # 5. Setup DataLoader
    print("Setting up DataLoader (batch_size=1)...")
    data_collator = VLMDataCollator(tokenizer=tokenizer)
    
    eval_dataloader = DataLoader(
        target_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True
    )

    # 6. Initialize Evaluator
    print("Initializing Evaluator...")
    evaluator = VLMEvaluator(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        config=config
    )

    # 7. Run Evaluation
    print("Starting Evaluation Loop...")
    mode_name = "LoRA_Finetuned" if args.checkpoint else "Base_Model"
    
    metrics, predictions, references = evaluator.evaluate_dataset(
        eval_dataloader, 
        task_name=mode_name,
        print_samples=args.max_samples if args.max_samples else 5
    )

    # 8. Save Detailed Results
    output_path = args.output_file
    print(f"Saving results to {output_path}...")
    
    detailed_samples = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        detailed_samples.append({
            "id": i,
            "ground_truth": ref,
            "prediction": pred,
            "exact_match": pred.strip() == ref.strip()
        })

    final_results = {
        "mode": mode_name,
        "config_file": args.config,
        "checkpoint_path": args.checkpoint,
        "dataset_split": args.split,
        "metrics": metrics,
        "samples": detailed_samples
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
        
    print("\nEVALUATION COMPLETED")
    print(f"  Dataset Split: {args.split}")
    print(f"  Result File: {output_path}")
    print("\n--- Metrics ---")
    print(f"ROUGE-1: {metrics.get('ROUGE-1', 0):.2f}")
    print(f"  ROUGE-2: {metrics.get('ROUGE-2', 0):.2f}")
    print(f"  ROUGE-L: {metrics.get('ROUGE-L', 0):.2f}")
    print(f"  TF-IDF:  {metrics.get('TF-IDF', 0):.2f}")

if __name__ == "__main__":
    main()