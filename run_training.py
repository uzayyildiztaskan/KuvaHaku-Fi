import torch
import gc
import os
import multiprocessing
from memory_diagnoser import print_summary, diagnose_memory_usage
from model import load_model
from dataset import load_combined_dataset
from train import train_model
from config import *
from upload import upload_to_huggingface

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Initial system state:")
    print_summary("Before loading anything")
    
    print("\nLoading dataset...")
    dataset_splits = load_combined_dataset(DATASET_PATH)
    print(f"Train set: {len(dataset_splits['train'])} examples")
    print(f"Test set: {len(dataset_splits['test'])} examples")
    
    print_summary("After dataset loading")
    
    print("\nLoading model...")
    model, processor = load_model()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    
    print_summary("After model loading")
    
    try:
        print("\nRunning memory diagnostics...")
        sample_batch = [dataset_splits["train"][0]]
        from collator import collate_fn
        
        print(f"Sample batch contains Finnish caption: {sample_batch[0]['finnish_caption'][0][:30]}...")
        
        sample_inputs = collate_fn(sample_batch, processor, device)
        
        for key, value in sample_inputs[0].items():
            if isinstance(value, torch.Tensor):
                print(f"Input tensor '{key}' is on device: {value.device}")
                
        print("Skipping detailed diagnostics to avoid CUDA initialization in parent process")
        
        print("\nStarting training...")
        trainer, model = train_model()
        
        print_summary("After training")
        
        print("\nSaving model...")
        model.save_pretrained(OUTPUT_DIR)
        print("Training complete!")

        upload_to_huggingface(trainer)
        
    except Exception as e:
        print(f"Error encountered: {e}")
        print_summary("Memory state at error")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        main()
    except Exception as e:
        print(f"Error encountered in main: {e}")
        print_summary("Memory state at error")
        import traceback
        traceback.print_exc()