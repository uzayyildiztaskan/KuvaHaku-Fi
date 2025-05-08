import torch
import gc
import psutil
import os

def print_summary(stage=""):
    """Print a memory usage summary."""
    print(f"\n--- Memory Summary: {stage} ---")
    
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"CPU Memory: {cpu_mem:.2f} MB")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)    # MB
            print(f"GPU {i} Allocated: {allocated:.2f} MB")
            print(f"GPU {i} Reserved: {reserved:.2f} MB")
    
    print("----------------------------\n")

def diagnose_memory_usage(model, inputs):
    """
    Run a forward pass with the model and inputs and trace memory usage.
    Make sure inputs are on the correct device.
    """
    print("\n--- Memory Diagnosis: Forward Pass ---")
    
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"Input '{key}' is on device: {value.device}")
            if value.device != device:
                print(f"Moving '{key}' to {device}")
                inputs[key] = value.to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        before_mem = torch.cuda.memory_allocated() / (1024 * 1024)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"Memory before: {before_mem:.2f} MB")
            print(f"Memory after: {after_mem:.2f} MB")
            print(f"Peak memory: {peak_mem:.2f} MB")
            print(f"Memory used in forward pass: {after_mem - before_mem:.2f} MB")
        
        print(f"Output keys: {list(outputs.__dict__.keys() if hasattr(outputs, '__dict__') else outputs.keys() if isinstance(outputs, dict) else ['tensor'])}")
        
        if hasattr(outputs, 'embeddings'):
            print(f"Embedding shape: {outputs.embeddings.shape}")
            print(f"Embedding mean: {outputs.embeddings.mean().item()}")
        
        print("Forward pass successful!")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    print("----------------------------\n")
    
    gc.collect()
    torch.cuda.empty_cache()