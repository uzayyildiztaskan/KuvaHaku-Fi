import torch
from config import MAX_LENGTH, LAZY_LOADING
from dataset import get_image_for_example
from datasets import load_dataset

_flickr_ds_cache = None

def get_flickr_dataset():
    global _flickr_ds_cache
    if _flickr_ds_cache is None:
        _flickr_ds_cache = load_dataset("nlphuji/flickr30k", split="test")
    return _flickr_ds_cache

def collate_fn(examples, processor, device):
    """
    Collate function that ensures all tensors end up on the same device as the model.
    """
    flickr_ds = get_flickr_dataset() if LAZY_LOADING else None
    
    texts = [ex["finnish_caption"][0][:MAX_LENGTH] for ex in examples]
    
    if LAZY_LOADING:
        images = [get_image_for_example(ex, flickr_ds).convert("RGB") for ex in examples]
    else:
        images = [ex["image"].convert("RGB") for ex in examples]
    
    with torch.no_grad():
        batch_texts = processor(
            text=texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        batch_images = processor(
            images=images, 
            return_tensors="pt"
        )
        
        for key, value in batch_texts.items():
            if isinstance(value, torch.Tensor):
                batch_texts[key] = value.to(device)
                
        for key, value in batch_images.items():
            if isinstance(value, torch.Tensor):
                batch_images[key] = value.to(device)
    
    del images
    torch.cuda.empty_cache()
    
    return (batch_texts, batch_images)