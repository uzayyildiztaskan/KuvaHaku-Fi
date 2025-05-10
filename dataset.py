from datasets import Dataset, load_dataset, Image
from config import LAZY_LOADING

def load_combined_dataset(DATASET_PATH):
    fi_ds = load_dataset(DATASET_PATH, split="train", 
                         keep_in_memory=False)
    
    flickr_ds = load_dataset("nlphuji/flickr30k", split="test",
                            keep_in_memory=False)
    
    flickr_index = {ex["img_id"]: i for i, ex in enumerate(flickr_ds)}
    
    def attach_image(example):
        idx = flickr_index.get(example["img_id"])
        
        if idx is None:
            example["image"] = None
            return example
            
        if LAZY_LOADING:
            example["flickr_idx"] = idx
            example["image"] = None
        else:
            example["image"] = flickr_ds[idx]["image"]
        
        return example
    
    fi_ds = fi_ds.map(attach_image, batched=False)
    
    if LAZY_LOADING:
        fi_ds = fi_ds.filter(lambda x: "flickr_idx" in x and x["finnish_caption"] is not None)
    else:
        fi_ds = fi_ds.filter(lambda x: x["image"] is not None and x["finnish_caption"] is not None)
    
    if not LAZY_LOADING:
        fi_ds = fi_ds.cast_column("image", Image())
    
    return fi_ds.train_test_split(test_size=0.05)

def get_image_for_example(example, flickr_ds):
    """Lazily load an image when needed"""
    if "image" in example and example["image"] is not None:
        return example["image"]
    
    if "flickr_idx" in example:
        return flickr_ds[example["flickr_idx"]]["image"].convert("RGB")
    
    return None