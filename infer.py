from model import load_model
from transformers import ColPaliProcessor
from datasets import load_dataset
from peft import PeftModel
import torch
from config import TOP_K, MODEL_PATH
import matplotlib.pyplot as plt

def load_dataset_with_limit(max_images=750):
    print(f"Loading dataset with limit of {max_images} images...")
    flickr_ds = load_dataset("nlphuji/flickr30k", split="test", keep_in_memory=False)
    
    if len(flickr_ds) > max_images:
        start_idx = max(0, len(flickr_ds) - max_images)
        flickr_ds = flickr_ds.select(range(start_idx, len(flickr_ds)))
    
    print(f"Dataset loaded with {len(flickr_ds)} images.")
    return flickr_ds

def search(texts, images):
    base_model, _ = load_model()

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2-hf")

    model.eval()

    with torch.no_grad():
        text_inputs = processor(text=texts, return_tensors="pt").to(model.device)
        text_embeddings = model(**text_inputs).embeddings
        
        batch_size = 8
        image_embeddings_list = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_inputs = processor(images=batch, return_tensors="pt").to(model.device)
            batch_embeddings = model(**batch_inputs).embeddings
            image_embeddings_list.append(batch_embeddings)
            
            torch.cuda.empty_cache()
            
        image_embeddings = torch.cat(image_embeddings_list, dim=0)

    scores = processor.score_retrieval(text_embeddings, image_embeddings)
    topk_indices = scores.topk(min(5, len(images)), dim=1).indices
    
    return topk_indices

if __name__ == "__main__":

    base_model, _ = load_model()

    flickr_ds = load_dataset_with_limit(750)
    
    images = [img["image"].convert("RGB") for img in flickr_ds]
    
    queries = [
        "Mies hyppää ilmaan rannalla",
        "Nainen istuu kahvilassa ja lukee kirjaa",
        "Koira juoksee nurmikolla pallon kanssa",
        "Lapset leikkivät lumessa talvipäivänä",
        "Aurinko laskee vuorten taakse järven yllä"
    ]
    
    print(f"Running search with {len(queries)} queries on {len(images)} images...")
    topk_indices = search(queries, images)
    
    # Display results
    for q_idx, query in enumerate(queries):
        print(f"\nQuery: {query}")
        for rank, idx in enumerate(topk_indices[q_idx]):
            plt.figure()
            plt.imshow(images[idx])
            plt.title(f"Rank {rank+1}")
            plt.axis("off")
            plt.show()
