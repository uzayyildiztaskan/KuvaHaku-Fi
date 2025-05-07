from model import load_model
from config import TOP_K
import torch

def search(texts, images):
    model, processor = load_model()
    model.eval()

    with torch.no_grad():
        text_embeddings = model(**processor(text=texts, return_tensors="pt").to(model.device)).embeddings
        image_embeddings = model(**processor(images=images, return_tensors="pt").to(model.device)).embeddings

    scores = processor.score_retrieval(text_embeddings, image_embeddings)
    topk_indices = scores.topk(TOP_K, dim=1).indices
    return topk_indices
