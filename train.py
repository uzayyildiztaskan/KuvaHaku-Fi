from transformers import TrainingArguments
from transformers import Trainer
from model import load_model
from dataset import load_combined_dataset
from collator import collate_fn
from config import *
import torch
import gc
from tqdm import tqdm
import os
import multiprocessing
import torch.nn.functional as F
from custom_loss import EnhancedColbertPairwiseCELoss, MaxMarginLoss


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)


class MemoryEfficientTrainer(Trainer):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        query_inputs, doc_inputs = inputs
        
        with torch.cuda.amp.autocast(enabled=MIXED_PRECISION=="fp16"):
            query_out = model(**query_inputs)
            
            doc_out = model(**doc_inputs)
            
            if query_out.embeddings.shape[0] > 0 and doc_out.embeddings.shape[0] > 0:
                if torch.isnan(query_out.embeddings).any() or torch.isnan(doc_out.embeddings).any():
                    print("WARNING: NaN values detected in embeddings")
                    loss = torch.tensor(1.0, device=model.device, requires_grad=True)
                else:
                    q_mean = query_out.embeddings.mean().item()
                    d_mean = doc_out.embeddings.mean().item()
                    q_norm = torch.norm(query_out.embeddings, dim=-1).mean().item()
                    d_norm = torch.norm(doc_out.embeddings, dim=-1).mean().item()
                    
                    loss = self.loss_func(query_out.embeddings, doc_out.embeddings)
                    
                    if loss < 0.001:
                        print(f"Query embed shape: {query_out.embeddings.shape}, mean: {q_mean}, norm: {q_norm}")
                        print(f"Doc embed shape: {doc_out.embeddings.shape}, mean: {d_mean}, norm: {d_norm}")
                
                if torch.isnan(loss) or torch.isinf(loss) or loss == 0.0:
                    print(f"WARNING: Abnormal loss value: {loss.item()}")
                    print(f"Query embed shape: {query_out.embeddings.shape}, mean: {query_out.embeddings.mean().item()}")
                    print(f"Doc embed shape: {doc_out.embeddings.shape}, mean: {doc_out.embeddings.mean().item()}")
     
                    if torch.isnan(loss) or torch.isinf(loss):
                        loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
            else:
                print("WARNING: Empty embeddings detected")
                loss = torch.tensor(1.0, device=model.device, requires_grad=True)
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        if not return_outputs:
            return loss
        else:
            return (loss, (query_out, doc_out))
    
    def prediction_step(self, model, inputs, **kwargs):
        query_inputs, doc_inputs = inputs
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION=="fp16"):
                query_outputs = model(**query_inputs)
                doc_outputs = model(**doc_inputs)
                loss = self.loss_func(query_outputs.embeddings, doc_outputs.embeddings)
            
            return loss, None, None


def train_model():
    print("Loading dataset...")
    dataset_splits = load_combined_dataset(DATASET_PATH)
    train_ds = dataset_splits["train"]
    test_ds = dataset_splits["test"]
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    
    print("Loading model...")
    model, processor = load_model()
    
    model.train()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params:.2%})")
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        dataloader_pin_memory=False,
        report_to="none",
        fp16=MIXED_PRECISION=="fp16",
        fp16_full_eval=MIXED_PRECISION=="fp16",
        dataloader_num_workers=0,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        learning_rate=5e-5,
        weight_decay=0.01,
    )
    
    loss_func = EnhancedColbertPairwiseCELoss(temperature=0.05, debug=True)
        
    trainer = MemoryEfficientTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=args,
        loss_func=loss_func,
        data_collator=lambda ex: collate_fn(ex, processor, model.device)
    )
    
    print("Starting training...")
    torch.cuda.empty_cache()
    gc.collect()
    
    trainer.train()
    
    return trainer, model


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    trainer, model = train_model()
    
    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)