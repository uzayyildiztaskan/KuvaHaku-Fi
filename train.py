from transformers import TrainingArguments
from colpali_engine.loss import ColbertPairwiseCELoss
from transformers import Trainer
from model import load_model
from dataset import load_combined_dataset
from collator import collate_fn
from config import *
from transformers import Trainer
from upload import upload_to_huggingface
import torch


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        query_inputs, doc_inputs = inputs
        query_out = model(**query_inputs)
        doc_out = model(**doc_inputs)
        loss = self.loss_func(query_out.embeddings, doc_out.embeddings)
        return (loss, (query_out, doc_out)) if return_outputs else loss
    
    def prediction_step(self, model, inputs):
        query_inputs, doc_inputs = inputs
        with torch.no_grad():
            query_outputs = model(**query_inputs)
            doc_outputs = model(**doc_inputs)
            loss = self.loss_func(query_outputs.embeddings, doc_outputs.embeddings)
            return loss, None, None

dataset_splits = load_combined_dataset(DATASET_PATH)
train_ds = dataset_splits["train"]
test_ds = dataset_splits["test"]

model, processor = load_model()

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    dataloader_pin_memory=True,
    report_to="none"
)

trainer = ContrastiveTrainer(
    model=model,
    train_dataset=train_ds,
    args=args,
    loss_func=ColbertPairwiseCELoss(),
    data_collator=lambda ex: collate_fn(ex, processor, model.device)
)


torch.cuda.empty_cache()

trainer.args.remove_unused_columns = False
trainer.train()

#upload_to_huggingface(trainer)

