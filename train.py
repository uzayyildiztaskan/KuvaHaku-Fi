from transformers import TrainingArguments
from colpali_engine.loss import ColbertPairwiseCELoss
from transformers import Trainer
from model import load_model
from dataset import load_finnish_dataset
from collator import collate_fn
from config import *
from transformers import Trainer

class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func

    def compute_loss(self, model, inputs, return_outputs=False):
        query_inputs, doc_inputs = inputs
        query_out = model(**query_inputs)
        doc_out = model(**doc_inputs)
        loss = self.loss_func(query_out.embeddings, doc_out.embeddings)
        return (loss, (query_out, doc_out)) if return_outputs else loss

train_ds, _ = load_finnish_dataset(DATASET_PATH)
model, processor = load_model()

args = TrainingArguments(
    output_dir="./kuvahaku_fi_model",
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    report_to="none"
)

trainer = ContrastiveTrainer(
    model=model,
    train_dataset=train_ds,
    args=args,
    loss_func=ColbertPairwiseCELoss(),
    data_collator=lambda ex: collate_fn(ex, processor, model.device)
)

trainer.args.remove_unused_columns = False
trainer.train()

trainer.push_to_hub("your-hf-username/kuvahaku-fi")

