from transformers import ColPaliForRetrieval, ColPaliProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from config import *

def load_model():
    if USE_Q8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif USE_Q4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quant_config = None

    model = ColPaliForRetrieval.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    ).eval()

    if USE_Q8 or USE_Q4 or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        )
        lora_config.inference_mode = False
        model = get_peft_model(model, lora_config)

    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    return model, processor
