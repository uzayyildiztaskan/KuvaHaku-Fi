from transformers import ColPaliForRetrieval, ColPaliProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from config import *
import gc

def load_model():
    torch.cuda.empty_cache()
    gc.collect()
    
    if USE_Q8:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0, 
            llm_int8_has_fp16_weight=True,
        )
    elif USE_Q4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quant_config = None
    
    model = ColPaliForRetrieval.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        offload_folder="offload",
        offload_state_dict=True,
    ).eval()
    
    if USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        lora_config.inference_mode = False
        model = get_peft_model(model, lora_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params} ({trainable_params/total_params:.2%} of total)")
    
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return model, processor