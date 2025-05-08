MODEL_NAME = "vidore/colpali-v1.2-hf"
USE_Q4 = True
USE_Q8 = False
USE_LORA = True  # Enable LoRA for memory efficiency
DEVICE = "cuda:0"

# Memory optimization parameters
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 1
TOP_K = 5

# Dataset parameters
DATASET_PATH = "mrbesher/flickr-fi"
OUTPUT_DIR = "/kaggle/working/kuvahaku_fi_model"

MAX_LENGTH = 77  # Maximum length for text inputs
LAZY_LOADING = True  # Enable lazy loading of images
MIXED_PRECISION = "fp16"  # Use mixed precision training

MODEL_PATH = "outputs/kuvahaku_fi_model/checkpoint-659"