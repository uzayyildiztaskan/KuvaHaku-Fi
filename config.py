# Config flags
MODEL_NAME = "vidore/colpali-v1.2-hf"
USE_Q4 = True
USE_Q8 = False
USE_LORA = False
DEVICE = "cuda:0"

TRAIN_BATCH_SIZE = 2
EPOCHS = 1
TOP_K = 5

DATASET_PATH = "mrbesher/flickr-fi"
OUTPUT_DIR = "/kaggle/working/kuvahaku_fi_model"
