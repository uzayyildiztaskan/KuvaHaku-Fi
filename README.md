
# KuvaHaku-Fi: Finnish Multimodal Image Retrieval with ColPali

**KuvaHaku-Fi** is a Finnish-language multimodal image retrieval system. Given a Finnish text description, the system retrieves the top-5 most semantically relevant images from a database. It leverages [ColPali](https://huggingface.co/vidore/colpali-v1.2-hf), a dual-encoder model that maps images and text into a shared embedding space for efficient similarity-based retrieval.

---

## 🔍 Project Overview

- **Architecture**: Dual encoder (ColPali) based on PaliGemma (image encoder) and Gemma (text encoder)
- **Language**: Finnish
- **Input**: Finnish text description
- **Output**: Top 5 most relevant images
- **Use Case**: Visual search, Finnish-language assistants, multilingual AI applications

---

## 🗃️ Dataset

### Source
- **Base**: [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) – 31,783 images, 5 captions per image
- **Captions**: Translations of the Flickr30k 'test' split into Finnish
- **Translation Model**: Google's Gemini 1.5 Flash via Google Generative Language API

### Hugging Face Dataset
- 📦 [mrbesher/flickr-fi](https://huggingface.co/datasets/mrbesher/flickr-fi)

### Fields
- `img_id`: Original Flickr30k image ID
- `original_caption`: List of one English caption
- `finnish_caption`: List of one translated Finnish caption

---

## 🧩 Dataset Augmentation

Since the original dataset did not include image data, this project uses the original Flickr30k image files to match `img_id` values and build a multimodal dataset.

### Required Image Folder
Place all Flickr30k images in a folder:

```
flickr30k_images/
├── 1000092795.jpg
├── 10002456.jpg
├── ...
```

### Dataset Construction Script
A Python script links Finnish captions to corresponding image paths and casts them as `datasets.Image` objects.

---

## 🛠️ Project Structure

```
kuvahaku_fi/
├── config.py               # Settings and flags
├── dataset.py             # Load and attach images
├── model.py               # Load ColPali with (Q)LoRA
├── collator.py            # Image-text batch prep
├── train.py               # Fine-tune ColPali
├── infer.py               # Run inference on new queries
├── requirements.txt       # Dependencies
└── README.md              # You are here
```

---

## 🚀 How to Run

1. **Prepare Image Folder**: Download Flickr30k images and place in `flickr30k_images/`
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train**:
   ```bash
   python train.py
   ```
4. **Inference**:
   ```bash
   python infer.py
   ```

---


## ✍️ License

The Flickr30k dataset is provided for academic use. Gemini translations are for non-commercial use only. This project is under MIT License unless otherwise stated.

---

## 👤 Authors

- 🧑‍💻 [Uzay YILDIZTASKAN](https://github.com/uzayyildiztaskan)
- 🤖 Gemini translations powered via Google Generative Language API
