from datasets import Dataset, load_dataset, Image

def load_combined_dataset(DATASET_PATH):
    fi_ds = load_dataset(DATASET_PATH)["train"]
    flickr_ds = load_dataset("nlphuji/flickr30k", split="test")

    flickr_ds = flickr_ds.cast_column("image", Image())  # Safe and lazy
    flickr_index = {ex["img_id"]: i for i, ex in enumerate(flickr_ds)}

    def attach_image(example):
        idx = flickr_index.get(example["img_id"])
        example["image"] = flickr_ds[idx]["image"] if idx is not None else None
        return example

    fi_ds = fi_ds.map(attach_image, batched=False)
    fi_ds = fi_ds.filter(lambda x: x["image"] is not None and x["finnish_caption"] is not None)
    return fi_ds.train_test_split(test_size=0.1)
