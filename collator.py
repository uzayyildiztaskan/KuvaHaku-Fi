def collate_fn(examples, processor, device):
    texts = [ex["finnish_caption"][0] for ex in examples]
    images = [ex["image"].convert("RGB") for ex in examples]
    batch_texts = processor(text=texts, return_tensors="pt")
    batch_images = processor(images=images, return_tensors="pt")
    return (batch_texts, batch_images)