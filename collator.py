def collate_fn(examples, processor, device):
    texts = [ex["finnish_caption"][0] for ex in examples]
    images = [ex["image"].convert("RGB") for ex in examples]
    return (
        processor(text=texts, return_tensors="pt").to(device),
        processor(images=images, return_tensors="pt").to(device),
    )
