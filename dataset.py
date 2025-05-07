from datasets import load_dataset

def load_finnish_dataset(path):
    ds = load_dataset(path)
    ds = ds["train"].train_test_split(test_size=0.1)
    train_ds = ds["train"].filter(lambda x: x["finnish_caption"] is not None)
    test_ds = ds["test"].filter(lambda x: x["finnish_caption"] is not None)
    return train_ds, test_ds
