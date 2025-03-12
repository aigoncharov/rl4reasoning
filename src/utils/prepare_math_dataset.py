from datasets import load_dataset

def prepare_math_dataset():
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default")
    ds = ds["train"].train_test_split(test_size=0.2, shuffle=True)
    return ds["train"], ds["test"]



