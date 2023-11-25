import pickle

import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

def format_mnist(args, conf):
    dataset = load_dataset("mnist")
    train_images = np.array(dataset["train"]["image"])/255
    test_images = np.array(dataset["test"]["image"])
    train_labels = dataset["train"]["labels"]
    test_labels = dataset["test"]["labels"]

    output = {
        "train":{
            "images":train_images,
            "labels":train_labels,
        },
        "test":{
            "images":test_images,
            "labels":test_labels,
        },
    }

    with open(args.output, "wb") as file:
        pickle.dump(output,file)
            
class MNISTDataLoader:
    def __init__(self, args, conf, mode="train"):
        with open(args.input, "rb") as file:
            self.dataset = pikle.load(file)[mode]
        self.batch_size = batch_size
        self.current_index = 0

    def __iter__(self):
        # Reset the current_index at the start of each new iteration
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, len(self.dataset))
        images = self.dataset["image"][self.current_index:end_index]
        labels = self.dataset["labels"][self.current_index:end_index]
        self.current_index = end_index
        return images, labels

