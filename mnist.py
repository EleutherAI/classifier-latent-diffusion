import pickle

import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

def format_mnist(args, conf):
    dataset = load_dataset("mnist")
    train_images = np.array(dataset["train"]["image"])/255
    test_images = np.array(dataset["test"]["image"])
    train_labels = np.array(dataset["train"]["label"])
    test_labels = np.array(dataset["test"]["label"])

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
    def __init__(self, input_file, batch_size, mode="train"):
        with open(input_file, "rb") as file:
            self.dataset = pickle.load(file)[mode]
        self.batch_size = batch_size 
        self.current_index = 0

    def __iter__(self):
        # Reset the current_index at the start of each new iteration
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.dataset["images"]):
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, len(self.dataset["images"]))
        images = self.dataset["images"][self.current_index:end_index]
        labels = self.dataset["labels"][self.current_index:end_index]
        self.current_index = end_index
        return images, labels

