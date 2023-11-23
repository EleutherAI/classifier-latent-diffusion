import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

class MNISTDataLoader:
    def __init__(self, batch_size=64, mode="train"):
        self.dataset = load_dataset("mnist")[mode]
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
        batch = self.dataset[self.current_index:end_index]
        self.current_index = end_index
        images = jnp.stack([np.array(b) for b in batch['image']])/255
        labels = jnp.array([b for b in batch['label']])
        print(self.current_index)

        return images, labels

