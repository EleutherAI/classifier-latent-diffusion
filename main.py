import jax
import optax
import matplotlib.pyplot as plt

import mnist
import diff as d
import models as m

N_HIDDEN = 32 
N_EPOCHS = 1
LR = 0.0003


def main():
    train()

def train():
    key = jax.random.PRNGKey(42)
    init_key, state_key, sample_key = jax.random.split(key,3)

    data_loader = mnist.MNISTDataLoader()
    
    model = m.MnistDiffusion(init_key, N_HIDDEN)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(model)

    state = model, opt_state, state_key

    loss_fn = lambda x,y,z: d.diffusion_loss(x, y, d.f_neg_gamma, z)

    for i in range(N_EPOCHS):
        for images, labels in data_loader:
            loss, state = d.update_state(state, images, optimizer, loss_fn)
            print(loss)

    trained_model = state[0]

    n_samples = 5
    n_steps = 100
    shape = (28,28)
    noise, samples = d.sample_diffusion(trained_model, d.f_neg_gamma, sample_key, n_steps, shape, n_samples)
    
    for i in range(n_samples):
        plt.imshow(samples[i])
        plt.show()

 #unsamples = unsample_diffusion(samples, trained_model, f_neg_gamma, sample_key, n_steps, shape, n_samples)

def main():
    train()

if __name__ == "__main__":
    main()
