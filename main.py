import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import pickle

import mnist
import diff as d
import models as m

def parse_args():
    parser = argparse.ArgumentParser(description="Classifier Latent Diffusion")
    parser.add_argument("--config", type=str)
    sub_parsers = parser.add_subparsers()
    
    make_dataset_parser = sub_parsers.add_parser("md")
    make_dataset_parser.add_argument("--output", type=str)
    make_dataset_parser.set_defaults(func=make_dataset)
    
    train_gen_parser = sub_parsers.add_parser("tg")
    train_gen_parser.add_argument("--input", type=str)
    train_gen_parser.add_argument("--output", type=str)
    train_gen_parser.set_defaults(func=train_gen)
    
    train_cls_parser = sub_parsers.add_parser("tc")
    train_cls_parser.add_argument("--input", type=str)
    train_cls_parser.add_argument("--output", type=str)
    train_cls_parser.set_defaults(func=train_cls)
    
    create_latent_parser = sub_parsers.add_parser("cl")
    create_latent_parser.add_argument("--input", type=str)
    create_latent_parser.add_argument("--output", type=str)
    create_latent_parser.add_argument("--model", type=str)
    create_latent_parser.set_defaults(func=create_latent)

    take_image_parser = sub_parsers.add_parser("ti")
    take_image_parser.add_argument("--input", type=str)
    take_image_parser.add_argument("--mode", type=str)
    take_image_parser.add_argument("--idx", type=str)
    take_image_parser.add_argument("--output", type=str)
    take_image_parser.set_defaults(func=take_image)
    
    create_advex_parser = sub_parsers.add_parser("ca")
    create_advex_parser.add_argument("--input", type=str)
    create_advex_parser.add_argument("--model", type=str)
    create_advex_parser.add_argument("--label", type=int)
    create_advex_parser.add_argument("--pmin", type=str)
    create_advex_parser.add_argument("--output", type=str)
    create_advex_parser.set_defaults(func=create_advex)

    resample_image_parser = sub_parsers.add_parser("ri") 
    resample_image_parser.add_argument("--input", type=str)
    resample_image_parser.add_argument("--output", type=str)
    resample_image_parser.add_argument("--model", type=str)
    resample_image_parser.set_defaults(func=resample_image)

    display_image_parser = sub_parsers.add_parser("di")
    display_image_parser.add_argument("--input", type=str)
    display_image_parser.set_defaults(func=display_image)

    args = parser.parse_args()
    return args

def parse_config(file_path):
    with open(file_path, 'r') as file:
        conf = json.load(file)
        return conf

def main():
    args = parse_args()
    conf = parse_config(args.config)

    if hasattr(args, 'func'):
        args.func(args, conf)
    else:
        parser.print_help()

def make_dataset(args, conf):
    mnist.format_mnist(args, conf)

def train_gen(args, conf):
    key = jax.random.PRNGKey(42)
    init_key, state_key, sample_key = jax.random.split(key,3)

    data_loader = mnist.MNISTDataLoader(args.input, conf["tg"]["batch_size"])
    
    model = m.MnistDiffusion(init_key, conf["tg"]["n_hidden"])

    optimizer = optax.adam(conf["tg"]["lr"])
    opt_state = optimizer.init(model)

    state = model, opt_state, state_key

    loss_fn = lambda x,y,z: d.diffusion_loss(x, y, d.f_neg_gamma, z)

    for i in range(conf["tg"]["n_epochs"]):
        for images, labels in data_loader:
            loss, state = d.update_state(state, images, optimizer, loss_fn)
            print(loss)

    trained_model = state[0]
    
    with open(args.output,"wb") as f:
        pickle.dump(trained_model, f)

def gen_samples(args, conf):
    n_samples = 5
    n_steps = 100
    shape = (28,28)
    noise, samples = d.sample_diffusion(trained_model, d.f_neg_gamma, sample_key, n_steps, shape, n_samples)
    
    for i in range(n_samples):
        plt.imshow(samples[i])
        plt.show()

def train_cls(args, conf):
    key = jax.random.PRNGKey(42)
    init_key, state_key, sample_key = jax.random.split(key,3)
    
    data_loader = mnist.MNISTDataLoader(args.input, conf["tc"]["batch_size"])
    
    model = m.MnistClassifier(init_key, conf["tc"]["n_hidden"])
    
    optimizer = optax.adam(conf["tc"]["lr"])
    opt_state = optimizer.init(model)

    state = model, opt_state, state_key

    def loss_fn(model, data, key):
        images, labels = data
        logits = jax.vmap(model)(images)
        losses = jax.vmap(lambda preds, idx: preds[idx])(logits, labels)
        return -jnp.mean(losses)
    k = 0
    for i in range(conf["tc"]["n_epochs"]):
        for data in data_loader:
            loss, state = d.update_state(state, data, optimizer, loss_fn)
            print(loss)
            k += 1
            if k > 5:
                break

    trained_model = state[0]
    
    with open(args.output,"wb") as f:
        pickle.dump(trained_model, f)

def create_latent(args, conf):
    with open(args.input,"rb") as f:
        data = pickle.load(f)
    
    bs = conf["cl"]["batch_size"]

    outputs = {}
    for mode in ["train","test"]:
        base_images = data[mode]["images"]
        latent_images = jnp.zeros_like(base_images)

        with open(args.model,"rb") as f:
            model = pickle.load(f)

        index = 0
        while index < len(base_images):
            print(index)
            print("bla")
            end_index = min(index + bs, len(base_images))
            image_batch = base_images[index:end_index]
            latent_batch = d.unsample_diffusion(
                    image_batch, model, d.f_neg_gamma, conf["cl"]["n_steps"])
            latent_images = latent_images.at[index:end_index].set(latent_batch)
            index = end_index

        outputs[mode] = {
                "images": np.array(latent_images),
                "labels": data[mode]["labels"]
            }

    with open(args.output,"wb") as f:
        pickle.dump(outputs,f)

def take_image(args, conf):
    with open(args.input,"rb") as f:
        images = pickle.load(f)
    image = images[args.mode]["images"][int(args.idx)]

    with open(args.output,"wb") as f:
        pickle.dump(image,f)

def create_advex(args, conf):
    with open(args.input,"rb") as f:
        image = jnp.array(pickle.load(f),dtype=jnp.float32)
    assert image.shape == (28,28)

    with open(args.model,"rb") as f:
        model = pickle.load(f)
    
    state_key = jax.random.PRNGKey(42)
    optimizer = optax.adam(conf["ca"]["lr"])
    opt_state = optimizer.init(image)
    state = image, opt_state, state_key
    
    def loss_fn(image, model, key):
        logits = model(image)
        log_p = logits[args.label]
        return -log_p

    while(True):
        loss, state = d.update_state(state, model, optimizer, loss_fn)
        log_p = -loss
        print(log_p)
        if (log_p > jnp.log(float(args.pmin))):
            break

    with open(args.output,"wb") as f:
        pickle.dump(np.array(image),f)

def resample_image(args, conf):
    with open(args.input,"rb") as f:
        image = pickle.load(f)
    with open(args.model,"rb") as f:
        model = pickle.load(f)
    resample = d.resample_diffusion(image[jnp.newaxis,:,:], model, d.f_neg_gamma, conf["ri"]["n_steps"])[0]
    with open(args.output,"wb") as f:
        pickle.dump(resample,f)

def display_image(args, conf):
    with open(args.input,"rb") as f:
        image = pickle.load(f)
    print(image.shape)
    assert image.shape == (28,28)
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()
