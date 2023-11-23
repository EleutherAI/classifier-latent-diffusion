import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import optax
import functools

#Data parameters
MEANS = (jnp.array([-0.5,0.5]),jnp.array([0.5,-0.5]))
VARIANCES = [jnp.eye(2)*0.1]*2
WEIGHTS = [0.5, 0.5]
PAIR_N = 2000


LR= 1e-3
B1 = 0.99
N_HIDDEN = 400 
N_ITER = int(1.5e5) 

K_BINS = 50 

def gaussian_mixture(key, means, covariances, weights, n_samples): 
    index_key, gaussain_key =  jax.random.split(key)
    
    samples = []
    for i in range(len(means)):
        gaussain_key,sub_key = jax.random.split(gaussain_key)
        m = means[i]
        c = covariances[i]
        samples.append(jax.random.multivariate_normal(sub_key, m, c, (n_samples,)))

    indicies = jax.random.randint(index_key, shape=(n_samples,), minval=0, maxval=len(means))
    output = jnp.zeros((n_samples,len(means[0])))
    for i in range(n_samples):
        output = output.at[i].set(samples[indicies[i]][i])
        
    return output, indicies

def make_data(means, variances, weights, pair_n, key):
    a_key, b_key, pair_key = jax.random.split(key,3)
    #plt.scatter(data[:,0],data[:,1], c=idx)
    #plt.show()
    pair_data, pair_idx = gaussian_mixture(pair_key, means, variances, weights, pair_n)

    data = pair_data
    idx = pair_idx
    #plt.hist(data[0][:,0])
    #plt.show()
    #plt.hist(data[1][:,0])
    #plt.show()
    return data, idx

#Linear SNR Schedule
def f_neg_gamma(t, min_snr= -10, max_snr = 10):
    #equivalent to log SNR
    return max_snr - t*(max_snr - min_snr)

def sigma_squared(neg_gamma):
    return jax.nn.sigmoid(-neg_gamma)

def alpha_squared(neg_gamma):
    return jax.nn.sigmoid(neg_gamma)

def diffusion_loss(model, data, f_neg_gamma,  key):
    #As defined in https://arxiv.org/abs/2107.00630 eq. #17 
    
    batch_size = data.shape[0]
    
    keys = jax.random.split(key, batch_size)

    def _diffusion_loss(model, f_neg_gamma, data, key):
        t_key, noise_key = jax.random.split(key,2)
        
        t = jax.random.uniform(t_key)
        
        neg_gamma, neg_gamma_prime = jax.value_and_grad(f_neg_gamma)(t)

        alpha, sigma = jnp.sqrt(alpha_squared(neg_gamma)), jnp.sqrt(sigma_squared(neg_gamma))

        epsilon = jax.random.normal(key, shape = data.shape)

        z = data*alpha + sigma*epsilon

        epsilon_hat = model(z, neg_gamma)

        loss = -1/2*neg_gamma_prime*(epsilon_hat-epsilon)**2

        return jnp.sum(loss)

    losses = jax.vmap(lambda x, y: _diffusion_loss(model, f_neg_gamma, x, y))(data, keys)
    mean_loss = jnp.sum(losses)/data.size

    return mean_loss

def sample_diffusion(model, f_neg_gamma, key, n_steps, shape, n_samples):
    #Following https://arxiv.org/abs/2202.00512 eq. #8
    time_steps = jnp.linspace(0, 1, num=n_steps+1)

    init_z = jax.random.normal(key, (n_samples,) + shape)
    z = init_z
    for i in range(n_steps):
        # t_s < t_t
        t_s, t_t = time_steps[n_steps-i-1], time_steps[n_steps-i]

        neg_gamma_s, neg_gamma_t = f_neg_gamma(t_s), f_neg_gamma(t_t)
        
        alpha_s = jnp.sqrt(alpha_squared(neg_gamma_s))
        alpha_t, sigma_t = jnp.sqrt(alpha_squared(neg_gamma_t)), jnp.sqrt(sigma_squared(neg_gamma_t))

        epsilon_hat = jax.vmap(lambda x: model(x, neg_gamma_t))(z)

        k = jnp.exp((neg_gamma_t-neg_gamma_s)/2)
        z = (alpha_s/alpha_t)*(z + sigma_t*epsilon_hat*(k-1))

    return init_z, z

def unsample_diffusion(z, model, f_neg_gamma, key, n_steps, shape, n_samples):
    
    time_steps = jnp.linspace(0, 1, num=n_steps+1)
    for i in list(range(n_steps))[::-1]:
        # t_s < t_t
        t_s, t_t = time_steps[n_steps-i-1], time_steps[n_steps-i]

        neg_gamma_s, neg_gamma_t = f_neg_gamma(t_s), f_neg_gamma(t_t)
        
        alpha_s = jnp.sqrt(alpha_squared(neg_gamma_s))
        alpha_t, sigma_t = jnp.sqrt(alpha_squared(neg_gamma_t)), jnp.sqrt(sigma_squared(neg_gamma_t))

        epsilon_hat = jax.vmap(lambda x: model(x, neg_gamma_t))(z)

        k = jnp.exp((neg_gamma_t-neg_gamma_s)/2)
        #z = (alpha_s/alpha_t)*(z + sigma_t*epsilon_hat*(k-1))
        
        z = z*(alpha_t/alpha_s) - sigma_t*epsilon_hat*(k-1)

    return z 

@functools.partial(jax.jit, static_argnums=(2, 3))
def update_state(state, data, optimizer, loss_fn):
    model, opt_state, key = state
    new_key, subkey = jax.random.split(key)
    
    loss, grads = jax.value_and_grad(loss_fn)(model, data, subkey)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    new_state = new_model, new_opt_state, new_key
    
    return loss,new_state

if __name__ == "__main__":
    main()
