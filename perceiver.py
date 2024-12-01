from typing import Sequence
import jax
import equinox
import einops

import jax_utils
import transformer

def make_fourier_features(max_seq_len, embedding_size):
    min_freq = 1.0
    max_resolution = max_seq_len
    num_bands = embedding_size // 2
    freq_bands = jax.numpy.linspace(min_freq, max_resolution / 2, num=num_bands)
    pos = jax.numpy.linspace(0, max_resolution - 1, num=max_resolution)
    pos = pos[:, None]
    per_pos_features = pos * freq_bands
    sine_features = jax.numpy.sin(jax.numpy.pi * per_pos_features)
    cosine_features = jax.numpy.cos(jax.numpy.pi * per_pos_features)
    fourier_features = jax.numpy.concatenate([sine_features, cosine_features], axis=-1)
    return fourier_features

class Token(equinox.Module):
    data: jax.Array
    timestep: jax.Array
    padding: jax.Array

class Perceiver(equinox.Module):
    max_timesteps: int = equinox.field(static=True)
    fourier_features: jax.Array = equinox.field(static=True)

    latent: jax.Array = equinox.field(static=False)
    encoders: Sequence[transformer.Layer] = equinox.field(static=False)
    output_projector: equinox.nn.Linear = equinox.field(static=False)
    def __init__(self, input_size, output_size, embedding_size, latent_count, num_layers, max_timesteps, rng):
        self.max_timesteps = max_timesteps
        self.fourier_features = make_fourier_features(max_seq_len=1000, embedding_size=embedding_size)
        rng, key = jax.random.split(rng)
        self.latent = jax.random.normal(key, shape=(latent_count, embedding_size + input_size))
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, num_layers)
        self.encoders = [transformer.Layer(embedding_size + input_size, rng=key) for key in keys]
        rng, key = jax.random.split(rng)
        self.output_projector = equinox.nn.Linear((embedding_size + input_size) * latent_count, output_size, key=key)
    def __call__(self, x: Sequence[Token]):
        len_x = len(x)
        x = jax_utils.tree_stack(x)
        context_datas = einops.pack([self.fourier_features[:len_x], x.data], "t *")[0]
        latent_datas = einops.rearrange(jax.numpy.repeat(einops.rearrange(self.latent, "s d -> 1 s d"), jax.numpy.array([self.max_timesteps]), axis=0, total_repeat_length=self.max_timesteps), "t s d -> (t s) d")
        mask = transformer.generate_io_mask(x.timestep, einops.repeat(jax.numpy.arange(self.max_timesteps), "d -> (d t)", t=self.latent.shape[0]))        
        for encoder in self.encoders:
            latent_datas = encoder(einops.pack([context_datas, latent_datas], "* d")[0], mask)[len_x:]
        return jax.vmap(self.output_projector)(einops.rearrange(latent_datas, "(t s) d -> t (s d)", s=self.latent.shape[0]))
    
import unittest

class TestPerceiver(unittest.TestCase):
    # TODO: This test needs to include multiple tokens per timesteps. this would more accurately reflect what we are trying to learn
    def test_perceiver_overfit(self):
        import optax
        embedding_size = 128
        input_size = 64
        output_size = input_size
        latent_count = 2
        num_layers = 2
        rng = jax.random.PRNGKey(0)
        num_batches = 2
        num_timesteps = 2
        perceiver = Perceiver(input_size, output_size, embedding_size, latent_count, num_layers, num_timesteps, rng)

        rng, key = jax.random.split(rng)
        input_data = jax.random.normal(key, (num_timesteps, input_size))
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, num_batches)
        input_data_batches = [input_data + (jax.random.normal(key, input_data.shape) * 1e-1) for key in keys]

        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, num_batches)
        output_data_batches = [jax.random.normal(key, (num_timesteps, input_size)) for key in keys]

        def loss_fn(perceiver, input_data, target):
            tokens = [Token(data=input_data[i], timestep=jax.numpy.array([i]), padding=jax.numpy.array([0])) for i in range(input_data.shape[0])]
            output = equinox.filter_jit(perceiver)(tokens)
            return jax.numpy.mean((output - target) ** 2)

        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(equinox.filter(perceiver, equinox.is_array_like))

        from tqdm import tqdm

        num_epochs = 1000
        for _ in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            epoch_losses = []
            for input_data, output_data in zip(input_data_batches, output_data_batches):
                loss, grads = equinox.filter_value_and_grad(loss_fn)(perceiver, input_data, output_data)
                updates, opt_state = optimizer.update(grads, opt_state, equinox.filter(perceiver, equinox.is_array_like))
                perceiver = equinox.apply_updates(perceiver, updates)
                epoch_losses.append(loss)
            avg_loss = jax.numpy.mean(jax.numpy.array(epoch_losses))
            tqdm.write(f"Epoch {_+1}, Average Loss: {avg_loss:.6f}")

        for input_data, output_data in zip(input_data_batches, output_data_batches):
            tokens = [Token(data=input_data[i], timestep=jax.numpy.array([i]), padding=jax.numpy.array([0])) for i in range(input_data.shape[0])]
            output = perceiver(tokens)
            loss = jax.numpy.mean((output - output_data) ** 2)
            if not (loss < 1e-3):
                raise AssertionError(f"Model did not overfit as expected. Loss: {loss}")


if __name__ == '__main__':
    unittest.main()

