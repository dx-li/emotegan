
import os
from itertools import cycle
import pickle

import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
from typing import List, Union
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

import torch
from torch.utils import data
import numpy as np

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model.Generator import Generator
from model.Discriminator import SpatialDiscriminator, TemporalDiscriminator

class GIFDataset(Dataset):
    def __init__(self, gif_dir, target_size=(56, 56), num_frames=16):
        self.gif_dir = gif_dir
        self.gif_files = [f for f in os.listdir(gif_dir) if f.endswith('.gif')]
        self.target_size = target_size
        self.num_frames = num_frames

    def __len__(self):
        return len(self.gif_files)

    def __getitem__(self, idx):
        gif_path = os.path.join(self.gif_dir, self.gif_files[idx])
        
        with Image.open(gif_path) as img:
            frames = []
            for i in range(self.num_frames):
                img.seek(i % img.n_frames)
                frame = img.convert('RGB').resize(self.target_size)
                frames.append(np.array(frame))
            
            # Stack frames and convert to float32
            gif_array = np.stack(frames, axis=0).astype(np.float16)
            
            # Normalize to [0, 1]
            gif_array /= 255.0
            
            # Rearrange dimensions to channels x frames x height x width
            gif_array = np.transpose(gif_array, (3, 0, 1, 2))

        return gif_array

def numpy_collate(batch):
  return jax.tree_util.tree_map(np.asarray, data.default_collate(batch))

def get_gif_dataloader(gif_dir, batch_size=32, shuffle=True, num_workers=0, drop_last=True):
    dataset = GIFDataset(gif_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers, collate_fn=numpy_collate, drop_last=drop_last)

def generate_fake_batch(generator, generator_state, key, batch_size, latent_size):
    noise = jr.truncated_normal(key, -2, 2, (batch_size, latent_size), dtype=jnp.bfloat16)
    fake_batch, new_generator_state = jax.vmap(
        generator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(noise, generator_state)
    return fake_batch, new_generator_state

@eqx.filter_value_and_grad(has_aux=True)
def compute_grads_spatial_discriminator(
    spatial_discriminator,
    fake_batch,
    real_batch,
    spatial_discriminator_state,
):
    fake_pred, new_state = jax.vmap(
        spatial_discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(fake_batch, spatial_discriminator_state)
    fake_loss = fake_pred.mean()

    real_pred, new_state = jax.vmap(
        spatial_discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(real_batch, new_state)
    real_loss = -real_pred.mean()

    total_loss = fake_loss + real_loss
    return total_loss, new_state

@eqx.filter_value_and_grad(has_aux=True)
def compute_grads_temporal_discriminator(
    temporal_discriminator,
    fake_batch,
    real_batch,
    temporal_discriminator_state,
):
    fake_pred, new_state = jax.vmap(
        temporal_discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(fake_batch, temporal_discriminator_state)
    fake_loss = fake_pred.mean()

    real_pred, new_state = jax.vmap(
        temporal_discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(real_batch, new_state)
    real_loss = -real_pred.mean()

    total_loss = fake_loss + real_loss
    return total_loss, new_state

@eqx.filter_jit
def step_discriminators(
    spatial_discriminator: eqx.Module,
    temporal_discriminator: eqx.Module,
    generator: eqx.Module,
    real_batch: jnp.ndarray,
    spatial_optimizer: optax.GradientTransformation,
    temporal_optimizer: optax.GradientTransformation,
    spatial_opt_state: optax.OptState,
    temporal_opt_state: optax.OptState,
    generator_state: eqx.nn.State,
    spatial_discriminator_state: eqx.nn.State,
    temporal_discriminator_state: eqx.nn.State,
    key: jr.PRNGKey,
    batch_size: int,
    latent_size: int,
):
    key, subkey = jr.split(key)
    fake_batch, new_generator_state = generate_fake_batch(generator, generator_state, subkey, batch_size, latent_size)
    
    # Spatial discriminator update
    (spatial_loss, new_spatial_state), spatial_grads = compute_grads_spatial_discriminator(
        spatial_discriminator,
        fake_batch,
        real_batch,
        spatial_discriminator_state,
    )
    
    spatial_updates, new_spatial_opt_state = spatial_optimizer.update(
        spatial_grads, spatial_opt_state, spatial_discriminator
    )
    new_spatial_discriminator = eqx.apply_updates(spatial_discriminator, spatial_updates)
    
    # Temporal discriminator update
    (temporal_loss, new_temporal_state), temporal_grads = compute_grads_temporal_discriminator(
        temporal_discriminator,
        fake_batch,
        real_batch,
        temporal_discriminator_state,
    )
    
    temporal_updates, new_temporal_opt_state = temporal_optimizer.update(
        temporal_grads, temporal_opt_state, temporal_discriminator
    )
    new_temporal_discriminator = eqx.apply_updates(temporal_discriminator, temporal_updates)
    
    # Clip discriminator weights
    # new_spatial_discriminator = eqx.tree_at(lambda d: eqx.filter(d, eqx.is_array), new_spatial_discriminator, 
    #                                         lambda x: jnp.clip(x, -0.01, 0.01))
    # new_temporal_discriminator = eqx.tree_at(lambda d: eqx.filter(d, eqx.is_array), new_temporal_discriminator, 
    #                                          lambda x: jnp.clip(x, -0.01, 0.01))
    
    total_loss = spatial_loss + temporal_loss
    
    return (
        total_loss,
        new_spatial_discriminator,
        new_temporal_discriminator,
        new_spatial_opt_state,
        new_temporal_opt_state,
        new_generator_state,
        new_spatial_state,
        new_temporal_state,
        key,
    )

@eqx.filter_value_and_grad(has_aux=True)
def compute_grads_generator(
    generator,
    spatial_discriminator,
    temporal_discriminator,
    spatial_discriminator_state,
    temporal_discriminator_state,
    generator_state,
    key,
    batch_size,
    latent_size
):
    key, subkey = jr.split(key)
    noise = jr.truncated_normal(subkey, -2, 2, (batch_size, latent_size), dtype=jnp.bfloat16)
    fake_batch, new_generator_state = jax.vmap(
        generator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(noise, generator_state)
    
    # Spatial discriminator prediction
    spatial_pred, new_spatial_state = jax.vmap(
        spatial_discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(fake_batch, spatial_discriminator_state)
    
    # Temporal discriminator prediction
    temporal_pred, new_temporal_state = jax.vmap(
        temporal_discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(fake_batch, temporal_discriminator_state)
    
    # Combine losses from both discriminators
    total_loss = -(spatial_pred.mean() + temporal_pred.mean())
    
    return total_loss, (new_spatial_state, new_temporal_state, new_generator_state, key)

@eqx.filter_jit
def step_generator(
    generator: eqx.Module,
    spatial_discriminator: eqx.Module,
    temporal_discriminator: eqx.Module,
    generator_optimizer: optax.GradientTransformation,
    generator_opt_state: optax.OptState,
    spatial_discriminator_state: eqx.nn.State,
    temporal_discriminator_state: eqx.nn.State,
    generator_state: eqx.nn.State,
    key: jr.PRNGKey,
    batch_size: int,
    latent_size: int,
):
    (
        (loss, (new_spatial_state, new_temporal_state, new_generator_state, key)),
        grads,
    ) = compute_grads_generator(
        generator,
        spatial_discriminator,
        temporal_discriminator,
        spatial_discriminator_state,
        temporal_discriminator_state,
        generator_state,
        key,
        batch_size,
        latent_size
    )
    updates, new_generator_opt_state = generator_optimizer.update(grads, generator_opt_state, generator)
    new_generator = eqx.apply_updates(generator, updates)
    return loss, new_generator, new_generator_opt_state, new_spatial_state, new_temporal_state, new_generator_state, key



def to_bfloat16(x):
    if eqx.is_inexact_array(x):
        return x.astype(jax.dtypes.bfloat16)
    else:
        return x


def save_generated_gifs(generated_images, output_folder='generated_gifs'):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Determine the number of GIFs to generate
    num_gifs = generated_images.shape[0]
    
    for i in range(num_gifs):
        # Extract the frames for this GIF
        gif_frames = generated_images[i]  # Shape: (3, num_frames, height, width)
        
        # Convert from JAX array to numpy array
        gif_frames = np.array(gif_frames)
        
        # Transpose to (num_frames, height, width, 3)
        gif_frames = np.transpose(gif_frames, (1, 2, 3, 0))
        
        # Scale to 0-255 range and convert to uint8
        gif_frames = (gif_frames * 255).clip(0, 255).astype(np.uint8)
        
        # Convert numpy arrays to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in gif_frames]
        
        # Save as GIF
        pil_frames[0].save(
            f"{output_folder}/generated_gif_{i}.gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=100,  # Duration for each frame in milliseconds
            loop=0  # 0 means loop indefinitely
        )
    
    print(f"Generated {num_gifs} GIFs in the '{output_folder}' folder.")

def save_checkpoint(step, generator, spatial_discriminator, temporal_discriminator, 
                    generator_state, spatial_discriminator_state, temporal_discriminator_state,
                    generator_opt_state, spatial_discriminator_opt_state, temporal_discriminator_opt_state):
    checkpoint_dir = f'checkpoints/step_{step}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    eqx.tree_serialise_leaves(f"{checkpoint_dir}/generator.eqx", (generator, generator_state))
    eqx.tree_serialise_leaves(f"{checkpoint_dir}/spatial_discriminator.eqx", (spatial_discriminator, spatial_discriminator_state))
    eqx.tree_serialise_leaves(f"{checkpoint_dir}/temporal_discriminator.eqx", (temporal_discriminator, temporal_discriminator_state))
    
    print(f"Checkpoint saved at step {step}")

def generate_and_save_samples(step, generator, generator_state, key, latent_size, num_samples=20):
    key, subkey = jr.split(key)
    noise = jr.truncated_normal(subkey, -2, 2, (num_samples, latent_size), dtype=jnp.bfloat16)
    
    @eqx.filter_jit
    def generate_batch(model, noise):
        out, _ = jax.vmap(model, in_axes=(0, None), out_axes=(0, None))(noise, generator_state)
        return out
    
    inference_gen = eqx.nn.inference_mode(generator)
    generated_images = generate_batch(inference_gen, noise)
    output_folder = f'generated_samples/step_{step}'
    save_generated_gifs(generated_images, output_folder)
    
    return key

def train(
    generator: eqx.Module,
    spatial_discriminator: eqx.Module,
    temporal_discriminator: eqx.Module,
    generator_optimizer: optax.GradientTransformation,
    spatial_discriminator_optimizer: optax.GradientTransformation,
    temporal_discriminator_optimizer: optax.GradientTransformation,
    generator_opt_state: optax.OptState,
    spatial_discriminator_opt_state: optax.OptState,
    temporal_discriminator_opt_state: optax.OptState,
    generator_state: eqx.nn.State,
    spatial_discriminator_state: eqx.nn.State,
    temporal_discriminator_state: eqx.nn.State,
    data_loader: torch.utils.data.DataLoader,
    num_steps: int,
    key: jr.PRNGKey,
    batch_size: int = 32,
    latent_size: int = 240,
    print_every: int = 1,
    checkpoint_every: int = 100,
    n_critic: int = 5,
):
    generator_losses = []
    discriminator_losses = []

    def infinite_trainloader(data_loader):
        return cycle(data_loader)

    for step, batch in zip(range(num_steps), infinite_trainloader(data_loader)):
        images = batch.astype(jnp.bfloat16) 

        for _ in range(n_critic):
            # Step discriminators
            (
                discriminator_loss,
                spatial_discriminator,
                temporal_discriminator,
                spatial_discriminator_opt_state,
                temporal_discriminator_opt_state,
                generator_state,
                spatial_discriminator_state,
                temporal_discriminator_state,
                key,
            ) = step_discriminators(
                spatial_discriminator,
                temporal_discriminator,
                generator,
                images,
                spatial_discriminator_optimizer,
                temporal_discriminator_optimizer,
                spatial_discriminator_opt_state,
                temporal_discriminator_opt_state,
                generator_state,
                spatial_discriminator_state,
                temporal_discriminator_state,
                key,
                batch_size,
                latent_size,
            )

        # Step generator
        (
            generator_loss,
            generator,
            generator_opt_state,
            spatial_discriminator_state,
            temporal_discriminator_state,
            generator_state,
            key,
        ) = step_generator(
            generator,
            spatial_discriminator,
            temporal_discriminator,
            generator_optimizer,
            generator_opt_state,
            spatial_discriminator_state,
            temporal_discriminator_state,
            generator_state,
            key,
            batch_size,
            latent_size,
        )

        generator_losses.append(generator_loss)
        discriminator_losses.append(discriminator_loss)

        if (step % print_every) == 0 or step == num_steps - 1:
            print(
                f"Step: {step}/{num_steps}, Generator loss: {generator_loss}, "
                f"Discriminator loss: {discriminator_loss}"
            )

        if (step % checkpoint_every) == 0 or step == num_steps - 1:
            save_checkpoint(step, generator, spatial_discriminator, temporal_discriminator,
                            generator_state, spatial_discriminator_state, temporal_discriminator_state,
                            generator_opt_state, spatial_discriminator_opt_state, temporal_discriminator_opt_state)
            key = generate_and_save_samples(step, generator, generator_state, key, latent_size)

    return (
        generator,
        spatial_discriminator,
        temporal_discriminator,
        generator_state,
        spatial_discriminator_state,
        temporal_discriminator_state,
        generator_losses,
        discriminator_losses,
        key,
    )


if __name__ == "__main__":
    gif_dir = 'data_collection/processed_gifs'



    
    # Hyperparameters
    lr = 0.00005 
    beta1 = 0.0
    beta2 = 0.999  
    batch_size = 4
    latent_size = 128
    num_steps = 15000
    n_critic = 5  # Number of discriminator updates per generator update


    dataloader = get_gif_dataloader(gif_dir, batch_size=batch_size)
    # for batch in dataloader:
    #     print(f"Batch shape: {batch.shape}")
    #     print(f"Batch dtype: {batch.dtype}")
    #     print(f"Batch min: {batch.min()}, max: {batch.max()}")
    #     break
    key = jr.PRNGKey(2503)

    key, gen_key, sdisc_key, tdisc_key = jr.split(key, 4)

    generator, gen_state = eqx.nn.make_with_state(Generator)(gen_key, latent_size, 4, 32, 16)
    generator = jax.tree_util.tree_map(to_bfloat16, generator)
    gen_state = jax.tree_util.tree_map(to_bfloat16, gen_state)
    s_discriminator, sd_state = eqx.nn.make_with_state(SpatialDiscriminator)(sdisc_key)
    s_discriminator = jax.tree_util.tree_map(to_bfloat16, s_discriminator)
    sd_state = jax.tree_util.tree_map(to_bfloat16, sd_state)
    t_discriminator, td_state = eqx.nn.make_with_state(TemporalDiscriminator)(tdisc_key)
    t_discriminator = jax.tree_util.tree_map(to_bfloat16, t_discriminator)
    td_state = jax.tree_util.tree_map(to_bfloat16, td_state)

    def create_optimizer(base_lr, clip = False, clip_val = 5):
        if clip:
            return optax.chain(
                optax.clip_by_global_norm(clip),
                optax.adamw(base_lr, b1=beta1, b2=beta2),
            )
        return optax.adamw(base_lr, b1=beta1, b2=beta2)

    generator_optimizer = create_optimizer(lr)
    s_discriminator_optimizer = create_optimizer(2*lr, clip=True)
    t_discriminator_optimizer = create_optimizer(2*lr, clip=True)

    generator_opt_state = generator_optimizer.init(eqx.filter(generator, eqx.is_array))
    s_discriminator_opt_state = s_discriminator_optimizer.init(eqx.filter(s_discriminator, eqx.is_array))
    t_discriminator_opt_state = t_discriminator_optimizer.init(eqx.filter(t_discriminator, eqx.is_array))

    (
        generator,
        spatial_discriminator,
        temporal_discriminator,
        generator_state,
        spatial_discriminator_state,
        temporal_discriminator_state,
        generator_losses,
        discriminator_losses,
        key,
    ) = train(
        generator,
        s_discriminator,
        t_discriminator,
        generator_optimizer,
        s_discriminator_optimizer,
        t_discriminator_optimizer,
        generator_opt_state,
        s_discriminator_opt_state,
        t_discriminator_opt_state,
        gen_state,
        sd_state,
        td_state,
        dataloader,
        num_steps,
        key,
        batch_size,
        latent_size,
        checkpoint_every=100,
        n_critic=n_critic,  
    )

    eqx.tree_serialise_leaves("generator.eqx", (generator, generator_state))

    key, subkey = jr.split(key)
    noise = jr.truncated_normal(subkey, -2, 2, (1000, latent_size), dtype=jnp.bfloat16)


    @eqx.filter_jit
    def evaluate(model, xx):
        out, _ = jax.vmap(model)(xx)
        return out


    inference_gen = eqx.nn.inference_mode(generator)
    inference_gen = eqx.Partial(inference_gen, state=generator_state)

    generated_images = evaluate(inference_gen, noise)
    save_generated_gifs(generated_images, output_folder='generated_gifs')