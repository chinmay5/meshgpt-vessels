import torch
from tqdm import tqdm
import trimesh
import numpy as np
import os
import csv
import json
from collections import OrderedDict

from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer
)
from meshgpt_pytorch.data import (
    derive_face_edges_from_faces
)


def create_model():
    autoencoder = MeshAutoencoder(
        # decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        decoder_dims_through_depth=(128,) * 3 + (192,) * 4 + (256,) * 23 + (384,) * 3,
        codebook_size=256,
        # Smaller vocab size will speed up the transformer training, however if you are training on meshes more then 250 triangle, I'd advice to use 16384 codebook size
        dim_codebook=192,
        dim_area_embed=16,
        dim_coor_embed=16,
        dim_normal_embed=16,
        dim_angle_embed=8,
        attn_decoder_depth=4,
        attn_encoder_depth=2
    ).to("cuda")
    total_params = sum(p.numel() for p in autoencoder.parameters())
    total_params = f"{total_params / 1000000:.1f}M"
    print(f"Total parameters: {total_params}")
    return autoencoder


def increase_dataset_size(dataset):
    dataset.data = [dict(d) for d in dataset.data] * 10
    print(len(dataset.data))


def continue_training(project_name, autoencoder):
    pkg = torch.load(str(f'mesh-encoder_{project_name}.pt'))
    autoencoder.load_state_dict(pkg['model'])
    for param in autoencoder.parameters():
        param.requires_grad = True


def train(data_dir):
    project_name = "demo_mesh"
    autoencoder = create_model()
    dataset = create_dataset(data_dir, project_name)
    increase_dataset_size(dataset)
    batch_size = 2  # The batch size should be max 64.
    grad_accum_every = 1
    # So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
    learning_rate = 1e-3  # Start with 1e-3 then at stagnation around 0.35, you can lower it to 1e-4.

    autoencoder.commit_loss_weight = 0.1  # Set dependant on the datasets size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.
    autoencoder_trainer = MeshAutoencoderTrainer(model=autoencoder, warmup_steps=10, dataset=dataset,
                                                 num_train_steps=100,
                                                 batch_size=batch_size,
                                                 grad_accum_every=grad_accum_every,
                                                 learning_rate=learning_rate,
                                                 checkpoint_every_epoch=50,
                                                 check_sample_every_n_steps=2)
    loss = autoencoder_trainer.train(100, stop_at_loss=0.2, diplay_graph=True)
    autoencoder_trainer.save(f'\mesh-encoder_{project_name}.pt')



if __name__ == "__main__":
    mesh_file = "/mnt/dog/chinmay/IntrA/annotated/obj/AN1_full.obj"
    mesh_dir = "/mnt/dog/chinmay/IntrA/annotated/obj"
    train(mesh_dir)
    # obj_datas = load_filename(directory=mesh_dir, variations=1)
    # print(obj_datas[0]["vertices"].shape)
    # print(obj_datas[0]["faces"].shape)