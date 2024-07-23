import os

import torch
import gc

from environment_setup import PROJECT_ROOT_DIR
from mesh_on_vessels.datasets.vessel_dataset import create_dataset
from mesh_on_vessels.vq_autoenc import create_autoencoder_model, get_latest_checkpoint
from meshgpt_pytorch import MeshTransformer, MeshTransformerTrainer


def get_gpt_model(autoencoder, dataset):
    torch.cuda.empty_cache()
    gc.collect()
    max_seq = max(len(d["faces"]) for d in dataset if "faces" in d) * (
                autoencoder.num_vertices_per_face * autoencoder.num_quantizers)
    print("Max token sequence:", max_seq)

    # GPT2-Small model
    gpt_transformer = MeshTransformer(
        autoencoder,
        dim=768,
        coarse_pre_gateloop_depth=3,
        fine_pre_gateloop_depth=3,
        attn_depth=12,
        attn_heads=12,
        max_seq_len=max_seq,
        condition_on_text=False,
        gateloop_use_heinsen=False,
        dropout=0.0,
    )

    total_params = sum(p.numel() for p in gpt_transformer.decoder.parameters())
    total_params = f"{total_params / 1000000:.1f}M"
    print(f"Decoder total parameters: {total_params}")
    return gpt_transformer


def load_gpt_checkpoint(gpt_checkpoint, gpt_transformer):
    pkg = torch.load(gpt_checkpoint)
    gpt_transformer.load_state_dict(pkg['model'])


def train_gpt_model(autoenc_checkpoint, is_train, resume_gpt_training):
    save_dir = f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/gpt_outputs'
    os.makedirs(save_dir, exist_ok=True)
    autoencoder = create_autoencoder_model(autoenc_checkpoint)
    dataset = create_dataset()
    gpt_transformer = get_gpt_model(autoencoder, dataset)
    if is_train:
        batch_size = 8  # Max 64
        grad_accum_every = 4

        # Set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  4 * 16 = 64
        learning_rate = 1e-3  # Start training with the learning rate at 1e-2 then lower it to 1e-3 at stagnation or at 0.5 loss.

        gpt_trainer = MeshTransformerTrainer(model=gpt_transformer, warmup_steps=1000, num_train_steps=50000,  # 10000
                                         dataset=dataset,
                                         grad_accum_every=grad_accum_every,
                                         learning_rate=learning_rate,
                                         batch_size=batch_size,
                                         checkpoint_every=1000,
                                         checkpoint_folder=f'{save_dir}/checkpoints'
                                         # FP16 training, it doesn't speed up very much but can increase the batch size which will in turn speed up the training.
                                         # However it might cause nan after a while.
                                         # accelerator_kwargs = {"mixed_precision" : "fp16"}, optimizer_kwargs = { "eps": 1e-7}
                                         )
        if resume_gpt_training:
            print("resuming training")
            gpt_checkpoint = get_latest_checkpoint(
                directory_path=f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/gpt_outputs/checkpoints')
            gpt_trainer.load(gpt_checkpoint)
        gpt_trainer()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpt_checkpoint = get_latest_checkpoint(directory_path=f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/gpt_outputs/checkpoints')
        print(f"Loaded {gpt_checkpoint}")
        load_gpt_checkpoint(gpt_transformer=gpt_transformer, gpt_checkpoint=gpt_checkpoint)
        gpt_transformer.to(device)
    text_coords = []
    for _ in range(5):
        text_coords.append(gpt_transformer.generate(texts=None, temperature=0))

    from meshgpt_pytorch import mesh_render
    mesh_render.save_rendering(f'{save_dir}/3d_models_all.obj', text_coords)


if __name__ == '__main__':
    autoenc_checkpoint = get_latest_checkpoint(directory_path=f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/checkpoints_lower')
    print(f"Loaded checkpoint {autoenc_checkpoint}")
    train_gpt_model(autoenc_checkpoint=autoenc_checkpoint, is_train=False, resume_gpt_training=True)