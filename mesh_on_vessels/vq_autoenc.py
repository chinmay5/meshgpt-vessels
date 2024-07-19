import glob
import os
import random

import torch
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR
from mesh_on_vessels.datasets.vessel_dataset import create_dataset, create_check_dataset
from meshgpt_pytorch import MeshAutoencoder, MeshAutoencoderTrainer
import torch.nn.functional as F


def create_autoencoder_model(checkpoint=None):
    autoencoder = MeshAutoencoder(
        decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        # codebook_size = 2048, for the 250 face dataset, more face count probably requires 16k.
        # Default value is 16384
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
    if checkpoint is not None:
        pkg = torch.load(checkpoint)
        autoencoder.load_state_dict(pkg['model'])
        for param in autoencoder.parameters():
            param.requires_grad = True
    return autoencoder


def increase_dataset_size(dataset):
    dataset.data = [dict(d) for d in dataset.data] * 50
    print(len(dataset.data))


def train(from_scratch):
    dataset = create_dataset()
    # NOTE: During creating the dataset itself, we have increased the input by a factor of 50
    # Uncomment this line only if there is a very good reason
    # increase_dataset_size(dataset)
    batch_size = 16  # The batch size should be max 64.
    grad_accum_every = 4
    # So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
    learning_rate = 1e-3  # Start with 1e-3 then at stagnation around 0.35, you can lower it to 1e-4.

    if from_scratch:
        # First phase of training
        # We perform first phase of the training that uses a lr = 1e-3 and gives loss of 0.5 after 20-50 epochs.
        # https://github.com/lucidrains/meshgpt-pytorch/issues/93#issuecomment-2223353905
        phase_training(num_train_steps=20000, batch_size=batch_size, checkpoint=None,
                       dataset=dataset, grad_accum_every=grad_accum_every, learning_rate=learning_rate, save_dir_suffix="")
    # Now the second phase.
    # We need to give the model checkpoint as well.
    learning_rate_new = 1e-4
    checkpoint = get_latest_checkpoint(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/checkpoints/')
    print(f"Loading checkpoint {checkpoint}")
    phase_training(num_train_steps=10000, batch_size=batch_size, checkpoint=checkpoint, dataset=dataset,
                   grad_accum_every=grad_accum_every, learning_rate=learning_rate_new, save_dir_suffix="_lower")


def get_latest_checkpoint(directory_path: str) -> str:
    """
    Gets the last modified `pt` file from the directory
    :param directory_path: The directory to search pt files in
    :return: The path to the last modified checkpoint file
    """
    # Get the list of all files in the directory with their full paths
    pt_files = glob.glob(f"{directory_path}/*.pt")

    # Filter out only files (not directories) and get their modification times
    pt_files_with_mtime = [(file, os.path.getmtime(file)) for file in pt_files if os.path.isfile(file)]

    # Find the file with the most recent modification time
    if not pt_files_with_mtime:
        raise FileNotFoundError("No files found in the specified directory.")

    checkpoint = max(pt_files_with_mtime, key=lambda x: x[1])[0]

    return checkpoint


def phase_training(num_train_steps:int, batch_size:int, checkpoint:str, dataset, grad_accum_every:int, learning_rate:float, save_dir_suffix:str) -> None:
    """
    The function is responsible for executing the training loop.
    It is a utility function to help in multi-stage training process. It simplifies executing the model with different
    learning rate and an initial checkpoint.
    :param num_train_steps: Total number of training iterations
    :param batch_size: batch size
    :param checkpoint: None if we train from scratch else fine tune the model
    :param dataset: The dataset used for training
    :param grad_accum_every: accumulate gradients for effectively larger training size
    :param learning_rate: Learning rate for training
    :param save_dir_suffix: A string to indicate whether we are training from scratch or fine tuning. Gets added to the save_dir
    :return: None
    """
    autoencoder = create_autoencoder_model(checkpoint)
    autoencoder.commit_loss_weight = 0.1  # Set dependant on the datasets size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.
    autoencoder_trainer = MeshAutoencoderTrainer(model=autoencoder, warmup_steps=10, dataset=dataset,
                                                 num_train_steps=num_train_steps,
                                                 batch_size=batch_size,
                                                 grad_accum_every=grad_accum_every,
                                                 learning_rate=learning_rate,
                                                 checkpoint_every_epoch=100,
                                                 use_wandb_tracking=False,
                                                 checkpoint_folder=f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/checkpoints{save_dir_suffix}',
                                                 check_sample_every_n_steps=200)
    autoencoder_trainer()


@torch.inference_mode
def check_trained_model():
    # Load the model
    # We do not pass the checkpoint since model params are made trainable by default in that case
    autoenc_model = create_autoencoder_model(None)
    checkpoint = get_latest_checkpoint(directory_path=f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/checkpoints')
    print(f"Loading {checkpoint}")
    pkg = torch.load(checkpoint)
    autoenc_model.load_state_dict(pkg['model'])
    autoenc_model.eval()

    dataset = create_check_dataset()
    min_mse, max_mse = float('inf'), float('-inf')
    random_samples, random_samples_pred, all_random_samples = [], [], []
    total_mse, sample_size = 0.0, 20

    random.shuffle(dataset.data)
    for item in tqdm(dataset.data[:sample_size]):
        codes = autoenc_model.tokenize(vertices=item['vertices'], faces=item['faces'], face_edges=item['face_edges'])

        codes = codes.flatten().unsqueeze(0)
        codes = codes[:, :codes.shape[-1] // autoenc_model.num_quantizers * autoenc_model.num_quantizers]

        coords, mask = autoenc_model.decode_from_codes_to_faces(codes)
        orgs = item['vertices'][item['faces']].unsqueeze(0)

        mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu()) ** 2)
        total_mse += mse

        if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
        if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs

        random_samples.append(coords)
        random_samples_pred.append(orgs)

    print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')
    from meshgpt_pytorch import mesh_render
    # mesh_render.combind_mesh_with_rows(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/mse_rows.obj', all_random_samples)
    pred_gt_meshes = [random_samples_pred, random_samples]
    mesh_render.create_individual_obj_file_pairs(save_folder=f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/disc_recons_samples',
                                                 pred_gt_meshes=pred_gt_meshes)


@torch.inference_mode
def encode_latents():
    # Load the model
    # We do not pass the checkpoint since model params are made trainable by default in that case
    autoenc_model = create_autoencoder_model(None)
    checkpoint = get_latest_checkpoint(directory_path=f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/checkpoints_lower')
    print(f"Loading {checkpoint}")
    pkg = torch.load(checkpoint)
    autoenc_model.load_state_dict(pkg['model'])
    autoenc_model.eval()

    dataset = create_dataset()
    min_mse, max_mse, total_mse = float('inf'), float('-inf'), 0
    random_samples, random_samples_pred, all_random_samples = [], [], []

    # Obtain the vq-codes of the dataset
    sample_size = 10
    # Needed, else we will keep getting the same images as result
    random.shuffle(dataset.data)
    quantized_tensor_list, quantized_mask_list = [], []
    # NOTE: The number of faces is fixed at 800.
    num_faces = 800
    for idx, item in tqdm(enumerate(dataset.data)):
        vertices, faces, face_edges = item['vertices'].unsqueeze(0), item['faces'].unsqueeze(0), item['face_edges'].unsqueeze(0)
        quantized, face_mask = autoenc_model.get_quantized_encoding(vertices=vertices, faces=faces, face_edges=face_edges)
        if quantized.size(1) < num_faces:
            padding = (0, 0, 0, num_faces - quantized.size(1))
            quantized_mask = torch.ones((quantized.size(0), num_faces), dtype=torch.bool)
            # Indicate a mask to show that the remaining entries are padded
            quantized_mask[:, quantized.size(1): ] = False
            padded_quantized_tensor = F.pad(quantized, padding, "constant", 0)
        else:
            quantized_mask = torch.ones((quantized.size(1), num_faces), dtype=torch.bool)
            padded_quantized_tensor = quantized.detach().clone()

        quantized_tensor_list.append(padded_quantized_tensor.cpu())
        quantized_mask_list.append(quantized_mask.cpu())
        coords, mask = autoenc_model.decode_from_quantized(quantized=quantized, face_mask=face_mask)
        orgs = item['vertices'][item['faces']].unsqueeze(0)

        mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu()) ** 2)
        total_mse += mse

        if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
        if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs
        if idx <= sample_size:
            random_samples.append(coords)
            random_samples_pred.append(orgs)

    print(f'MSE AVG: {total_mse:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')
    from meshgpt_pytorch import mesh_render
    pred_gt_meshes = [random_samples_pred, random_samples]
    mesh_render.create_individual_obj_file_pairs(save_folder=f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/cont_recons_samples',
                                                 pred_gt_meshes=pred_gt_meshes)
    # Let us save the quantized tensor
    # Combine tensors in a dictionary
    data = {'data': torch.cat(quantized_tensor_list, dim=0),
            'mask': torch.cat(quantized_mask_list, dim=0)}
    # Save the dictionary
    torch.save(data, f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/cont_recons_tensor.pt')
    print("Quantized vectors saved")


if __name__ == '__main__':
    # train(from_scratch=True)
    check_trained_model()
    # encode_latents()