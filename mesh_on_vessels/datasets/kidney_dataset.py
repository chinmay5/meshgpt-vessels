import glob
import os
from collections import OrderedDict

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR
from mesh_on_vessels.datasets.dataset_utils import check_discretization, compare_subsampled_mesh_stats, \
    subsample_meshes, decimate_mesh
from meshgpt_pytorch.data import derive_face_edges_from_faces
from meshgpt_pytorch.mesh_dataset import MeshDataset


def get_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = np.asarray(mesh.vertices)
    if ".off" in file_path:  # ModelNet datasets
        mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(-90), 0])
        mesh.rotate(rotation_matrix)
        # Extract vertices and faces from the rotated mesh
        vertices = np.asarray(mesh.vertices)

    # mesh.show()

    faces = np.asarray(mesh.triangles)

    centered_vertices = vertices - np.mean(vertices, axis=0)
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)  # Limit vertices to [-0.95, 0.95]

    min_y = np.min(vertices[:, 1])
    difference = -0.95 - min_y
    vertices[:, 1] += difference

    def sort_vertices(vertex):
        return vertex[1], vertex[2], vertex[0]

    seen = OrderedDict()
    for point in vertices:
        key = tuple(point)
        if key not in seen:
            seen[key] = point

    unique_vertices = list(seen.values())
    sorted_vertices = sorted(unique_vertices, key=sort_vertices)

    vertices_as_tuples = [tuple(v) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v) for v in sorted_vertices]

    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples) for
                  new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples) if
                  vertex_tuple == sorted_vertex_tuple}
    reindexed_faces = [[vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]] for face in faces]
    sorted_faces = [sorted(sub_arr) for sub_arr in reindexed_faces]
    return np.array(sorted_vertices), np.array(sorted_faces)


def augment_mesh(vertices, scale_factor):
    jitter_factor = 0.01
    possible_values = np.arange(-jitter_factor, jitter_factor, 0.0005)
    offsets = np.random.choice(possible_values, size=vertices.shape)
    vertices = vertices + offsets

    vertices = vertices * scale_factor
    # To ensure that the mesh models are on the "ground"
    min_y = np.min(vertices[:, 1])
    difference = -0.95 - min_y
    vertices[:, 1] += difference
    return vertices


def load_filename(directory, variations):
    obj_datas = []
    possible_values = np.arange(0.75, 1.0, 0.005)
    scale_factors = np.random.choice(possible_values, size=variations)

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith((".obj", ".stl")):
            file_path = os.path.join(directory, filename)
            vertices, faces = get_mesh(file_path)

            faces = torch.tensor(faces.tolist(), dtype=torch.long).to("cuda")
            face_edges = derive_face_edges_from_faces(faces)
            texts, ext = os.path.splitext(filename)

            for scale_factor in scale_factors:
                aug_vertices = augment_mesh(vertices.copy(), scale_factor)
                obj_data = {"vertices": torch.tensor(aug_vertices.tolist(), dtype=torch.float).to("cuda"),
                            "faces": faces, "face_edges": face_edges, "texts": texts}
                obj_datas.append(obj_data)

    print(f"[create_mesh_dataset] Returning {len(obj_data)} meshes")
    return obj_datas


def create_dataset(data_dir, num_augmentations=5):
    dataset_path = f"{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/kidney_dataset.npz"
    if not os.path.isfile(dataset_path):
        data = load_filename(data_dir, num_augmentations)
        dataset = MeshDataset(data)
        dataset.generate_face_edges()
        dataset.save(dataset_path)
    dataset = MeshDataset.load(dataset_path)
    return dataset


def create_check_dataset(data_dir):
    dataset_path = f"{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/check_kidney_dataset.npz"
    if not os.path.isfile(dataset_path):
        data = load_filename(data_dir, 1)
        dataset = MeshDataset(data)
        dataset.generate_face_edges()
        dataset.save(dataset_path)
    dataset = MeshDataset.load(dataset_path)
    return dataset

def _downsample_single_mesh(filename, max_faces=800, save_mesh=False):
    """
    The function downsamples a mesh to generate a relatively smaller size. It happens in two steps
    1. Vertex cluserting with twice the voxel size
    2. Decimation on the coarser mesh
    :param filename: Filename to downsample
    :return: Downsampled mesh file
    """
    mesh_in = o3d.io.read_triangle_mesh(filename)
    # assert mesh_in.is_watertight(), "Original mesh is not water tight"
    mesh_in.compute_vertex_normals()
    if len(mesh_in.triangles) <= max_faces:
        return mesh_in
    # print(
    #     f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles'
    # )
    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 32
    # print(f'voxel_size = {voxel_size:e}')
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    # print(
    #     f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    # )
    # assert mesh_smp.is_watertight(), "Mesh is not water tight. Too large voxel size"
    base_path = os.path.split(filename)[0]
    filename = os.path.basename(filename).replace(".stl", "_check.obj")
    new_filename = f"{base_path}/{filename}"
    o3d.io.write_triangle_mesh(new_filename, mesh_smp)
    if len(mesh_smp.triangles) <= max_faces:
        return mesh_smp
    # Now we try to decimate the mesh
    decimated_mesh = decimate_mesh(new_filename, max_faces=max_faces, visualize=False)
    if save_mesh:
        final_filename = f"{base_path}/{filename.replace('_check', '_decimated')}"
        o3d.io.write_triangle_mesh(final_filename, decimated_mesh)
    return decimated_mesh
    # assert decimated_mesh.is_watertight(), "Mesh is not water tight. Too large decimation"



def down_sample_all_meshes(data_dir, max_faces):
    target_dir = f"{os.path.split(data_dir)[0]}/subsampled_meshes"
    os.makedirs(target_dir, exist_ok=True)
    for filename in tqdm(glob.glob(f"{data_dir}/*.stl")):
        decimated_mesh = _downsample_single_mesh(filename=filename, max_faces=max_faces)
        # Save this mesh
        # We save only as obj
        filename = filename.replace('stl', "obj")
        o3d.io.write_triangle_mesh(f'{target_dir}/{os.path.basename(filename)}', decimated_mesh)


if __name__ == '__main__':
    # subsample_meshes(data_dir='/mnt/dog/chinmay/temp_outputs/mesh_checks', max_faces=800, extension='.stl')
    # check_discretization("/mnt/dog/chinmay/IntrA/annotated/obj/AN1_full.obj")
    # compare_subsampled_mesh_stats(base_dir="/mnt/dog/chinmay/IntrA/annotated/obj",
    #                               subsampled_dir="/mnt/dog/chinmay/IntrA/annotated/subsampled")
    down_sample_all_meshes(data_dir='/mnt/dog/chinmay/temp_outputs/mesh_checks', max_faces=800)

