import glob
import io
import os
import random
from collections import OrderedDict

import numpy as np
import open3d as o3d
import torch
from pymeshfix import MeshFix
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR
from mesh_on_vessels.datasets.dataset_utils import decimate_mesh
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


def load_filename(dataset_file_names, variations):
    obj_datas = []
    possible_values = np.arange(0.75, 1.0, 0.005)
    scale_factors = np.random.choice(possible_values, size=variations)
    with open(dataset_file_names, 'r') as file:
        file_list = file.read().splitlines()
    # Filter spurious relations away
    file_list = [x for x in file_list if x.endswith('.obj')]
    for file_path in file_list:
        vertices, faces = get_mesh(file_path)
        filename = os.path.basename(file_path)
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


def create_train_dataset(num_augmentations=5):
    dataset_path = f"{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/kidney_dataset.npz"
    train_file_names = f'{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/train.txt'
    if not os.path.isfile(dataset_path):
        data = load_filename(train_file_names, num_augmentations)
        dataset = MeshDataset(data)
        dataset.generate_face_edges()
        dataset.save(dataset_path)
    dataset = MeshDataset.load(dataset_path)
    return dataset


def create_val_dataset():
    dataset_path = f"{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/check_kidney_dataset.npz"
    val_file_names = f'{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/val.txt'
    if not os.path.isfile(dataset_path):
        data = load_filename(val_file_names, 1)
        dataset = MeshDataset(data)
        dataset.generate_face_edges()
        dataset.save(dataset_path)
    dataset = MeshDataset.load(dataset_path)
    return dataset


def ensure_water_tight(mesh, filename, buffer=None) -> o3d.geometry.TriangleMesh:
    """
    Converts the given mesh into a water-tight one.
    :param mesh: Original mesh
    :param filename: Filename of the original mesh. Done for logging purpose
    :return: Water tight mesh or the closest approximation
    """
    if not mesh.is_watertight():
        write_buffer(buffer, f"Original sample {filename} is not water tight\n")
        mesh = fix_mesh(mesh)
        if not mesh.is_watertight():
            write_buffer(buffer, f"ERROR: {filename} is not fixed. Please remove\n")
    return mesh


def _downsample_single_mesh(filename, max_faces=800, save_mesh=False, buffer=None):
    """
    The function downsamples a mesh to generate a relatively smaller size. It happens in two steps
    1. Vertex cluserting with twice the voxel size
    2. Decimation on the coarser mesh
    :param filename: Filename to downsample
    :param: The string buffer object
    :return: Downsampled mesh file
    """
    mesh_in = o3d.io.read_triangle_mesh(filename)
    if len(mesh_in.triangles) <= max_faces:
        return mesh_in
    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 32
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)

    # assert mesh_smp.is_watertight(), "Mesh is not water tight. Too large voxel size"
    base_path = os.path.split(filename)[0]
    filename = os.path.basename(filename).replace(".stl", "_check.obj")
    if len(mesh_smp.triangles) <= max_faces:
        return mesh_smp
    # Now we try to decimate the mesh
    decimated_mesh = decimate_mesh(filename='', max_faces=max_faces, visualize=False, mesh=mesh_smp)
    if save_mesh:
        final_filename = f"{base_path}/{filename.replace('_check', '_decimated')}"
        o3d.io.write_triangle_mesh(final_filename, decimated_mesh)
    return decimated_mesh


def _convert_pyvista_to_open3d(pyvista_mesh):
    """
    Converts the pyvista mesh into open3d. Needed for pymeshfix
    :param pyvista_mesh: The pyvista mesh
    :return: open3d mesh
    """

    vertices = pyvista_mesh.points
    faces = pyvista_mesh.faces.reshape((-1, 4))[:, 1:]  # Remove the first column which is the number of points per face

    # Create an Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()

    # Assign vertices and faces to the Open3D mesh
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh


def fix_mesh(decimated_mesh):
    """
    The function takes an open3d mesh and fills in the holes.
    It uses the pyMeshFix library
    :param decimated_mesh: open3d mesh file
    :return: o3d.geometry.TriangleMesh
    """
    vertices = np.asarray(decimated_mesh.vertices)
    faces = np.asarray(decimated_mesh.triangles)
    meshfix = MeshFix(vertices, faces)
    meshfix.repair()
    mesh = meshfix.mesh
    open3d_mesh = _convert_pyvista_to_open3d(mesh)
    return open3d_mesh


def write_buffer(buffer, message):
    if buffer is None:
        return
    buffer.write(message)


def down_sample_all_meshes(data_dir, max_faces, buffer=None):
    """
    Reads the meshes in the directory and downsamples them using
    :param data_dir:
    :param max_faces:
    :return:
    """
    target_dir = f"{os.path.split(data_dir)[0]}/subsampled_meshes"
    os.makedirs(target_dir, exist_ok=True)
    for filename in tqdm(glob.glob(f"{data_dir}/*.stl")):
        decimated_mesh = _downsample_single_mesh(filename=filename, max_faces=max_faces, buffer=buffer)
        decimated_mesh = ensure_water_tight(decimated_mesh, filename, buffer=buffer)
        # Let us save the mesh only when we can ensure it is water-tight
        if decimated_mesh.is_watertight():
            # We save only as obj
            filename = filename.replace('stl', "obj")
            o3d.io.write_triangle_mesh(f'{target_dir}/{os.path.basename(filename)}', decimated_mesh)
        else:
            print(f"{os.path.basename(filename)} is not water-tight. Skipping")


def split_train_test(base_folder, file_extension='.obj', train_ratio=0.8):
    # Get the list of all files with the specified extension in the folder
    all_files = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if f.endswith(file_extension)]

    # Shuffle the list to ensure random division
    random.shuffle(all_files)

    # Determine the split index
    split_index = int(len(all_files) * train_ratio)

    # Split the files into training and testing sets
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    # Write the training file paths to train.txt
    with open(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/train.txt', 'w') as train_file:
        for file_path in train_files:
            train_file.write(f"{file_path}\n")

    # Write the testing file paths to test.txt
    with open(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/val.txt', 'w') as val_file:
        for file_path in val_files:
            val_file.write(f"{file_path}\n")

    print(f"Train and test files have been written to train.txt and test.txt respectively.")


if __name__ == '__main__':
    buffer = io.StringIO()
    down_sample_all_meshes(data_dir='/mnt/dog/chinmay/temp_outputs/mesh_checks', max_faces=800,
                           buffer=buffer)
    # split_train_test('/mnt/dog/chinmay/temp_outputs/subsampled_meshes')
    # create_train_dataset(num_augmentations=5)
    final_content = buffer.getvalue()
    buffer.close()
    with open(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/mesh_extraction_log.txt', "w") as file:
        file.write(final_content)
