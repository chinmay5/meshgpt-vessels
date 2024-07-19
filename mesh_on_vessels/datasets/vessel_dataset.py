import glob
import os
from collections import OrderedDict

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR
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
        if filename.endswith((".obj", ".glb", ".off")):
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


def create_dataset():
    data_dir = "/mnt/dog/chinmay/IntrA/annotated/subsampled"
    dataset_path = f"{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/dataset.npz"
    if not os.path.isfile(dataset_path):
        data = load_filename(data_dir, 50)
        dataset = MeshDataset(data)
        dataset.generate_face_edges()
        dataset.save(dataset_path)
    dataset = MeshDataset.load(dataset_path)
    return dataset


def create_check_dataset():
    data_dir = "/mnt/dog/chinmay/IntrA/generated/vessel/subsampled"
    dataset_path = f"{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/check_dataset.npz"
    if not os.path.isfile(dataset_path):
        data = load_filename(data_dir, 1)
        dataset = MeshDataset(data)
        dataset.generate_face_edges()
        dataset.save(dataset_path)
    dataset = MeshDataset.load(dataset_path)
    return dataset


def decimate_mesh(filename: str, max_faces: int, visualize: bool = True) -> o3d.geometry.TriangleMesh:
    original_mesh = o3d.io.read_triangle_mesh(filename)
    simplified_mesh = original_mesh.simplify_quadric_decimation(
        target_number_of_triangles=max_faces)  # triangles are the faces
    if visualize:
        # Let us compute the surface normals for better visualization and easier Phong shading.
        original_mesh.compute_vertex_normals()
        simplified_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([original_mesh])
        o3d.visualization.draw_geometries([simplified_mesh])
    print(f"{np.asarray(original_mesh.triangles).shape=} \n {np.asarray(simplified_mesh.triangles).shape=}")
    return simplified_mesh


def subsample_meshes(data_dir: str, max_faces: int) -> None:
    """
    The function subsamples the meshes. The idea is to reduce the number of faces and vertices in the mesh
    without much loss to the quality. This makes learning on these meshes more compute friendly.
    :param data_dir: The directory to use
    :param max_faces: The maximum number of faces in the resulting mesh
    :return: None
    """
    target_dir = f"{os.path.split(data_dir)[0]}/subsampled"
    os.makedirs(target_dir, exist_ok=True)
    for filename in tqdm(glob.glob(f"{data_dir}/*.obj")):
        decimated_mesh = decimate_mesh(filename=filename, max_faces=max_faces, visualize=False)
        # Save this mesh
        o3d.io.write_triangle_mesh(f'{target_dir}/{os.path.basename(filename)}', decimated_mesh)


def check_discretization(filename):
    """
    The function visualizes the result of discretization operation of the mesh.
    The idea is, if these discrete tokens are too close to each other, that does not help us much
    :param filename:
    :return:
    """
    save_folder = f"{PROJECT_ROOT_DIR}/mesh_on_vessels/datasets/debug_dataset"
    os.makedirs(save_folder, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices, faces = np.asarray(mesh.vertices), torch.tensor(np.asarray(mesh.triangles), dtype=torch.float)
    # Same preprocessing as done by the actual dataloader.
    centered_vertices = vertices - np.mean(vertices, axis=0)
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)  # Limit vertices to [-0.95, 0.95]
    # Converting to torch.Tensor for corresponding API
    vertices = torch.tensor(vertices, dtype=torch.float)
    from meshgpt_pytorch.meshgpt_pytorch import discretize
    discrete_vertices = discretize(vertices, continuous_range=(-1, 1))
    mesh.vertices = o3d.utility.Vector3dVector(discrete_vertices.cpu().numpy())
    o3d.io.write_triangle_mesh(f'{save_folder}/discrete_mesh.obj', mesh)
    print("File saved")

def compare_subsampled_mesh_stats(base_dir:str, subsampled_dir:str) -> None:
    """
    The function will take a look at the original mesh and its subsampled version.
    We will just print the total number of faces and vertices.
    The API assumes that the filename is the same in both the folders.
    :param base_dir: Directory containing high res meshes
    :param subsampled_dir: Directory containing subsampled meshes
    :return: None
    """
    for filename in glob.glob(f"{base_dir}/*.obj"):
        original_mesh = o3d.io.read_triangle_mesh(filename)
        subsampled_mesh = o3d.io.read_triangle_mesh(f"{subsampled_dir}/{os.path.basename(filename)}")
        print(f"Original mesh has vertices = {np.asarray(original_mesh.vertices).shape[0]}, faces = {np.asarray(original_mesh.triangles).shape[0]}; Subsampled has vertices = {np.asarray(subsampled_mesh.vertices).shape[0]}, faces = {np.asarray(subsampled_mesh.triangles).shape[0]}")


# NOTE: These are some of the utility code that were used at various points during the code development.
# Leaving it here for possible reuse
    # meshfiles = glob.glob(f"{data_dir}/*.obj")
    # for file in meshfiles:
    #     vertices, faces = get_mesh(file)
    #     print(f"{vertices.shape=} and {faces.shape=}")
    # decimate_mesh("/mnt/dog/chinmay/IntrA/annotated/obj/AN1_full.obj")
    # vertices, faces = get_mesh("/mnt/dog/chinmay/IntrA/annotated/obj/AN1_full.obj")
    # print(f"{vertices.shape=} and {faces.shape=}")
    # create_dataset()
    # data_dir = "/mnt/dog/chinmay/IntrA/annotated/obj"
    # subsample_meshes(data_dir=data_dir, max_faces=800)
    # dataset = create_dataset()
    # print(dataset[0])
    # data_dir = "/mnt/dog/chinmay/IntrA/generated/vessel/vessel_subset"
    # subsample_meshes(data_dir=data_dir, max_faces=800)

if __name__ == '__main__':
    check_discretization("/mnt/dog/chinmay/IntrA/annotated/obj/AN1_full.obj")
    compare_subsampled_mesh_stats(base_dir="/mnt/dog/chinmay/IntrA/annotated/obj",
                                  subsampled_dir="/mnt/dog/chinmay/IntrA/annotated/subsampled")

