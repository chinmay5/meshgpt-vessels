import glob
import os

import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d
from pymeshfix import MeshFix

from environment_setup import PROJECT_ROOT_DIR


def decimate_mesh(filename: str, max_faces: int, visualize: bool = True, mesh=None) -> o3d.geometry.TriangleMesh:
    if mesh is None:
        original_mesh = o3d.io.read_triangle_mesh(filename)
    else:
        original_mesh = mesh
    simplified_mesh = original_mesh.simplify_quadric_decimation(
        target_number_of_triangles=max_faces)  # triangles are the faces
    if visualize:
        # Let us compute the surface normals for better visualization and easier Phong shading.
        original_mesh.compute_vertex_normals()
        simplified_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([original_mesh])
        o3d.visualization.draw_geometries([simplified_mesh])
    # print(f"{np.asarray(original_mesh.triangles).shape=} \n {np.asarray(simplified_mesh.triangles).shape=}")
    return simplified_mesh


def sanitize_extension(extension):
    return extension.replace(".", "")


def subsample_meshes(data_dir: str, max_faces: int, extension: str) -> None:
    """
    The function subsamples the meshes. The idea is to reduce the number of faces and vertices in the mesh
    without much loss to the quality. This makes learning on these meshes more compute friendly.
    :param data_dir: The directory to use
    :param max_faces: The maximum number of faces in the resulting mesh
    :param extension: obj, stl etc
    :return: None
    """
    target_dir = f"{os.path.split(data_dir)[0]}/subsampled"
    os.makedirs(target_dir, exist_ok=True)
    extension = sanitize_extension(extension)
    for filename in tqdm(glob.glob(f"{data_dir}/*.{extension}")):
        decimated_mesh = decimate_mesh(filename=filename, max_faces=max_faces, visualize=False)
        # Save this mesh
        # We save only as obj
        filename = filename.replace(extension, "obj")
        o3d.io.write_triangle_mesh(f'{target_dir}/{os.path.basename(filename)}', decimated_mesh)


def check_discretization(filename, extension):
    """
    The function visualizes the result of discretization operation of the mesh.
    The idea is, if these discrete tokens are too close to each other, that does not help us much
    :param filename: Sample file whose decimation needs to be verified
    :param extension: obj, stl etc
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
    o3d.io.write_triangle_mesh(f'{save_folder}/discrete_mesh.{extension}', mesh)
    print("File saved")

def compare_subsampled_mesh_stats(base_dir:str, subsampled_dir:str, extension: str) -> None:
    """
    The function will take a look at the original mesh and its subsampled version.
    We will just print the total number of faces and vertices.
    The API assumes that the filename is the same in both the folders.
    :param base_dir: Directory containing high res meshes
    :param subsampled_dir: Directory containing subsampled meshes
    :param extension: obj, stl etc
    :return: None
    """
    for filename in glob.glob(f"{base_dir}/*.{extension}"):
        original_mesh = o3d.io.read_triangle_mesh(filename)
        subsampled_mesh = o3d.io.read_triangle_mesh(f"{subsampled_dir}/{os.path.basename(filename)}")
        print(f"Original mesh has vertices = {np.asarray(original_mesh.vertices).shape[0]}, faces = {np.asarray(original_mesh.triangles).shape[0]}; Subsampled has vertices = {np.asarray(subsampled_mesh.vertices).shape[0]}, faces = {np.asarray(subsampled_mesh.triangles).shape[0]}")


def convert_pyvista_to_open3d(pyvista_mesh):
    """
    Utility function that converts the pyvista mesh into the open3d equivalent
    :param pyvista_mesh: pyvista mesh object
    :return: open3d mesh
    """
    # Extract vertices and faces from the PyVista mesh
    vertices = pyvista_mesh.points
    faces = pyvista_mesh.faces.reshape((-1, 4))[:, 1:]  # Remove the first column which is the number of points per face

    # Create an Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()

    # Assign vertices and faces to the Open3D mesh
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh

def convert_to_watertight_mesh(mesh_file:str):
    """
    Takes a mesh and converts it into a water-tight one.
    :param mesh_file: Mesh filename
    :return: None
    """
    orig_mesh = o3d.io.read_triangle_mesh(mesh_file)
    # orig_mesh.plot_boundaries()
    print(f"{orig_mesh.is_watertight()=}")

    vertices = np.asarray(orig_mesh.vertices)
    faces = np.asarray(orig_mesh.triangles)
    meshfix = MeshFix(vertices, faces)
    # holes = meshfix.extract_holes()
    meshfix.repair()
    # meshfix.plot()
    mesh = meshfix.mesh
    open3d_mesh = convert_pyvista_to_open3d(mesh)
    print(open3d_mesh.is_watertight())


if __name__ == '__main__':
    mesh_file = '/mnt/dog/chinmay/temp_outputs/mesh_checks/kidney_00010.stl'
    convert_to_watertight_mesh(mesh_file)

