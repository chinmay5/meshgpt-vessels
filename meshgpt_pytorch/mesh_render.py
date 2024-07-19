import os.path
import shutil

import numpy as np
import math


def orient_triangle_upward(v1, v2, v3):
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)

    up = np.array([0, 1, 0])
    if np.dot(normal, up) < 0:
        v1, v3 = v3, v1
    return v1, v2, v3


def get_angle(v1, v2, v3):
    v1, v2, v3 = orient_triangle_upward(v1, v2, v3)
    vec1 = v2 - v1
    vec2 = v3 - v1
    angle_rad = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return math.degrees(angle_rad)


def combind_mesh_with_rows(path, meshes):
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    translation_distance = 0.5
    obj_file_content = ""

    for row, mesh in enumerate(meshes):
        for r, faces_coordinates in enumerate(mesh):
            numpy_data = faces_coordinates[0].cpu().numpy().reshape(-1, 3)
            numpy_data[:, 0] += translation_distance * (r / 0.2 - 1)
            numpy_data[:, 2] += translation_distance * (row / 0.2 - 1)

            for vertex in numpy_data:
                all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            for i in range(1, len(numpy_data), 3):
                all_faces.append(f"f {i + vertex_offset} {i + 1 + vertex_offset} {i + 2 + vertex_offset}\n")

            vertex_offset += len(numpy_data)

        obj_file_content = "".join(all_vertices) + "".join(all_faces)

    with open(path, "w") as file:
        file.write(obj_file_content)


def save_rendering(path, input_meshes):
    all_vertices, all_faces = [], []
    vertex_offset = 0
    translation_distance = 0.5
    obj_file_content = ""
    meshes = input_meshes if isinstance(input_meshes, list) else [input_meshes]

    for row, mesh in enumerate(meshes):
        mesh = mesh if isinstance(mesh, list) else [mesh]
        cell_offset = 0
        for tensor, mask in mesh:
            for tensor_batch, mask_batch in zip(tensor, mask):
                numpy_data = tensor_batch[mask_batch].cpu().numpy().reshape(-1, 3)
                numpy_data[:, 0] += translation_distance * (cell_offset / 0.2 - 1)
                numpy_data[:, 2] += translation_distance * (row / 0.2 - 1)
                cell_offset += 1
                for vertex in numpy_data:
                    all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

                mesh_center = np.mean(numpy_data, axis=0)
                for i in range(0, len(numpy_data), 3):
                    v1 = numpy_data[i]
                    v2 = numpy_data[i + 1]
                    v3 = numpy_data[i + 2]

                    normal = np.cross(v2 - v1, v3 - v1)
                    if get_angle(v1, v2, v3) > 60:
                        direction_vector = mesh_center - np.mean([v1, v2, v3], axis=0)
                        direction_vector = -direction_vector
                    else:
                        direction_vector = [0, 1, 0]

                    if np.dot(normal, direction_vector) > 0:
                        order = [0, 1, 2]
                    else:
                        order = [0, 2, 1]

                    reordered_vertices = [v1, v2, v3][order[0]], [v1, v2, v3][order[1]], [v1, v2, v3][order[2]]
                    indices = [np.where((numpy_data == vertex).all(axis=1))[0][0] + 1 + vertex_offset for vertex in
                               reordered_vertices]
                    all_faces.append(f"f {indices[0]} {indices[1]} {indices[2]}\n")

                vertex_offset += len(numpy_data)
        obj_file_content = "".join(all_vertices) + "".join(all_faces)

    with open(path, "w") as file:
        file.write(obj_file_content)

    print(f"[Save_rendering] Saved at {path}")


def create_individual_obj_file_pairs(save_folder, pred_gt_meshes):
    translation_distance = 1
    # We create a new base folder where we put all the entries
    print("cleaning existing folder")
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder)

    for idx, (pred, gt) in enumerate(zip(*pred_gt_meshes)):
        numpy_data_pred = pred[0].cpu().numpy().reshape(-1, 3)
        numpy_data_gt = gt[0].cpu().numpy().reshape(-1, 3)
        # Let us translate the gt data a bit for ease in visualization
        numpy_data_gt[:, 0] += translation_distance
        numpy_data_gt[:, 2] += translation_distance

        # Create gt and pred vertex list
        pred_vertices, gt_vertices = [], []
        for vertex in numpy_data_pred:
            pred_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for vertex in numpy_data_gt:
            gt_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Create gt and pred face list
        pred_faces, gt_faces = [], []
        for i in range(1, len(numpy_data_pred), 3):
            pred_faces.append(f"f {i} {i + 1} {i + 2}\n")
        # An offset so that we can view both the meshes in the same obj file
        vertex_offset = len(numpy_data_pred)
        for i in range(1, len(numpy_data_gt), 3):
            gt_faces.append(f"f {i + vertex_offset} {i + 1 + vertex_offset} {i + 2 + vertex_offset}\n")

        obj_file_content = "".join(gt_vertices) + "".join(pred_vertices) + "".join(gt_faces) + "".join(pred_faces)

        with open(f"{save_folder}/gt_pred_pair_{idx}.obj", "w") as file:
            file.write(obj_file_content)