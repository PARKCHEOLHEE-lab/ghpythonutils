import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils

random.seed(777)


def load_point_cloud(file_path: str) -> np.array:
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        num_vertices, _, _ = map(int, lines[1].split())

        # Parse vertices
        vertices = []
        for line in lines[2 : 2 + num_vertices]:
            x, y, z = map(float, line.strip().split())
            vertices.append([x, y, z])

        # Parse faces
        faces = []
        for line in lines[2 + num_vertices :]:
            parts = line.strip().split()
            if len(parts) > 3:  # Ignore faces with less than 3 vertices
                num_vertices_in_face = int(parts[0])
                face_indices = list(map(int, parts[1:]))
                if len(face_indices) == num_vertices_in_face:
                    faces.append(face_indices)

        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)

        return vertices, faces

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


class PointSampler:
    def __init__(self, output_size: int):
        self.output_size = output_size

    def _compute_triangle_area(
        self, point_1: np.ndarray, point_2: np.ndarray, point_3: np.ndarray
    ) -> float:
        x1, y1, z1 = point_1
        x2, y2, z2 = point_2
        x3, y3, z3 = point_3

        area = 0.5 * abs(
            x1 * (y2 * z3 - y3 * z2)
            + x2 * (y3 * z1 - y1 * z3)
            + x3 * (y1 * z2 - y2 * z1)
        )

        return area

    def _compute_sample_point(
        self, point_1: np.ndarray, point_2: np.ndarray, point_3: np.ndarray
    ):
        start, end = sorted([random.random(), random.random()])

        random_x = (
            start * point_1[0]
            + (end - start) * point_2[0]
            + (1 - end) * point_3[0]
        )
        random_y = (
            start * point_1[1]
            + (end - start) * point_2[1]
            + (1 - end) * point_3[1]
        )
        random_z = (
            start * point_1[2]
            + (end - start) * point_2[2]
            + (1 - end) * point_3[2]
        )

        return (random_x, random_y, random_z)

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = self._compute_triangle_area(
                verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]]
            )

        sampled_faces = random.choices(
            faces, weights=areas, cum_weights=None, k=self.output_size
        )

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = self._compute_sample_point(
                verts[sampled_faces[i][0]],
                verts[sampled_faces[i][1]],
                verts[sampled_faces[i][2]],
            )

        return sampled_points


class PointCloudData(Dataset):
    def __init__(
        self, data_dir: str, folder: str = "train", transform: transforms = None
    ):
        self.data_dir = data_dir
        self.folder = folder
        self.transform = transform
        self.categories = [
            category for category in os.listdir(data_dir) if "." not in category
        ]
        self.labels = {
            category: i for i, category in enumerate(self.categories)
        }

    def __len__(self):
        length = 0

        for category in self.categories:
            category_dir = glob.glob(
                os.path.join(self.data_dir, category, f"{self.folder}/*")
            )
            category_dir = [
                name for name in category_dir if name.endswith(".off")
            ]

            length += len(category_dir)

        return length

    def __getitem__(self, idx):
        num_samples_per_category = len(self) // len(self.categories)
        category_idx = idx // num_samples_per_category
        sample_idx_within_category = idx % num_samples_per_category

        category = self.categories[category_idx]
        category_dir = glob.glob(
            os.path.join(self.data_dir, category, f"{self.folder}/*")
        )
        category_dir = [name for name in category_dir if name.endswith(".off")]

        filename = category_dir[sample_idx_within_category]

        point_cloud = load_point_cloud(filename)

        if self.transform:
            point_cloud = self.transform(point_cloud)

        label = self.labels[category]

        return point_cloud, label


class TNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(__file__, "../data/ModelNet10"))

    sample_index = 0
    data_sofa_sample = glob.glob(
        os.path.abspath(os.path.join(data_dir, "sofa/train/*"))
    )[sample_index]
    vertices, faces = load_point_cloud(data_sofa_sample)

    point_cloud = PointSampler(1024)((vertices, faces))

    train_dataset = PointCloudData(data_dir=data_dir, transform=None)
    valid_dataset = PointCloudData(
        data_dir=data_dir, folder="test", transform=None
    )

    classes = {
        label: category for category, label in train_dataset.labels.items()
    }

    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))
    print("Number of classes:", len(classes))

    sample_point_cloud, sample_point_cloud_class = train_dataset[0]
    print("Sample pointcloud shape:", sample_point_cloud.shape)
    print("Sample pointcloud class:", classes[sample_point_cloud_class])
