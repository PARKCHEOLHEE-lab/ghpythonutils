import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms, utils

from APIs.mesh2points.mesh2pointsutils import (
    PointSampler, 
    RandomNoise, 
    RnadomRotationZ, 
    Normalize, 
    PointCloudData, 
    point_cloud_show, 
    load_point_cloud
)


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
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    data_dir = os.path.abspath(os.path.join(__file__, "../data/ModelNet10"))

    # sample_index = 0
    # data_sofa_sample = glob.glob(
    #     os.path.abspath(os.path.join(data_dir, "sofa/train/*"))
    # )[sample_index]
    # sofa_sample_vertices, sofa_sample_faces = load_point_cloud(data_sofa_sample)

    # sofa_sample_point_cloud = PointSampler(1024)(
    #     (sofa_sample_vertices, sofa_sample_faces)
    # )
    
    # point_cloud_show(
    #     sofa_sample_vertices,
    #     sofa_sample_point_cloud,
    #     labels=["sofa_sample_vertices", "sofa_sample_point_cloud"],
    # )
    
    # point_cloud_show(
    #     Normalize()(sofa_sample_vertices),
    #     Normalize()(sofa_sample_point_cloud),
    #     labels=["sofa_sample_vertices_normalized", "sofa_sample_point_cloud_normalized"]
    # )
    
    # normalized = Normalize()(sofa_sample_point_cloud)
    # noised = RandomNoise()(normalized)
    
    # point_cloud_show(
    #     normalized,
    #     noised,
    #     labels=["normalized", "noised"]
    # )
    
    train_transforms = transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            RnadomRotationZ(),
            RandomNoise(),
            transforms.ToTensor()
        ]
    )

    train_dataset = PointCloudData(data_dir=data_dir, transform=train_transforms)
    valid_dataset = PointCloudData(
        data_dir=data_dir, folder="test", transform=train_transforms
    )

    classes = {
        label: category for category, label in train_dataset.labels.items()
    }

    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))
    print("Number of classes:", len(classes))

    sample_point_cloud, sample_point_cloud_class = train_dataset[0]
    print("Sample pointcloud shape:", sample_point_cloud[0].shape)
    print("Sample pointcloud class:", classes[sample_point_cloud_class])

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=64)