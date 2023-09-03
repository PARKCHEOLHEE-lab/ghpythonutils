import os
import trimesh


if __name__ == "__main__":
    path = os.path.abspath(os.path.join(__file__, "../data/ModelNet10"))
    
    mesh = trimesh.load(os.path.join(path, "chair/train/chair_0001.off"))
    mesh.show()