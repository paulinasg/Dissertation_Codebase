import numpy as np
import trimesh
import pickle

def inspect_files(ply_path, pkl_path):
    print("\n=== PLY File Analysis ===")
    # Load and inspect PLY
    mesh = trimesh.load(ply_path)
    print(f"PLY Mesh Info:")
    print(f"- Number of vertices: {len(mesh.vertices)}")
    print(f"- Number of faces: {len(mesh.faces)}")
    print(f"- Vertex shape: {mesh.vertices.shape}")
    print(f"- Face shape: {mesh.faces.shape}")
    print(f"- Bounds min: {mesh.bounds[0]}")
    print(f"- Bounds max: {mesh.bounds[1]}")
    
    print("\n=== PKL File Analysis ===")
    # Load and inspect PKL
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("PKL Contents:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"- {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"- {key}: type {type(value)}")

def main():
    ply_path = "SMPLX/mesh-f00011_smplx.ply"
    pkl_path = "SMPLX/mesh-f00011_smplx.pkl"
    
    inspect_files(ply_path, pkl_path)

if __name__ == "__main__":
    main()