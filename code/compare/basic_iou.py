import trimesh
import numpy as np

def normalize_mesh(mesh):
    """
    Center and scale mesh to fit in a unit cube.
    
    Args:
        mesh: trimesh.Trimesh object
    
    Returns:
        trimesh.Trimesh: Normalized mesh
    """
    # Make a copy to avoid modifying original
    mesh = mesh.copy()
    
    # Center
    mesh.vertices -= mesh.center_mass
    
    # Scale to unit cube
    scale = 1.0 / np.max(np.abs(mesh.vertices))
    mesh.vertices *= scale
    
    return mesh

def calculate_shape_iou(mesh1, mesh2, voxel_resolution=64):
    """
    Calculate Intersection over Union (IoU) between two 3D meshes.
    
    Args:
        mesh1: First trimesh mesh
        mesh2: Second trimesh mesh
        voxel_resolution: Resolution of voxel grid (default: 64)
    
    Returns:
        tuple: (iou_score, voxels1, voxels2)
            - iou_score (float): The IoU score
            - voxels1 (np.ndarray): Voxelized representation of first mesh
            - voxels2 (np.ndarray): Voxelized representation of second mesh
    """

    
    # Voxelize with same resolution
    pitch = 2.0 / voxel_resolution
    vox1 = mesh1.voxelized(pitch=pitch)
    vox2 = mesh2.voxelized(pitch=pitch)
    
    # Get matrices
    mat1 = vox1.matrix
    mat2 = vox2.matrix
    
    # Ensure same dimensions by padding with zeros
    max_x = max(mat1.shape[0], mat2.shape[0])
    max_y = max(mat1.shape[1], mat2.shape[1])
    max_z = max(mat1.shape[2], mat2.shape[2])
    
    # Pad first matrix if needed
    if mat1.shape != (max_x, max_y, max_z):
        pad_x = max_x - mat1.shape[0]
        pad_y = max_y - mat1.shape[1]
        pad_z = max_z - mat1.shape[2]
        mat1 = np.pad(mat1, ((0, pad_x), (0, pad_y), (0, pad_z)))
    
    # Pad second matrix if needed
    if mat2.shape != (max_x, max_y, max_z):
        pad_x = max_x - mat2.shape[0]
        pad_y = max_y - mat2.shape[1]
        pad_z = max_z - mat2.shape[2]
        mat2 = np.pad(mat2, ((0, pad_x), (0, pad_y), (0, pad_z)))
    
    # Calculate intersection and union
    intersection = np.logical_and(mat1, mat2)
    union = np.logical_or(mat1, mat2)
    
    # Calculate IoU
    union_sum = np.sum(union)
    if union_sum == 0:
        return 0.0, mat1, mat2
        
    iou = np.sum(intersection) / union_sum
    return iou, mat1, mat2


# Example usage
if __name__ == "__main__":
    # Load two similar meshes
    mesh1 = trimesh.load_mesh("/Users/paulinagerchuk/Downloads/Outer/Take9/Code/obj_4d-dress_files/output.obj")
    mesh2 = trimesh.load_mesh("/Users/paulinagerchuk/Downloads/Outer/Take9/Code/obj_4d-dress_files/output.obj")
    
    # Calculate IoU
    iou_score, voxels1, voxels2 = calculate_shape_iou(mesh1, mesh2)
    print(f"IoU Score: {iou_score:.4f}")
    print(f"Voxel shapes: {voxels1.shape}, {voxels2.shape}")

