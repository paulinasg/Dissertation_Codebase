import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import warnings
import vtk
from scipy.spatial import KDTree

def rigid_icp(source_mesh, target_mesh, max_iterations=50, tolerance=1e-6, try_flipped=True):
    """
    Perform rigid ICP to align source_mesh to target_mesh.
    try_flipped: If True, try both original and 180° rotated orientations
    """
    def run_icp(source_points):
        points = source_points.copy()
        target_points = target_mesh.vertices
        target_kd_tree = KDTree(target_points)
        prev_error = float('inf')
        best_points = points.copy()
        min_error = float('inf')
        
        for iteration in range(max_iterations):
            _, indices = target_kd_tree.query(points)
            target_correspondences = target_points[indices]
            
            source_centroid = np.mean(points, axis=0)
            target_centroid = np.mean(target_correspondences, axis=0)
            
            points_centered = points - source_centroid
            target_correspondences_centered = target_correspondences - target_centroid
            
            H = np.dot(points_centered.T, target_correspondences_centered)
            U, _, Vt = np.linalg.svd(H)
            R_icp = np.dot(Vt.T, U.T)
            t_icp = target_centroid - np.dot(source_centroid, R_icp)
            
            points = np.dot(points, R_icp.T) + t_icp
            
            error = np.mean(np.linalg.norm(target_correspondences - points, axis=1))
            print(f"ICP Iteration {iteration + 1}, Error: {error}")
            
            if error < min_error:
                min_error = error
                best_points = points.copy()
            
            if np.abs(prev_error - error) < tolerance:
                break
                
            prev_error = error
            
        return best_points, min_error

    # Try original orientation
    points_original, error_original = run_icp(source_mesh.vertices)
    
    if try_flipped:
        # Try flipped orientation (180° rotation around Y-axis)
        rot_180 = Rotation.from_euler('y', np.pi).as_matrix()
        flipped_vertices = source_mesh.vertices @ rot_180.T
        points_flipped, error_flipped = run_icp(flipped_vertices)
        
        print(f"\nOriginal orientation error: {error_original}")
        print(f"Flipped orientation error: {error_flipped}")
        
        # Choose the better result
        if error_flipped < error_original:
            print("Using flipped orientation (better alignment)")
            return trimesh.Trimesh(vertices=points_flipped, faces=source_mesh.faces)
    
    print("Using original orientation")
    return trimesh.Trimesh(vertices=points_original, faces=source_mesh.faces)

def align_human_meshes_with_icp(mesh1_path: str, mesh2_path: str, decimate: bool = True) -> trimesh.Trimesh:
    """Align meshes using rigid ICP."""
    mesh1 = trimesh.load(mesh1_path)
    mesh2 = trimesh.load(mesh2_path)
    
    # Add diagnostic prints
    print(f"\nMesh vertex counts before processing:")
    print(f"Target mesh (mesh1): {len(mesh1.vertices):,} vertices")
    print(f"Source mesh (mesh2): {len(mesh2.vertices):,} vertices")
    
    init_scale, init_angles, init_trans = fast_initial_transform(mesh1, mesh2)
    R_init = Rotation.from_euler('xyz', init_angles).as_matrix()
    mesh2_transformed = init_scale * (mesh2.vertices @ R_init.T) + init_trans
    mesh2_transformed_mesh = trimesh.Trimesh(vertices=mesh2_transformed, faces=mesh2.faces)
    
# Explicitly enable orientation testing
    aligned_mesh = rigid_icp(mesh2_transformed_mesh, mesh1, try_flipped=True)
    
    if decimate:
        print(f"\nDecimating aligned mesh from {len(aligned_mesh.vertices):,} vertices to target {len(mesh1.vertices):,} vertices")
        aligned_mesh = fast_decimate(aligned_mesh, len(mesh1.vertices))
        print(f"After decimation: {len(aligned_mesh.vertices):,} vertices")
    
    return aligned_mesh

def fast_decimate(mesh, target_vertices: int) -> trimesh.Trimesh:
    """Decimate using PyVista with suppressed warnings."""
    # Suppress VTK warnings
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Convert trimesh faces to PyVista format
        faces_with_count = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces))
        faces_with_count = faces_with_count.flatten()
        
        mesh_pv = pv.PolyData(mesh.vertices, faces_with_count)
    
    # Calculate reduction ratio
    current_vertices = mesh_pv.n_points
    reduction_ratio = 1 - (target_vertices / current_vertices)
    reduction_ratio = np.clip(reduction_ratio, 0, 0.99)
    
    # Decimate
    decimated = mesh_pv.decimate(reduction_ratio)
    
    # Convert back to trimesh format
    vertices = np.array(decimated.points)
    faces = np.array(decimated.faces).reshape(-1, 4)[:, 1:4]
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def get_key_points(vertices):
    """Extract key points with better center calculation."""
    y_coords = vertices[:, 1]
    head_idx = np.argmax(y_coords)
    feet_idx = np.argmin(y_coords)
    
    # Calculate center as midpoint of bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    
    return vertices[head_idx], vertices[feet_idx], center

def get_facing_direction(vertices, center):
    """
    Improved facing direction detection using multiple approaches.
    Returns a more reliable facing direction vector.
    """
    # Project points onto XZ plane
    points_xz = vertices.copy()
    points_xz[:, 1] = 0  # zero out Y component
    center_xz = center.copy()
    center_xz[1] = 0
    
    # Method 1: Use point density distribution
    # Split the space into front/back hemispheres and compare point counts
    directions = points_xz - center_xz
    directions_normalized = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10)
    
    # Calculate point density in different sectors
    angles = np.arctan2(directions_normalized[:, 2], directions_normalized[:, 0])
    sectors = np.linspace(-np.pi, np.pi, 8)  # Split into 8 sectors
    counts = np.histogram(angles, bins=sectors)[0]
    
    # Find the densest sector
    densest_sector = sectors[np.argmax(counts)]
    
    # Method 2: Use asymmetry of the shape
    # Calculate the center of mass for the front half and back half
    front_mask = angles >= 0
    back_mask = ~front_mask
    if np.sum(front_mask) > 0 and np.sum(back_mask) > 0:
        front_com = np.mean(points_xz[front_mask], axis=0)
        back_com = np.mean(points_xz[back_mask], axis=0)
        asymmetry_vector = front_com - back_com
    else:
        asymmetry_vector = np.array([0, 0, 0])
    
    # Method 3: Use the original furthest point method as backup
    dists = np.linalg.norm(directions, axis=1)
    furthest_idx = np.argmax(dists)
    furthest_dir = directions_normalized[furthest_idx]
    
    # Combine the methods with weights
    final_direction = np.array([np.cos(densest_sector), 0, np.sin(densest_sector)]) * 0.5 + \
                     (asymmetry_vector / (np.linalg.norm(asymmetry_vector) + 1e-10)) * 0.3 + \
                     furthest_dir * 0.2
    
    # Normalize the result
    return final_direction / np.linalg.norm(final_direction)


def fast_initial_transform(mesh1, mesh2):
    """Compute quick initial transformation with improved facing direction detection."""
    # Get key points
    head1, feet1, center1 = get_key_points(mesh1.vertices)
    head2, feet2, center2 = get_key_points(mesh2.vertices)
    
    # Calculate scale based on height
    height1 = np.linalg.norm(head1 - feet1)
    height2 = np.linalg.norm(head2 - feet2)
    scale = height1 / height2
    
    # Calculate initial translation to align centers
    translation = center1 - (scale * center2)
    
    # Get facing directions using improved method
    dir1 = get_facing_direction(mesh1.vertices, center1)
    dir2 = get_facing_direction(mesh2.vertices, center2)
    
    # Calculate rotation angle between the two directions
    rot_y = np.arctan2(np.cross(dir2, dir1)[1], np.dot(dir1, dir2))
    
    # Check if we need to add 180 degrees
    # Test both possibilities and use the one that gives better alignment
    R1 = [0, rot_y, 0]
    R2 = [0, rot_y + np.pi, 0]
    
    # Quick test of both rotations
    rot1 = Rotation.from_euler('xyz', R1).as_matrix()
    rot2 = Rotation.from_euler('xyz', R2).as_matrix()
    
    # Modify subset_idx generation to ensure indices do not exceed number of vertices
    num_vertices = len(mesh1.vertices)
    subset_idx = np.linspace(0, num_vertices - 1, 100, endpoint=False, dtype=int)
    
    # Transform a subset of points for quick comparison
    points1 = scale * (mesh2.vertices[subset_idx] @ rot1.T) + translation
    points2 = scale * (mesh2.vertices[subset_idx] @ rot2.T) + translation
    
    # Compare distances to target points
    dist1 = np.mean(np.linalg.norm(points1 - mesh1.vertices[subset_idx], axis=1))
    dist2 = np.mean(np.linalg.norm(points2 - mesh1.vertices[subset_idx], axis=1))
    
    # Choose the better rotation
    R = R1 if dist1 < dist2 else R2
    
    return scale, R, translation


def main():
    """Example usage."""
    mesh1_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_files/inner/00122.obj"
    mesh2_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_files/inner/00122.obj"
    aligned_mesh = align_human_meshes_with_icp(mesh1_path, mesh2_path, decimate=True)
    aligned_mesh.export("00122.obj")

if __name__ == "__main__":
    main()
