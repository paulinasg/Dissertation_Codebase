import trimesh
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

# Add color mappings from old version
BODY_PART_LABELS = {
    (0.0, 0.8, 0.8): "Legs",
    (1.0, 0.0, 1.0): "Bottom-Half Clothes",
    (0.0, 0.0, 1.0): "Shoes",
    (1.0, 1.0, 0.0): "Top-Half Clothes",
    (1.0, 0.5, 0.0): "Hands",
    (0.0, 1.0, 0.0): "Hair",
    (1.0, 0.0, 0.0): "Head"
}

def compute_chamfer_distance(points1, points2):
    """
    Compute chamfer distance between two point clouds
    """
    # Build kd-trees for both point sets
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Find nearest neighbors in both directions
    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)
    
    # Compute chamfer distance
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return chamfer_dist

def extract_labeled_parts(file_path):
    """
    Extract vertices for each labeled part based on RGB values directly from OBJ file
    """
    vertices = []
    vertex_colors = []
    labeled_parts = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Get vertex coordinates
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                # Get color values
                color = tuple(float(x) for x in parts[4:7])
                
                # Add vertex to corresponding color group
                if color not in labeled_parts:
                    labeled_parts[color] = []
                labeled_parts[color].append(vertex)
    
    # Convert lists to numpy arrays
    for color in labeled_parts:
        labeled_parts[color] = np.array(labeled_parts[color])
        part_name = BODY_PART_LABELS.get(color, "Unknown Part")
        # print(f"Found {part_name} with {len(labeled_parts[color])} vertices")
    
    return labeled_parts

def main():
    # Load the segmented and non-segmented meshes
    segmented_file = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/00163.obj'
    non_segmented_file = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_aligned_files/00163.obj'
    
    # Extract labeled parts directly from OBJ file
    labeled_parts = extract_labeled_parts(segmented_file)
    
    # Load non-segmented mesh
    non_segmented_mesh = trimesh.load(non_segmented_file)
    non_segmented_vertices = np.array(non_segmented_mesh.vertices)
    
    # Print model scale information
    bbox_size = np.ptp(non_segmented_vertices, axis=0)
    print(f"\nModel scale information:")
    print(f"Model bounding box dimensions (x,y,z): {bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f}")
    print(f"Model height (y-axis): {bbox_size[1]:.3f} units")
    print("\nChamfer distances:")
    
    # Calculate chamfer distance for each labeled part
    results = {}
    for color, part_vertices in labeled_parts.items():
        chamfer_dist = compute_chamfer_distance(part_vertices, non_segmented_vertices)
        results[color] = chamfer_dist
        part_name = BODY_PART_LABELS.get(color, "Unknown Part")
        print(f"{part_name}: {len(part_vertices)} vertices, Chamfer distance: {chamfer_dist:.6f} units")
    
    """ NOT USING VISUALISATION
    # Visualize using PyVista
    plotter = pv.Plotter()
    
    # Add segmented mesh with proper colors
    for color, part_vertices in labeled_parts.items():
        if len(part_vertices) > 0:
            cloud = pv.PolyData(part_vertices)
            plotter.add_mesh(cloud, color=color)
    
    # Add non-segmented mesh as wireframe
    non_seg_cloud = pv.PolyData(non_segmented_vertices)
    plotter.add_mesh(non_seg_cloud, style='wireframe', color='black')
    
    plotter.show()
    """

if __name__ == "__main__":
    main()