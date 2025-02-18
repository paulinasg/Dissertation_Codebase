import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
import trimesh

def decimate_points(vertices, faces, target_reduction=0.5):
    """
    Decimate the mesh to reduce vertex count while preserving shape
    
    Parameters:
    vertices: np.array of vertex coordinates
    faces: np.array of face indices
    target_reduction: float between 0 and 1, amount to reduce mesh by
    
    Returns:
    np.array of decimated vertices
    """
    # Create PyVista mesh
    pv_mesh = pv.PolyData()
    pv_mesh.points = vertices
    pv_mesh.faces = np.hstack((np.full((len(faces), 1), 3), faces))
    
    # Decimate the mesh
    decimated = pv_mesh.decimate(target_reduction)
    
    # Extract just the vertices
    return np.array(decimated.points)

# Define color to body part mapping
BODY_PART_LABELS = {
    (0.5, 1.0, 0.0): "Left Leg",
    (0.5, 0.0, 1.0): "Right Leg",
    (1.0, 0.0, 1.0): "Bottom-Half Clothes",
    (0.0, 0.0, 1.0): "Shoes",
    (1.0, 1.0, 0.0): "Top-Half Clothes",
    (0.0, 0.5, 1.0): "Left Hand",
    (1.0, 0.5, 0.0): "Right Hand",
    (0.0, 1.0, 0.0): "Hair",
    (1.0, 0.0, 0.0): "Head"
}

# Define color names
COLOR_NAMES = {
    (0.5, 1.0, 0.0): "Lime Green",
    (0.5, 0.0, 1.0): "Purple",
    (1.0, 0.0, 1.0): "Magenta",
    (0.0, 0.0, 1.0): "Blue",
    (1.0, 1.0, 0.0): "Yellow",
    (0.0, 0.5, 1.0): "Light Blue",
    (1.0, 0.5, 0.0): "Orange",
    (0.0, 1.0, 0.0): "Green",
    (1.0, 0.0, 0.0): "Red"
}

def normalize_points(points):
    """
    Normalize points to fit in a unit cube centered at origin
    """
    # Center points
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale to unit cube
    scale = np.max(np.abs(points))
    points = points / scale
    
    return points

def load_obj_with_colors(file_path, target_reduction=0.5):
    """
    Load OBJ file and group vertices by their colors
    Returns a dictionary of color: vertices pairs
    
    Parameters:
    file_path: str, path to OBJ file
    target_reduction: float between 0 and 1, amount to reduce mesh by
    """
    vertices = []
    faces = []
    colors = []
    vertex_colors = []  # Store colors in vertex order
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Get vertex coordinates
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
                # Get color values
                color = [float(parts[4]), float(parts[5]), float(parts[6])]
                vertex_colors.append(color)
            elif line.startswith('f '):
                # Get face indices (assuming 1-based indexing in OBJ file)
                face_parts = line.strip().split()[1:]
                face = [int(fp.split('/')[0]) - 1 for fp in face_parts]
                faces.append(face)
    
    # Convert to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)
    vertex_colors = np.array(vertex_colors)
    
    # Group vertices and faces by color
    color_groups = {}
    unique_colors = np.unique(vertex_colors, axis=0)
    
    print(f"Original vertex count: {len(vertices)}")
    print(f"Original face count: {len(faces)}")
    
    for color in unique_colors:
        # Get vertices of this color
        mask = np.all(vertex_colors == color, axis=1)
        color_vertices = vertices[mask]
        
        # Get faces that use these vertices
        vertex_indices = np.where(mask)[0]
        face_mask = np.isin(faces, vertex_indices).all(axis=1)
        color_faces = faces[face_mask]
        
        # Remap face indices to new vertex subset
        vertex_map = {old: new for new, old in enumerate(vertex_indices)}
        remapped_faces = np.array([[vertex_map[idx] for idx in face] for face in color_faces])
        
        if len(color_vertices) > 0 and len(remapped_faces) > 0:
            # Decimate this part
            decimated_vertices = decimate_points(color_vertices, remapped_faces, target_reduction)
            # Normalize the decimated vertices
            normalized_vertices = normalize_points(decimated_vertices)
            color_groups[tuple(color)] = normalized_vertices
            
            body_part = BODY_PART_LABELS.get(tuple(color), "Unknown Part")
            color_name = COLOR_NAMES.get(tuple(color), "Unknown Color")
            print(f"\n{body_part} ({color_name} - RGB{tuple(color)}):")
            print(f"  - Original vertices: {len(color_vertices)}")
            print(f"  - Decimated vertices: {len(decimated_vertices)}")
    
    return color_groups

def chamfer_distance(points1, points2):
    """
    Calculate the Chamfer distance between two point clouds
    """
    # Create KD trees for efficient nearest neighbor search
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Find nearest neighbors in both directions
    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)
    
    # Calculate average distance in both directions
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return chamfer_dist / 2.0

def compare_colored_parts(obj1_path, obj2_path):
    """
    Compare corresponding colored parts between two OBJ models
    Returns a dictionary of color: chamfer_distance pairs
    """
    # Load both models
    model1_parts = load_obj_with_colors(obj1_path)
    model2_parts = load_obj_with_colors(obj2_path)
    
    # Compare each colored part
    distances = {}
    
    # Find common colors between models
    common_colors = set(model1_parts.keys()) & set(model2_parts.keys())
    
    for color in common_colors:
        points1 = model1_parts[color]
        points2 = model2_parts[color]
        
        # Calculate Chamfer distance for this part
        dist = chamfer_distance(points1, points2)
        distances[color] = dist
        
    return distances

# Example usage
if __name__ == "__main__":
    obj1_path = "/Users/paulinagerchuk/Downloads/Outer/Take9/Code/smpl_segmentation/colored_model_woman.obj"
    obj2_path = "/Users/paulinagerchuk/Downloads/Outer/Take9/Code/pifuhd_segmentation/woman_colored_pi.obj"
    
    distances = compare_colored_parts(obj1_path, obj2_path)
    
    # Print results with interpretation
    print("\nChamfer distances for each body part:")
    for color, distance in distances.items():
        body_part = BODY_PART_LABELS.get(color, "Unknown Part")
        color_name = COLOR_NAMES.get(color, "Unknown Color")
        print(f"\n{body_part} ({color_name} - RGB{color}): {distance:.6f}")
        # Interpret the distance (assuming normalized coordinates)
        if distance < 0.02:
            print("  → Excellent match - parts are nearly identical")
        elif distance < 0.05:
            print("  → Good match - minor differences")
        elif distance < 0.10:
            print("  → Moderate differences")
        else:
            print("  → Significant differences detected")
