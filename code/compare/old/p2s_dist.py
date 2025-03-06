import os
import numpy as np
import trimesh
import pyvista as pv
from tqdm import tqdm

# Body part color mappings
BODY_PART_LABELS = {
    (0.0, 0.8, 0.8): "Legs",
    (1.0, 0.0, 1.0): "Bottom-Half Clothes",
    (0.0, 0.0, 1.0): "Shoes",
    (1.0, 1.0, 0.0): "Top-Half Clothes", 
    (1.0, 0.5, 0.0): "Hands",
    (0.0, 1.0, 0.0): "Hair",
    (1.0, 0.0, 0.0): "Head"
}

# Define directories directly in the script
SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files'
NON_SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_aligned_files'

def load_obj_with_colors(file_path):
    """
    Custom loader for OBJ files with vertex colors.
    
    Args:
        file_path: Path to the OBJ file
    
    Returns:
        pv.PolyData: PyVista mesh
    """
    # First try using trimesh which handles vertex colors better
    try:
        mesh = trimesh.load(file_path)
        # Convert to PyVista
        vertices = mesh.vertices
        faces = mesh.faces
        if faces.shape[1] == 3:  # Triangular mesh
            faces = np.column_stack((np.full(len(faces), 3), faces))
        pv_mesh = pv.PolyData(vertices, faces)
        
        # If there are vertex colors, add them
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Normalize to 0-1
            pv_mesh['colors'] = colors
            
        return pv_mesh
    except Exception as e:
        print(f"Warning: Trimesh failed to load the mesh properly: {e}")

def extract_labeled_parts(obj_file_path):
    """
    Extract vertices for each labeled part from an OBJ file with vertex colors
    
    Args:
        obj_file_path: Path to the OBJ file with vertex colors
    
    Returns:
        dict: Dictionary mapping RGB color tuples to arrays of vertices for that color
        np.array: Array of all vertices
    """
    # Initialize storage for vertices and their colors
    vertices = []
    vertex_colors = []
    
    # Read OBJ file
    with open(obj_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            if parts[0] == 'v' and len(parts) >= 7:
                # Extract vertex coordinates (x, y, z)
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(coords)
                
                # Extract vertex color (r, g, b)
                color = [float(parts[4]), float(parts[5]), float(parts[6])]
                vertex_colors.append(color)
    
    # Convert to numpy arrays
    vertices = np.array(vertices)
    vertex_colors = np.array(vertex_colors)
    
    # Group vertices by color
    labeled_parts = {}
    for i, color in enumerate(vertex_colors):
        color_tuple = tuple(color)
        if color_tuple not in labeled_parts:
            labeled_parts[color_tuple] = []
        labeled_parts[color_tuple].append(vertices[i])
    
    # Convert lists to numpy arrays
    for color in labeled_parts:
        labeled_parts[color] = np.array(labeled_parts[color])
    
    return labeled_parts, vertices

def compute_bounding_box(vertices, padding=0.05):
    """
    Compute bounding box with optional padding
    """
    if len(vertices) == 0:
        return np.array([0, 0, 0]), np.array([0, 0, 0])
    
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    
    # Add padding
    bbox_size = max_vals - min_vals
    min_vals -= bbox_size * padding
    max_vals += bbox_size * padding
    
    return min_vals, max_vals

def compute_point_to_surface_distance(points, mesh):
    """
    Compute the minimum distance from each point to the mesh surface
    """
    if len(points) == 0:
        return float('inf')
    
    # Use trimesh's built-in function to find closest points on the mesh
    distances, _, _ = trimesh.proximity.closest_point(mesh, points)
    
    # Return the average distance
    return np.mean(distances)

def process_file_pair(segmented_file, non_segmented_file):
    """
    Process a pair of files and compute point-to-surface distances for each body part
    """
    # Extract labeled parts from segmented file
    labeled_parts, segmented_all_vertices = extract_labeled_parts(segmented_file)
    
    # Load non-segmented mesh
    non_segmented_mesh = trimesh.load(non_segmented_file)
    
    # Get filename for reporting purposes
    filename = os.path.basename(segmented_file)
    
    # Print only the filename
    print(f"\n{filename}")
    print("Point-to-surface distances by body part:")
    
    # Calculate overall point-to-surface distance first
    overall_p2s_dist = compute_point_to_surface_distance(segmented_all_vertices, non_segmented_mesh)
    print(f"Overall shape: {len(segmented_all_vertices)} vertices, "
          f"Point-to-surface distance: {overall_p2s_dist:.6f} units")
    
    # Check if overall distance exceeds threshold
    if overall_p2s_dist > 50:
        print(f"HIGH DISTANCE ALERT - File: {filename}, Label: Overall, Distance: {overall_p2s_dist:.6f}")
    
    # Calculate point-to-surface distance for each labeled part
    results = {'overall': overall_p2s_dist}
    for color, part_vertices in labeled_parts.items():
        if color in BODY_PART_LABELS:
            part_name = BODY_PART_LABELS[color]
            
            # Compute point-to-surface distance for this part
            p2s_dist = compute_point_to_surface_distance(part_vertices, non_segmented_mesh)
            
            results[color] = p2s_dist
            print(f"{part_name}: {len(part_vertices)} vertices, "
                  f"Point-to-surface distance: {p2s_dist:.6f} units")
            
            # Check if distance exceeds threshold
            if p2s_dist > 50:
                print(f"HIGH DISTANCE ALERT - File: {filename}, Label: {part_name}, Distance: {p2s_dist:.6f}")
    
    return results

def main():
    # Get list of files in both directories
    segmented_files = [f for f in os.listdir(SEGMENTED_DIR) if f.endswith('.obj')]
    non_segmented_files = [f for f in os.listdir(NON_SEGMENTED_DIR) if f.endswith('.obj')]
    
    # Find common file names
    seg_basenames = {os.path.splitext(f)[0] for f in segmented_files}
    non_seg_basenames = {os.path.splitext(f)[0] for f in non_segmented_files}
    common_basenames = seg_basenames.intersection(non_seg_basenames)
    
    print(f"Found {len(common_basenames)} matching file pairs")
    
    # Process each file pair
    all_results = {}
    for basename in tqdm(common_basenames):
        segmented_file = os.path.join(SEGMENTED_DIR, f"{basename}.obj")
        non_segmented_file = os.path.join(NON_SEGMENTED_DIR, f"{basename}.obj")
        
        try:
            results = process_file_pair(segmented_file, non_segmented_file)
            all_results[basename] = results
        except Exception as e:
            print(f"Error processing {basename}: {e}")
    
    # Compute average results across all files
    print("\n----- SUMMARY -----")
    print(f"Processed {len(all_results)} file pairs successfully")
    
    # Calculate overall average first
    overall_distances = [results.get('overall', float('nan')) for results in all_results.values()]
    valid_overall_distances = [d for d in overall_distances if not np.isnan(d) and d != float('inf')]
    if valid_overall_distances:
        avg_overall_distance = np.mean(valid_overall_distances)
        print(f"Average overall point-to-surface distance: {avg_overall_distance:.6f} units")
    
    # Aggregate results by body part
    body_part_results = {}
    for color, part_name in BODY_PART_LABELS.items():
        distances = [results.get(color, float('nan')) for results in all_results.values()]
        valid_distances = [d for d in distances if not np.isnan(d) and d != float('inf')]
        
        if valid_distances:
            avg_distance = np.mean(valid_distances)
            body_part_results[part_name] = avg_distance
            print(f"Average point-to-surface distance for {part_name}: {avg_distance:.6f} units")

if __name__ == "__main__":
    main()