import os
import numpy as np
import trimesh
from tqdm import tqdm
import trimesh.proximity

# Body part color mappings
BODY_PART_LABELS = {
    (0.0, 0.8, 0.8): "Legs",
    (1.0, 0.0, 1.0): "Bottom-Half Clothes",
    (0.0, 0.0, 1.0): "Shoes",
    (1.0, 1.0, 0.0): "Top-Half Clothes", 
    (1.0, 0.5, 0.0): "Head",
    (0.0, 1.0, 0.0): "Hair",
    (1.0, 0.0, 0.0): "Hands"
}

# Add color names for reporting
COLOR_NAMES = {
    (0.0, 0.8, 0.8): "Cyan",
    (1.0, 0.0, 1.0): "Magenta",
    (0.0, 0.0, 1.0): "Blue",
    (1.0, 1.0, 0.0): "Yellow", 
    (1.0, 0.5, 0.0): "Orange",
    (0.0, 1.0, 0.0): "Green",
    (1.0, 0.0, 0.0): "Red"
}

# Define directories
SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files'
NON_SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_labelled_files'

def load_obj_with_face_groups(file_path):
    """
    Load OBJ file and extract vertices, faces, and face groups by color
    """
    vertices = []
    faces = []
    vertex_colors = []
    face_groups = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Get vertex coordinates
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
                
                # Get color values if they exist
                if len(parts) >= 7:
                    color = tuple(float(parts[i]) for i in range(4, 7))
                    vertex_colors.append(color)
                else:
                    vertex_colors.append(None)
                    
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                # OBJ indices start from 1, so subtract 1 for 0-based indexing
                face = [int(part.split('/')[0]) - 1 for part in parts]
                faces.append(face)
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Assign faces to color groups based on vertex colors
    for face_idx, face in enumerate(faces):
        # Get the most common color for this face's vertices
        face_colors = [vertex_colors[v_idx] for v_idx in face if vertex_colors[v_idx] is not None]
        if not face_colors:
            continue
            
        # Simple majority vote for face color
        unique_colors, counts = np.unique(face_colors, axis=0, return_counts=True)
        face_color = tuple(unique_colors[np.argmax(counts)])
        
        if face_color not in face_groups:
            face_groups[face_color] = []
        face_groups[face_color].append(face_idx)
    
    # Convert lists to numpy arrays for each group
    for color in face_groups:
        face_groups[color] = np.array(face_groups[color])
    
    return vertices, faces, face_groups

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

def compute_face_centers(vertices, faces):
    """
    Compute the center point of each face
    """
    return np.mean(vertices[faces], axis=1)

def filter_faces_in_bbox(face_centers, bbox_min, bbox_max):
    """
    Filter face indices that have centers within a bounding box
    """
    mask = np.all((face_centers >= bbox_min) & (face_centers <= bbox_max), axis=1)
    return np.where(mask)[0]

def compute_face_normals(vertices, faces):
    """
    Compute normal vectors for each face
    """
    # Extract the vertices for each face
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute face normals using cross product
    normals = np.cross(v1 - v0, v2 - v0)
    
    # Normalize the normal vectors
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # Fix: Ensure mask has the right shape for boolean indexing
    zero_norm_indices = norms < 1e-10
    
    # Handle normalization without using boolean indexing
    normalized_normals = np.zeros_like(normals)
    valid_indices = np.where(~zero_norm_indices.flatten())[0]
    
    if len(valid_indices) > 0:
        normalized_normals[valid_indices] = normals[valid_indices] / norms[valid_indices]
    
    return normalized_normals

def compute_normal_consistency(vertices1, faces1, normals1, vertices2, faces2, normals2):
    """
    Compute normal consistency between two sets of normals using spatial correspondence
    with tolerance for flipped normals
    """
    if len(normals1) == 0 or len(normals2) == 0:
        return 0.0
    
    # Convert to trimesh for faster proximity queries
    mesh2 = trimesh.Trimesh(vertices=vertices2, faces=faces2, process=False)
    
    # Calculate face centers for the first mesh
    face_centers1 = compute_face_centers(vertices1, faces1)
    
    # For each face center in mesh1, find closest point on mesh2
    closest_points, _, triangle_ids = trimesh.proximity.closest_point(mesh2, face_centers1)
    
    # Get normals for the closest triangles
    closest_normals = normals2[triangle_ids]
    
    # Normalize all normals (ensure unit length)
    n1 = normals1 / np.maximum(np.linalg.norm(normals1, axis=1, keepdims=True), 1e-10)
    n2 = closest_normals / np.maximum(np.linalg.norm(closest_normals, axis=1, keepdims=True), 1e-10)
    
    # Compute absolute dot products (treating flipped normals as similar)
    dot_products = np.abs(np.sum(n1 * n2, axis=1))
    
    # Average the cosine similarities
    mean_consistency = np.mean(dot_products)
    return mean_consistency

def process_file_pair(segmented_file, non_segmented_file):
    """
    Process a pair of files and compute normal consistency for each body part
    """
    # Load segmented mesh with face groups
    seg_vertices, seg_faces, seg_face_groups = load_obj_with_face_groups(segmented_file)
    seg_face_normals = compute_face_normals(seg_vertices, seg_faces)
    
    # Load non-segmented mesh
    non_seg_mesh = trimesh.load(non_segmented_file)
    non_seg_vertices = np.array(non_seg_mesh.vertices)
    non_seg_faces = np.array(non_seg_mesh.faces)
    non_seg_face_normals = compute_face_normals(non_seg_vertices, non_seg_faces)
    
    # Print only the filename
    print(f"\n{os.path.basename(segmented_file)}")
    print("Normal consistency by body part:")
    
    # Calculate normal consistency for each labeled part
    results = {}
    for color, face_indices in seg_face_groups.items():
        if color in BODY_PART_LABELS:
            part_name = BODY_PART_LABELS[color]
            
            # Get part vertices and faces
            part_faces = seg_faces[face_indices]
            unique_vertices = np.unique(part_faces.flatten())
            part_vertices = seg_vertices[unique_vertices]
            
            # Reindex faces for this part only
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
            remapped_faces = np.array([[vertex_map[v] for v in face] for face in part_faces])
            
            # Get normals for this part
            part_normals = seg_face_normals[face_indices]
            
            # Compute normal consistency using spatial correspondence
            consistency = compute_normal_consistency(
                part_vertices, remapped_faces, part_normals,
                non_seg_vertices, non_seg_faces, non_seg_face_normals
            )
            
            results[color] = consistency
            
            print(f"{part_name}: {len(part_normals)} faces, "
                  f"Normal consistency: {consistency:.4f} (1.0 is perfect)")
    
    # Calculate overall normal consistency
    overall_consistency = compute_normal_consistency(
        seg_vertices, seg_faces, seg_face_normals,
        non_seg_vertices, non_seg_faces, non_seg_face_normals
    )
    results['overall'] = overall_consistency
    
    print(f"Overall shape normal consistency: {overall_consistency:.4f} (1.0 is perfect)")
    
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
    
    # Calculate overall statistics
    overall_consistencies = [results.get('overall', float('nan')) for results in all_results.values()]
    valid_overall = [c for c in overall_consistencies if not np.isnan(c)]
    if valid_overall:
        avg_overall = np.mean(valid_overall)
        std_overall = np.std(valid_overall)
        print(f"Overall normal consistency: mean={avg_overall:.4f}, std={std_overall:.4f} (1.0 is perfect)")
    
    # Aggregate results by body part
    print("\n----- BODY PART STATISTICS -----")
    body_part_results = {}
    for color, part_name in BODY_PART_LABELS.items():
        consistencies = [results.get(color, float('nan')) for results in all_results.values()]
        valid_consistencies = [c for c in consistencies if not np.isnan(c)]
        
        if valid_consistencies:
            avg_consistency = np.mean(valid_consistencies)
            std_consistency = np.std(valid_consistencies)
            body_part_results[part_name] = avg_consistency
            color_name = COLOR_NAMES.get(color, "Unknown")
            print(f"{part_name} ({color_name}): mean={avg_consistency:.4f}, std={std_consistency:.4f}")

    # Find models with min/max normal consistency
    print("\n----- MIN/MAX VALUES -----")
    
    # Overall min/max
    min_overall = {'value': float('inf'), 'file': None}
    max_overall = {'value': float('-inf'), 'file': None}
    
    for basename, results in all_results.items():
        overall_value = results.get('overall', float('nan'))
        if not (np.isnan(overall_value) or overall_value == float('inf') or overall_value == float('-inf')):
            if overall_value < min_overall['value']:
                min_overall['value'] = overall_value
                min_overall['file'] = basename
            if overall_value > max_overall['value']:
                max_overall['value'] = overall_value
                max_overall['file'] = basename
    
    if min_overall['file'] is not None:
        print(f"Lowest overall normal consistency: {min_overall['value']:.4f} (Model: {min_overall['file']})")
    if max_overall['file'] is not None:
        print(f"Highest overall normal consistency: {max_overall['value']:.4f} (Model: {max_overall['file']})")
    
    # Body part min/max
    for color, part_name in BODY_PART_LABELS.items():
        min_part = {'value': float('inf'), 'file': None}
        max_part = {'value': float('-inf'), 'file': None}
        
        for basename, results in all_results.items():
            part_value = results.get(color, float('nan'))
            if not (np.isnan(part_value) or part_value == float('inf') or part_value == float('-inf')):
                if part_value < min_part['value']:
                    min_part['value'] = part_value
                    min_part['file'] = basename
                if part_value > max_part['value']:
                    max_part['value'] = part_value
                    max_part['file'] = basename
        
        if min_part['file'] is not None and max_part['file'] is not None:
            color_name = COLOR_NAMES.get(color, "Unknown")
            print(f"{part_name} ({color_name}):")
            print(f"  Lowest normal consistency: {min_part['value']:.4f} (Model: {min_part['file']})")
            print(f"  Highest normal consistency: {max_part['value']:.4f} (Model: {max_part['file']})")

if __name__ == "__main__":
    main()