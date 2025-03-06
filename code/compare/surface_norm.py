import os
import numpy as np
import trimesh
from tqdm import tqdm

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

def compute_normal_consistency(normals1, normals2):
    """
    Compute normal consistency between two sets of normals
    Returns average cosine similarity (1 is perfect alignment, -1 is opposite)
    """
    if len(normals1) == 0 or len(normals2) == 0:
        return 0.0
    
    # Normalize all normals (ensure unit length)
    n1 = normals1 / np.maximum(np.linalg.norm(normals1, axis=1, keepdims=True), 1e-10)
    n2 = normals2 / np.maximum(np.linalg.norm(normals2, axis=1, keepdims=True), 1e-10)
    
    # For each normal in n1, find closest normal in n2
    closest_dot_products = []
    for normal in n1:
        # Compute dot products between this normal and all normals in n2
        dot_products = np.abs(np.dot(n2, normal))
        closest_dot_products.append(np.max(dot_products))
    
    # Average the cosine similarities
    mean_consistency = np.mean(closest_dot_products)
    return mean_consistency

def process_file_pair(segmented_file, non_segmented_file):
    """
    Process a pair of files and compute normal consistency for each body part
    """
    # Load segmented mesh with face groups
    seg_vertices, seg_faces, seg_face_groups = load_obj_with_face_groups(segmented_file)
    seg_mesh = trimesh.Trimesh(vertices=seg_vertices, faces=seg_faces)
    seg_face_normals = compute_face_normals(seg_vertices, seg_faces)
    
    # Load non-segmented mesh
    non_seg_mesh = trimesh.load(non_segmented_file)
    non_seg_vertices = np.array(non_seg_mesh.vertices)
    non_seg_faces = np.array(non_seg_mesh.faces)
    non_seg_face_normals = compute_face_normals(non_seg_vertices, non_seg_faces)
    non_seg_face_centers = compute_face_centers(non_seg_vertices, non_seg_faces)
    
    # Print only the filename
    print(f"\n{os.path.basename(segmented_file)}")
    print("Normal consistency by body part:")
    
    # Calculate normal consistency for each labeled part
    results = {}
    for color, face_indices in seg_face_groups.items():
        if color in BODY_PART_LABELS:
            part_name = BODY_PART_LABELS[color]
            
            # Get part vertices from faces
            part_vertices = seg_vertices[np.unique(seg_faces[face_indices].flatten())]
            
            # Compute bounding box for this part
            bbox_min, bbox_max = compute_bounding_box(part_vertices)
            
            # Filter non-segmented faces to those within the bounding box
            filtered_face_indices = filter_faces_in_bbox(non_seg_face_centers, bbox_min, bbox_max)
            
            # Skip if no faces in bounding box
            if len(filtered_face_indices) == 0:
                print(f"{part_name}: No corresponding faces found in non-segmented model")
                continue
            
            # Get normals for part and corresponding region
            part_normals = seg_face_normals[face_indices]
            corresponding_normals = non_seg_face_normals[filtered_face_indices]
            
            # Compute normal consistency
            consistency = compute_normal_consistency(part_normals, corresponding_normals)
            results[color] = consistency
            
            print(f"{part_name}: {len(part_normals)} faces, {len(corresponding_normals)} faces in bbox, "
                  f"Normal consistency: {consistency:.4f} (1.0 is perfect)")
    
    # Calculate overall normal consistency
    all_seg_normals = seg_face_normals
    overall_consistency = compute_normal_consistency(all_seg_normals, non_seg_face_normals)
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
    
    # Calculate overall average first
    overall_consistencies = [results.get('overall', float('nan')) for results in all_results.values()]
    valid_overall = [c for c in overall_consistencies if not np.isnan(c)]
    if valid_overall:
        avg_overall = np.mean(valid_overall)
        print(f"Average overall normal consistency: {avg_overall:.4f} (1.0 is perfect)")
    
    # Aggregate results by body part
    body_part_results = {}
    for color, part_name in BODY_PART_LABELS.items():
        consistencies = [results.get(color, float('nan')) for results in all_results.values()]
        valid_consistencies = [c for c in consistencies if not np.isnan(c)]
        
        if valid_consistencies:
            avg_consistency = np.mean(valid_consistencies)
            body_part_results[part_name] = avg_consistency
            color_name = COLOR_NAMES.get(color, "Unknown")
            print(f"Average normal consistency for {part_name} ({color_name}): {avg_consistency:.4f}")

if __name__ == "__main__":
    main()