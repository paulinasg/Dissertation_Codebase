import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from scipy.spatial import distance
import trimesh
import os
import glob
import argparse

def load_obj_with_colors_and_normals(file_path):
    """Load OBJ file with vertex colors and compute normals."""
    vertices = []
    colors = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
                if len(parts) > 6:
                    color = list(map(float, parts[4:7]))
                    colors.append(color)
                else:
                    colors.append([1.0, 1.0, 1.0])
            elif line.startswith('f '):
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    
    vertices_array = np.array(vertices)
    colors_array = np.array(colors)
    faces_array = np.array(faces)
    
    # Create trimesh object to compute normals
    mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
    
    # Initialize normals array with same size as vertices
    vertex_normals = np.zeros_like(vertices_array)
    
    # Calculate face normals and average for vertices
    face_normals = mesh.face_normals
    vertex_face_count = np.zeros(len(vertices_array))
    
    # Sum up face normals for each vertex
    for face_idx, face in enumerate(faces_array):
        face_normal = face_normals[face_idx]
        for vertex_idx in face:
            vertex_normals[vertex_idx] += face_normal
            vertex_face_count[vertex_idx] += 1
    
    # Average and normalize
    for i in range(len(vertex_normals)):
        if vertex_face_count[i] > 0:
            vertex_normals[i] /= vertex_face_count[i]
            norm = np.linalg.norm(vertex_normals[i])
            if norm > 0:
                vertex_normals[i] /= norm
    
    print(f"Loaded mesh with {len(vertices_array)} vertices, {len(faces_array)} faces")
    print(f"Generated {len(vertex_normals)} vertex normals")
    
    return vertices_array, colors_array, faces_array, vertex_normals

def save_colored_obj(file_path, vertices, colors, faces):
    """Save OBJ file with vertex colors."""
    with open(file_path, 'w') as f:
        for v, c in zip(vertices, colors):
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n')
        for face in faces:
            face_str = ' '.join(str(idx + 1) for idx in face)
            f.write(f'f {face_str}\n')

def get_extremity_weights(vertices):
    """Calculate weights for extremities based on distance from center."""
    # Get bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    height = max_coords[1] - min_coords[1]
    
    # Calculate distances from center
    dists = np.linalg.norm(vertices - center, axis=1)
    max_dist = np.max(dists)
    
    # Create weights
    weights = dists / max_dist
    
    # Additional weight for vertical extremities
    vert_dist = np.abs(vertices[:, 1] - center[1])
    vert_weights = vert_dist / height
    
    # Combine weights
    combined_weights = (weights + vert_weights) / 2
    
    # Enhance extremity weights
    enhanced_weights = np.power(combined_weights, 2)  # Square to enhance difference
    
    return enhanced_weights

def transfer_colors_enhanced(source_vertices, source_colors, source_normals, source_faces,
                           target_vertices, target_normals, target_faces, k_base=8, k_max=32):
    """Transfer colors using position and normal information with adaptive k values."""
    print("Transferring colors with enhanced matching...")
    
    # Verify dimensions match before proceeding
    print(f"Source shapes - Vertices: {source_vertices.shape}, Normals: {source_normals.shape}")
    print(f"Target shapes - Vertices: {target_vertices.shape}, Normals: {target_normals.shape}")
    
    # Regenerate normals but validate they match before using
    print("Regenerating normals for both meshes...")
    source_mesh = trimesh.Trimesh(vertices=source_vertices, faces=source_faces)
    regenerated_source_normals = source_mesh.vertex_normals
    
    target_mesh = trimesh.Trimesh(vertices=target_vertices, faces=target_faces)
    regenerated_target_normals = target_mesh.vertex_normals
    
    # Only use regenerated normals if they match the vertex count
    if len(regenerated_source_normals) == len(source_vertices):
        source_normals = regenerated_source_normals
    else:
        print(f"Warning: Regenerated source normals count ({len(regenerated_source_normals)}) doesn't match vertices count ({len(source_vertices)}). Using original normals.")
    
    if len(regenerated_target_normals) == len(target_vertices):
        target_normals = regenerated_target_normals
    else:
        print(f"Warning: Regenerated target normals count ({len(regenerated_target_normals)}) doesn't match vertices count ({len(target_vertices)}). Using original normals.")
    
    print(f"After validation - Source normals: {source_normals.shape}, Target normals: {target_normals.shape}")
    
    # Check for potential flipped normals
    # Sample some points and check normal alignment
    sample_size = min(1000, len(source_vertices))
    sample_indices = np.random.choice(len(source_vertices), sample_size, replace=False)
    source_sample_normals = source_normals[sample_indices]
    target_sample_normals = np.zeros_like(source_sample_normals)
    
    # Find closest target vertices to sample source vertices
    tree_pos_only = cKDTree(target_vertices)
    _, closest_indices = tree_pos_only.query(source_vertices[sample_indices])
    target_sample_normals = target_normals[closest_indices]
    
    # Calculate average dot product to see if normals are generally aligned
    dot_products = np.sum(source_sample_normals * target_sample_normals, axis=1)
    avg_dot_product = np.mean(dot_products)
    
    # If average dot product is negative, normals are likely flipped
    normals_are_flipped = avg_dot_product < 0
    if normals_are_flipped:
        print("Detected flipped normals! Flipping target normals...")
        target_normals = -target_normals
    
    # Get extremity weights
    extremity_weights = get_extremity_weights(target_vertices)
    
    # Build KD-tree for source vertices
    # Combine position and normal information, with reduced normal influence
    normal_weight = 0.25  # Reduced from 0.5 to make less sensitive to normal direction
    source_features = np.concatenate([
        source_vertices,
        source_normals * normal_weight
    ], axis=1)
    
    target_features = np.concatenate([
        target_vertices,
        target_normals * normal_weight
    ], axis=1)
    
    tree = cKDTree(source_features)
    
    # Initialize output colors
    target_colors = np.zeros((len(target_vertices), 3))
    
    # Process each vertex with adaptive k value
    for i in range(len(target_vertices)):
        # Adaptive k based on extremity weight
        k = int(k_base + (k_max - k_base) * extremity_weights[i])
        k = min(k, len(source_vertices) - 1)  # Make sure k isn't too large
        
        # Query k nearest neighbors
        distances, indices = tree.query(target_features[i], k=k)
        
        # Enhanced inverse distance weighting with safeguards
        weights = 1.0 / (distances**2 + 1e-6)  # Reduced from cubic to squared for smoother falloff
        weights = weights / np.sum(weights)
        
        # Additional normal-based weighting using absolute value to handle flipped normals
        normal_dots = np.abs(np.sum(target_normals[i].reshape(1, 3) * source_normals[indices], axis=1))
        
        # Apply normal weighting with a safety floor to prevent zero weights
        weights = weights * (normal_dots + 0.2)  # Add 0.2 to prevent zero weights
        weights = weights / (np.sum(weights) or 1.0)  # Avoid division by zero
        
        # Compute weighted color
        target_colors[i] = np.average(source_colors[indices], weights=weights, axis=0)
    
    return target_colors

def find_dominant_colors(colors, n_clusters=6):
    """Find dominant colors in the mesh using KMeans clustering."""
    print("Finding dominant colors...")
    
    # Reshape colors to 2D array if needed
    colors_2d = colors.reshape(-1, 3)
    
    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(colors_2d)
    
    # Get the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_
    
    # Get labels for each color
    labels = kmeans.labels_
    
    return dominant_colors, labels

def clean_colors(vertices, faces, colors, threshold=0.1):
    """Clean up colors by enforcing consistency within connected regions."""
    dominant_colors, _ = find_dominant_colors(colors)
    
    # Build adjacency map
    adjacency = {i: set() for i in range(len(vertices))}
    for face in faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    # Clean up colors
    cleaned_colors = colors.copy()
    iterations = 3  # Increased iterations
    
    for iteration in range(iterations):
        for vertex in range(len(vertices)):
            neighbor_colors = [cleaned_colors[n] for n in adjacency[vertex]]
            if not neighbor_colors:
                continue
            
            current_color = cleaned_colors[vertex]
            avg_neighbor_color = np.mean(neighbor_colors, axis=0)
            
            if distance.euclidean(current_color, avg_neighbor_color) > threshold:
                cleaned_colors[vertex] = avg_neighbor_color
    
    return cleaned_colors

def smooth_colors(vertices, faces, colors, iterations=2):
    """Smooth colors across mesh with edge preservation."""
    adjacency = {i: set() for i in range(len(vertices))}
    for face in faces:
        for v1 in face:
            for v2 in face:
                if v1 != v2:
                    adjacency[v1].add(v2)
    
    smoothed_colors = colors.copy()
    for _ in range(iterations):
        new_colors = smoothed_colors.copy()
        for v in range(len(vertices)):
            neighbor_colors = np.array([smoothed_colors[n] for n in adjacency[v]])
            if len(neighbor_colors) > 0:
                # Edge-preserving smoothing
                color_diffs = np.linalg.norm(neighbor_colors - smoothed_colors[v], axis=1)
                weights = np.exp(-color_diffs * 2.5)  # Gaussian weighting
                weights = weights / np.sum(weights)
                new_colors[v] = np.average(neighbor_colors, weights=weights, axis=0)
        
        smoothed_colors = new_colors
    
    return smoothed_colors

def process_meshes(source_obj_path, target_obj_path, output_path):
    """Main function to process and transfer colors between meshes."""
    
    # Load meshes with normals
    print("\nLoading source mesh...")
    source_vertices, source_colors, source_faces, source_normals = load_obj_with_colors_and_normals(source_obj_path)
    
    print("\nLoading target mesh...")
    target_vertices, _, target_faces, target_normals = load_obj_with_colors_and_normals(target_obj_path)
    
    # Transfer colors with enhanced matching
    print("\nTransferring colors...")
    target_colors = transfer_colors_enhanced(
        source_vertices, source_colors, source_normals, source_faces,
        target_vertices, target_normals, target_faces,
        k_base=12,    # Increased base k
        k_max=48      # Much larger k for extremities
    )
    
    # Clean up colors
    print("\nCleaning up colors...")
    cleaned_colors = clean_colors(target_vertices, target_faces, target_colors, threshold=0.15)
    
    # Smooth colors
    print("\nSmoothing colors...")
    final_colors = smooth_colors(target_vertices, target_faces, cleaned_colors, iterations=3)
    
    # Save result
    print("\nSaving colored mesh...")
    save_colored_obj(output_path, target_vertices, final_colors, target_faces)
    
    print("\nProcess completed!")

def process_directory_pairs(source_dir, target_dir, output_dir):
    """Process pairs of files from source and target directories."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of source and target files
    source_files = sorted(glob.glob(os.path.join(source_dir, "*.obj")))
    target_files = sorted(glob.glob(os.path.join(target_dir, "*.obj")))
    
    print(f"Found {len(source_files)} source files and {len(target_files)} target files")
    
    # Process each pair
    for source_file in source_files:
        source_basename = os.path.basename(source_file)
        source_id = os.path.splitext(source_basename)[0]
        
        # Find target file with matching ID
        matching_target = None
        for target_file in target_files:
            target_basename = os.path.basename(target_file)
            target_id = os.path.splitext(target_basename)[0]
            
            if source_id == target_id:
                matching_target = target_file
                break
        
        # If matching target found, process the pair
        if matching_target:
            output_file = os.path.join(output_dir, f"{source_id}.obj")
            print(f"\n\nProcessing pair: {source_id}")
            print(f"Source: {source_file}")
            print(f"Target: {matching_target}")
            print(f"Output: {output_file}")
            
            try:
                process_meshes(source_file, matching_target, output_file)
            except Exception as e:
                print(f"Error processing {source_id}: {e}")
        else:
            print(f"\nNo matching target found for source: {source_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer colors between pairs of meshes")
    parser.add_argument("--source_dir", default="/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/inner",
                        help="Directory containing source (colored) mesh files")
    parser.add_argument("--target_dir", default="/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_aligned_files/inner/bonus",
                        help="Directory containing target mesh files to be colored")
    parser.add_argument("--output_dir", default="/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_labelled_files/inner",
                        help="Directory where colored output files will be saved")
    
    args = parser.parse_args()
    
    process_directory_pairs(args.source_dir, args.target_dir, args.output_dir)