import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from scipy.spatial import distance
import trimesh

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
    
    # Create trimesh object to compute normals
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    vertex_normals = mesh.vertex_normals
    
    print(f"Loaded mesh with {len(vertices)} vertices, {len(faces)} faces")
    return np.array(vertices), np.array(colors), np.array(faces), vertex_normals

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

def transfer_colors_enhanced(source_vertices, source_colors, source_normals, 
                           target_vertices, target_normals, k_base=8, k_max=32):
    """Transfer colors using position and normal information with adaptive k values."""
    print("Transferring colors with enhanced matching...")
    
    # Get extremity weights
    extremity_weights = get_extremity_weights(target_vertices)
    
    # Build KD-tree for source vertices
    # Combine position and normal information
    source_features = np.concatenate([
        source_vertices,
        source_normals * 0.5  # Scale normals to balance their influence
    ], axis=1)
    
    target_features = np.concatenate([
        target_vertices,
        target_normals * 0.5
    ], axis=1)
    
    tree = cKDTree(source_features)
    
    # Initialize output colors
    target_colors = np.zeros((len(target_vertices), 3))
    
    # Process each vertex with adaptive k value
    for i in range(len(target_vertices)):
        # Adaptive k based on extremity weight
        k = int(k_base + (k_max - k_base) * extremity_weights[i])
        
        # Query k nearest neighbors
        distances, indices = tree.query(target_features[i], k=k)
        
        # Enhanced inverse distance weighting
        weights = 1.0 / (distances**3 + 1e-6)  # Cubic falloff for sharper transitions
        weights = weights / np.sum(weights)
        
        # Additional normal-based weighting
        normal_dots = np.abs(np.dot(target_normals[i], source_normals[indices]))
        weights *= normal_dots
        weights = weights / np.sum(weights)
        
        # Compute weighted color
        target_colors[i] = np.average(source_colors[indices], weights=weights, axis=0)
    
    return target_colors

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
                weights = np.exp(-color_diffs * 2)  # Gaussian weighting
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
        source_vertices, source_colors, source_normals,
        target_vertices, target_normals,
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



if __name__ == "__main__":
    source_obj = "/Users/paulinagerchuk/Downloads/Outer/Take9/Code/smpl_segmentation/colored_model_woman.obj"  # Your colored mesh
    target_obj = "/Users/paulinagerchuk/Downloads/Outer/Take9/Code/aligned_result.obj"          # Mesh to be colored
    output_obj = "woman_colored_pi.obj"  # Output path
    
    process_meshes(
        source_obj_path=source_obj,
        target_obj_path=target_obj,
        output_path=output_obj # Adjust for more/less aggressive color cleanup
    )
