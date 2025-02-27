import numpy as np
from scipy.spatial import cKDTree
import numpy.linalg as LA
from sklearn.decomposition import PCA
import trimesh

def load_obj_with_colors(file_path):
    vertices = []
    colors = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Get vertex coordinates
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
                # Get colors if present (assumes RGB values after XYZ)
                if len(parts) > 6:
                    color = list(map(float, parts[4:7]))
                    colors.append(color)
                else:
                    colors.append([1.0, 1.0, 1.0])  # Default white
            elif line.startswith('f '):
                # Handle faces (assuming simple format f v1 v2 v3)
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    
    print(f"Loaded mesh with {len(vertices)} vertices, {len(faces)} faces")
    return np.array(vertices), np.array(colors), np.array(faces)

def save_colored_obj(file_path, vertices, colors, faces):
    with open(file_path, 'w') as f:
        # Write vertices with colors
        for v, c in zip(vertices, colors):
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n')
        
        # Write faces
        for face in faces:
            face_str = ' '.join(str(idx + 1) for idx in face)
            f.write(f'f {face_str}\n')
    
    print(f"Saved mesh with {len(vertices)} vertices, {len(faces)} faces")

def normalize_mesh(vertices):
    """Normalize mesh to unit cube while preserving aspect ratio"""
    # Get bounding box
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    
    # Get scale factor
    scale = np.max(max_bounds - min_bounds)
    
    # Center and scale
    centered = vertices - np.mean(vertices, axis=0)
    normalized = centered / scale
    
    return normalized

def align_meshes(source_vertices, target_vertices):
    """Align target mesh to source mesh using PCA and centroid alignment with validation"""
    print("Starting mesh alignment...")
    
    # Normalize both meshes first
    source_norm = normalize_mesh(source_vertices)
    target_norm = normalize_mesh(target_vertices)
    
    # Store original centers and scales
    source_center = np.mean(source_vertices, axis=0)
    source_scale = np.max(np.max(source_vertices, axis=0) - np.min(source_vertices, axis=0))
    target_center = np.mean(target_vertices, axis=0)
    
    # Compute PCA
    pca_source = PCA(n_components=3)
    pca_target = PCA(n_components=3)
    
    pca_source.fit(source_norm)
    pca_target.fit(target_norm)
    
    # Check principal components explanation ratio to ensure meaningful alignment
    print("PCA explained variance ratios:")
    print("Source:", pca_source.explained_variance_ratio_)
    print("Target:", pca_target.explained_variance_ratio_)
    
    # Compute rotation matrix
    rotation_matrix = np.dot(pca_target.components_.T, pca_source.components_)
    
    # Ensure proper rotation matrix (right-handed coordinate system)
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, -1] *= -1
    
    # Apply transformation with original scale
    aligned = np.dot(target_norm, rotation_matrix)
    aligned = aligned * source_scale + source_center
    
    # Verify alignment
    print("Alignment stats:")
    print("Source bounds:", np.min(source_vertices, axis=0), np.max(source_vertices, axis=0))
    print("Aligned bounds:", np.min(aligned, axis=0), np.max(aligned, axis=0))
    
    return aligned

def transfer_colors(source_vertices, source_colors, target_vertices, k=8):
    """Transfer colors from source to target using k-nearest neighbors with validation"""
    print("Transferring colors...")
    
    # Build KD-tree for source vertices
    tree = cKDTree(source_vertices)
    
    # Find k nearest neighbors for each target vertex
    distances, indices = tree.query(target_vertices, k=k)
    
    # Weight by inverse squared distance
    weights = 1.0 / (distances**2 + 1e-6)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    # Compute weighted average of colors
    target_colors = np.zeros((len(target_vertices), 3))
    for i in range(len(target_vertices)):
        target_colors[i] = np.average(source_colors[indices[i]], weights=weights[i], axis=0)
    
    # Validate color transfer
    print("Color transfer stats:")
    print("Min color values:", np.min(target_colors, axis=0))
    print("Max color values:", np.max(target_colors, axis=0))
    
    return target_colors

def smooth_colors(vertices, faces, colors, iterations=1):
    """Smooth colors across mesh with reduced iterations"""
    print("Smoothing colors...")
    
    # Build adjacency list
    adjacency = {i: set() for i in range(len(vertices))}
    for face in faces:
        for v1 in face:
            for v2 in face:
                if v1 != v2:
                    adjacency[v1].add(v2)
    
    smoothed_colors = colors.copy()
    for i in range(iterations):
        new_colors = smoothed_colors.copy()
        for v in range(len(vertices)):
            neighbor_colors = np.array([smoothed_colors[n] for n in adjacency[v]])
            if len(neighbor_colors) > 0:
                new_colors[v] = np.mean(neighbor_colors, axis=0) * 0.5 + smoothed_colors[v] * 0.5
        smoothed_colors = new_colors
        
        # Print progress
        if i % 5 == 0:
            print(f"Smoothing iteration {i+1}/{iterations}")
    
    return smoothed_colors

def process_meshes(source_obj_path, target_obj_path, output_path):
    """Main function to process and transfer colors between meshes"""
    
    # Load meshes
    print("\nLoading source mesh...")
    source_vertices, source_colors, source_faces = load_obj_with_colors(source_obj_path)
    
    print("\nLoading target mesh...")
    target_vertices, _, target_faces = load_obj_with_colors(target_obj_path)
    
    # Align meshes
    print("\nAligning meshes...")
    aligned_target_vertices = align_meshes(source_vertices, target_vertices)
    
    # Transfer colors
    print("\nTransferring colors...")
    target_colors = transfer_colors(source_vertices, source_colors, aligned_target_vertices, k=8)
    
    # Smooth colors (reduced iterations)
    print("\nSmoothing colors...")
    smoothed_colors = smooth_colors(aligned_target_vertices, target_faces, target_colors, iterations=1)
    
    # Save result
    print("\nSaving colored mesh...")
    save_colored_obj(output_path, aligned_target_vertices, smoothed_colors, target_faces)
    
    print("\nProcess completed!")

# Usage
if __name__ == "__main__":
    source_obj = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/code/4d_label/00122.obj"  # Your colored mesh
    target_obj = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/code/align/aligned_result_man.obj"  # Mesh to be colored
    output_obj = "00122_test.obj"  # Output path
    
    process_meshes(source_obj, target_obj, output_obj)