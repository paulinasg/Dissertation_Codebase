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
        k = min(k, len(source_vertices) - 1)  # Make sure k isn't too large
        
        # Query k nearest neighbors
        distances, indices = tree.query(target_features[i], k=k)
        
        # Enhanced inverse distance weighting
        weights = 1.0 / (distances**3 + 1e-6)  # Cubic falloff for sharper transitions
        weights = weights / np.sum(weights)
        
        # Additional normal-based weighting
        # FIX: Replace dot product with element-wise calculation for batch processing
        normal_dots = np.abs(np.sum(target_normals[i].reshape(1, 3) * source_normals[indices], axis=1))
        
        weights *= normal_dots
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
    iterations = 2  # Reduced iterations (was 3)
    
    for iteration in range(iterations):
        for vertex in range(len(vertices)):
            neighbor_colors = [cleaned_colors[n] for n in adjacency[vertex]]
            if not neighbor_colors:
                continue
            
            current_color = cleaned_colors[vertex]
            avg_neighbor_color = np.mean(neighbor_colors, axis=0)
            
            if distance.euclidean(current_color, avg_neighbor_color) > threshold:
                # Only partially move toward average color for sharper boundaries
                cleaned_colors[vertex] = current_color * 0.7 + avg_neighbor_color * 0.3
    
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
                # Edge-preserving smoothing with sharper edges
                color_diffs = np.linalg.norm(neighbor_colors - smoothed_colors[v], axis=1)
                weights = np.exp(-color_diffs * 5.0)  # Increased from 2.5 for sharper transitions
                weights = weights / np.sum(weights)
                new_colors[v] = np.average(neighbor_colors, weights=weights, axis=0)
        
        smoothed_colors = new_colors
    
    return smoothed_colors

def quantize_colors(colors, n_colors=8):
    """Quantize colors to a limited palette."""
    print("Quantizing colors...")
    
    colors_2d = colors.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(colors_2d)
    centers = kmeans.cluster_centers_
    
    quantized = centers[labels].reshape(colors.shape)
    return quantized, labels, centers

def create_segmentation_map(vertices, faces, colors, n_segments=8):
    """Create a segmentation map with regularized boundaries."""
    print("Creating segmentation map...")
    
    # First quantize colors to get the initial segmentation
    quantized_colors, segment_labels, color_centers = quantize_colors(colors, n_colors=n_segments)
    
    # Create mesh connectivity graph
    adjacency = {i: set() for i in range(len(vertices))}
    for face in faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    # Regularize boundaries with graph-based smoothing
    print("Regularizing segment boundaries...")
    final_labels = segment_labels.copy()
    iterations = 5  # Number of iterations for boundary regularization
    
    for iteration in range(iterations):
        print(f"Regularization iteration {iteration+1}/{iterations}")
        changes = 0
        
        for vertex in range(len(vertices)):
            neighbors = list(adjacency[vertex])
            if not neighbors:
                continue
            
            # Count segment labels of neighbors
            neighbor_labels = [final_labels[n] for n in neighbors]
            label_counts = np.bincount(neighbor_labels, minlength=n_segments)
            
            # Find most common label among neighbors
            most_common = np.argmax(label_counts)
            
            # If vertex is surrounded by mostly different labels, change it
            # This creates more regular boundaries
            if final_labels[vertex] != most_common and label_counts[most_common] > len(neighbors) // 2:
                final_labels[vertex] = most_common
                changes += 1
        
        print(f"  Made {changes} changes")
        if changes == 0:
            break
    
    # Convert regularized labels back to colors
    regularized_colors = color_centers[final_labels].copy()
    
    return regularized_colors

def regularize_boundaries(vertices, faces, colors):
    """Create more linear and regular boundaries between color regions."""
    # Step 1: Build mesh connectivity
    adjacency = {i: set() for i in range(len(vertices))}
    for face in faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    # Step 2: Identify boundary vertices
    boundary_vertices = []
    
    # Find dominant colors
    quantized, labels, centers = quantize_colors(colors, n_colors=8)
    
    # Identify vertices at color boundaries
    for vertex in range(len(vertices)):
        neighbors = list(adjacency[vertex])
        if not neighbors:
            continue
        
        # Check if any neighbor has a different label
        my_label = labels[vertex]
        for neighbor in neighbors:
            if labels[neighbor] != my_label:
                boundary_vertices.append(vertex)
                break
    
    # Step 3: Linearize boundaries by aligning to local axes
    print(f"Processing {len(boundary_vertices)} boundary vertices...")
    
    # Create more regular boundary by voting
    regularized_labels = labels.copy()
    
    # Process boundary vertices multiple times to create smoother borders
    for _ in range(3):
        for vertex in boundary_vertices:
            neighbors = list(adjacency[vertex])
            if not neighbors:
                continue
                
            # Get the labels of neighbors
            neighbor_labels = [regularized_labels[n] for n in neighbors]
            
            # Count occurrences of each label
            label_counts = {}
            for label in neighbor_labels:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            
            # Find the two most common labels
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            
            # If there's a clear majority, use that label
            if len(sorted_labels) >= 1 and sorted_labels[0][1] > len(neighbors) // 2:
                regularized_labels[vertex] = sorted_labels[0][0]
    
    # Convert regularized labels back to colors
    regularized_colors = centers[regularized_labels]

    return regularized_colors

def process_meshes(source_obj_path, target_obj_path, output_path):
    """Main function to process and transfer colors between meshes."""
    
    # Load meshes with normals
    print("\nLoading source mesh...")
    source_vertices, source_colors, source_faces, source_normals = load_obj_with_colors_and_normals(source_obj_path)
    
    print("\nLoading target mesh...")
    target_vertices, _, target_faces, target_normals = load_obj_with_colors_and_normals(target_obj_path)
    
    # Transfer colors with enhanced matching - use smaller k values for less blending
    print("\nTransferring colors...")
    target_colors = transfer_colors_enhanced(
        source_vertices, source_colors, source_normals, source_faces,
        target_vertices, target_normals, target_faces,
        k_base=3,     # Significantly reduced for less blending
        k_max=8       # Significantly reduced for less blending
    )
    
    # Create a hard segmentation using strict color clustering
    print("\nSegmenting into distinct regions...")
    # First quantize to get initial segments
    _, initial_labels, color_centers = quantize_colors(target_colors, n_colors=8)  # Reduce number of colors
    
    # Build adjacency map for mesh connectivity
    adjacency = {i: set() for i in range(len(target_vertices))}
    for face in target_faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    # Aggressive boundary regularization - many iterations for straighter lines
    print("\nAggressively straightening boundaries...")
    final_labels = initial_labels.copy()
    iterations = 10  # More iterations for straighter boundaries
    
    for iteration in range(iterations):
        print(f"Straightening iteration {iteration+1}/{iterations}")
        changes = 0
        
        for vertex in range(len(target_vertices)):
            neighbors = list(adjacency[vertex])
            if not neighbors:
                continue
                
            # Strong majority voting - Only keep label if surrounded by same label
            neighbor_labels = [final_labels[n] for n in neighbors]
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            
            # If there's a clear majority (more than 60% neighbors)
            threshold = int(len(neighbors) * 0.6)  # 60% threshold
            most_common_idx = np.argmax(counts)
            most_common = unique_labels[most_common_idx]
            
            if counts[most_common_idx] >= threshold and final_labels[vertex] != most_common:
                final_labels[vertex] = most_common
                changes += 1
        
        print(f"  Made {changes} changes")
        if changes == 0:
            break
    
    # Remove small unconnected components and replace with surrounding colors
    print("\nRemoving small components...")
    cleaned_labels = remove_small_components(target_vertices, target_faces, final_labels, color_centers, min_size_ratio=0.05)
    
    # Convert final labels back to colors
    final_colors = color_centers[cleaned_labels]
    
    # Save result
    print("\nSaving colored mesh...")
    save_colored_obj(output_path, target_vertices, final_colors, target_faces)
    
    print("\nProcess completed!")

# Add this new function to perform tensor voting for straighter boundaries
def straighten_boundaries(vertices, faces, labels, n_segments):
    """Use tensor voting to create straighter segment boundaries."""
    print("Applying tensor voting for straighter boundaries...")
    
    # Build adjacency list
    adjacency = {i: set() for i in range(len(vertices))}
    for face in faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    # Find boundary vertices
    boundaries = []
    for i in range(len(vertices)):
        neighbors = adjacency[i]
        if not neighbors:
            continue
            
        my_label = labels[i]
        for n in neighbors:
            if labels[n] != my_label:
                boundaries.append(i)
                break
    
    print(f"Found {len(boundaries)} boundary vertices")
    
    # Apply tensor voting for each boundary vertex
    # This is a simplified version that tries to create more linear boundaries
    for _ in range(3):  # Multiple passes
        new_labels = labels.copy()
        
        for v_idx in boundaries:
            # Get 2-ring neighborhood
            neighbors1 = list(adjacency[v_idx])
            neighbors2 = []
            for n1 in neighbors1:
                neighbors2.extend(list(adjacency[n1]))
            
            all_neighbors = set(neighbors1 + neighbors2)
            all_neighbors.discard(v_idx)  # Remove self
            
            # If not enough neighbors, skip
            if len(all_neighbors) < 5:
                continue
                
            # Get positions of neighborhood
            neighbor_pos = vertices[list(all_neighbors)]
            
            # Compute principal directions (simplified tensor voting)
            # We'll use PCA to find the principal direction
            neighbor_centered = neighbor_pos - vertices[v_idx]
            cov = neighbor_centered.T @ neighbor_centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # The eigenvector with the largest eigenvalue is the main direction
            principal_direction = eigvecs[:, np.argmax(eigvals)]
            
            # Project neighbors onto principal direction
            projections = neighbor_centered @ principal_direction
            
            # Divide neighbors into two sides based on projection
            pos_side = [n for i, n in enumerate(all_neighbors) if projections[i] > 0]
            neg_side = [n for i, n in enumerate(all_neighbors) if projections[i] <= 0]
            
            # Get most common label for each side
            if pos_side:
                pos_labels = [labels[i] for i in pos_side]
                pos_common = np.argmax(np.bincount(pos_labels))
            else:
                pos_common = labels[v_idx]
                
            if neg_side:
                neg_labels = [labels[i] for i in neg_side]
                neg_common = np.argmax(np.bincount(neg_labels))
            else:
                neg_common = labels[v_idx]
                
            # If the two sides have different common labels, assign based on projection
            if pos_common != neg_common:
                # Keep current label if it matches one of the sides
                if labels[v_idx] != pos_common and labels[v_idx] != neg_common:
                    new_labels[v_idx] = pos_common if len(pos_side) > len(neg_side) else neg_common
                
        labels = new_labels
    
    return labels

def remove_small_components(vertices, faces, labels, color_centers, min_size_ratio=0.05):
    """
    Remove small unconnected components from each segment and replace with surrounding colors.
    For blue and red colors, keep top 2 components.
    """
    print("\nRemoving small unconnected components...")
    
    # Build adjacency map for mesh connectivity
    adjacency = {i: set() for i in range(len(vertices))}
    for face in faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)

    # Function to find connected components within a specific label
    def find_connected_components(label_value):
        # Get all vertices with this label
        vertices_with_label = [v for v in range(len(vertices)) if labels[v] == label_value]
        
        # Find connected components
        components = []
        unvisited = set(vertices_with_label)
        
        while unvisited:
            # Start a new component
            start = next(iter(unvisited))
            component = set()
            to_visit = [start]
            
            # Expand the component using BFS
            while to_visit:
                current = to_visit.pop()
                if current in component:
                    continue
                    
                component.add(current)
                unvisited.discard(current)
                
                # Add neighbors with the same label
                for neighbor in adjacency[current]:
                    if neighbor in unvisited and labels[neighbor] == label_value:
                        to_visit.append(neighbor)
            
            components.append(component)
        
        return components

    # Get unique labels
    unique_labels = np.unique(labels)
    new_labels = labels.copy()
    
    # Process each label
    for label in unique_labels:
        components = find_connected_components(label)
        
        # Get component sizes
        component_sizes = [len(c) for c in components]
        
        if not component_sizes:
            continue
            
        # Sort components by size (largest first)
        sorted_indices = np.argsort(component_sizes)[::-1]
        
        # Calculate the sum of all components' sizes
        total_size = sum(component_sizes)
        
        # For blue and red (identify them by their RGB values), keep top 2 components
        color = color_centers[label]
        
        # Determine if this is red or blue by checking RGB values
        is_red = color[0] > 0.5 and color[1] < 0.3 and color[2] < 0.3
        is_blue = color[2] > 0.5 and color[0] < 0.3 and color[1] < 0.3
        
        # Decide how many components to keep
        if is_red or is_blue:
            keep_count = min(2, len(components))  # Keep top 2 for red and blue
            print(f"Color {color} identified as {'red' if is_red else 'blue'}, keeping top {keep_count} components")
        else:
            keep_count = 1  # Keep only largest for other colors
        
        # Process components to remove (i.e., all except the largest ones to keep)
        for i in range(keep_count, len(components)):
            component = components[sorted_indices[i]]
            component_size = len(component)
            
            # Only remove if it's small enough relative to the largest component
            if component_size / total_size < min_size_ratio:
                print(f"Removing component with {component_size} vertices (ratio: {component_size / total_size:.4f})")
                
                # For each vertex in the small component, replace with surrounding color
                for vertex in component:
                    # Find most common label among neighbors
                    neighbor_labels = [new_labels[n] for n in adjacency[vertex] 
                                      if n not in component]  # Exclude vertices from the same component
                    
                    if neighbor_labels:
                        # Find most common neighboring label that's not the current label
                        label_counts = {}
                        for nl in neighbor_labels:
                            if nl != label:  # Exclude current label
                                label_counts[nl] = label_counts.get(nl, 0) + 1
                                
                        if label_counts:
                            # Replace with most common surrounding label
                            most_common_label = max(label_counts, key=label_counts.get)
                            new_labels[vertex] = most_common_label
    
    return new_labels

if __name__ == "__main__":
    source_obj = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/code/4d_label/00163.obj"  # Your colored mesh
    target_obj = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/code/align/aligned_result_woman.obj"          # Mesh to be colored
    output_obj = "00163_colored_harsh.obj"  # Output path
    
    process_meshes(
        source_obj_path=source_obj,
        target_obj_path=target_obj,
        output_path=output_obj # Adjust for more/less aggressive color cleanup
    )
