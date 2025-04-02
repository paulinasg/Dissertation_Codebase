import pickle
import networkx as nx
import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN, KMeans
import sys

def check_and_fix_orange_components(obj_file_path, output_file_path=None, connectivity_threshold=5, bridge_ratio=0.001, 
                                   use_clustering=True, cluster_eps=0.1, min_cluster_size=100, 
                                   aggressive_mode=False, force_split=False):
    """
    Check if orange vertices form a single connected component and fix if needed.
    
    Args:
        obj_file_path: Path to the colored OBJ file
        output_file_path: Path to save the fixed OBJ file (defaults to original with _fixed suffix)
        connectivity_threshold: Minimum number of connections between components to consider them truly connected
        bridge_ratio: Maximum ratio of bridge vertices to total orange vertices to consider components connected
        use_clustering: Whether to apply spatial clustering to separate components
        cluster_eps: Maximum distance between points in same cluster for DBSCAN
        min_cluster_size: Minimum number of points to form a cluster for DBSCAN
        aggressive_mode: Whether to use more aggressive techniques for difficult models
        force_split: Whether to force a split even if all other methods fail
    """
    if output_file_path is None:
        output_file_path = obj_file_path.replace(".obj", "_fixed.obj")
    
    print(f"\nProcessing: {obj_file_path}")
    
    # Read the OBJ file
    with open(obj_file_path, "r") as obj_file:
        lines = obj_file.readlines()
    
    # Extract vertices with their colors
    vertices = []
    orange_indices = []
    vertex_colors = []
    
    vertex_index = 0
    for line in lines:
        if line.startswith("v "):  # Vertex line
            parts = line.strip().split()
            if len(parts) >= 7:  # Vertex with color
                x, y, z = map(float, parts[1:4])
                r, g, b = map(float, parts[4:7])
                
                vertices.append((x, y, z))
                vertex_colors.append((r, g, b))
                
                # Check if this is an orange vertex (RGB: 1.0, 0.5, 0.0)
                if abs(r - 1.0) < 0.01 and abs(g - 0.5) < 0.01 and abs(b) < 0.01:
                    orange_indices.append(vertex_index)
                
                vertex_index += 1
    
    print(f"Found {len(orange_indices)} orange vertices.")
    
    # If no orange vertices found, return
    if not orange_indices:
        print("No orange vertices found. No changes needed.")
        return
    
    # Convert to numpy array for faster operations
    vertices_np = np.array(vertices)
    orange_vertices = vertices_np[orange_indices]
    
    # Extract faces that connect orange vertices
    orange_faces = []
    orange_set = set(orange_indices)
    
    for line in lines:
        if line.startswith("f "):  # Face line
            parts = line.strip().split()
            face_vertex_indices = []
            
            # Parse each vertex reference in the face
            for part in parts[1:]:
                # Handle formats like "v", "v/vt", "v/vt/vn", "v//vn"
                vertex_idx = int(part.split('/')[0]) - 1  # OBJ indices start at 1
                face_vertex_indices.append(vertex_idx)
            
            # Check if all vertices in this face are orange
            if all(idx in orange_set for idx in face_vertex_indices):
                orange_faces.append(face_vertex_indices)
    
    print(f"Found {len(orange_faces)} faces connecting orange vertices.")
    
    # Try face-based connectivity first (using only orange faces)
    G_face = nx.Graph()
    G_face.add_nodes_from(orange_indices)
    
    # Add edges based on faces
    for face in orange_faces:
        for i in range(len(face)):
            for j in range(i+1, len(face)):
                G_face.add_edge(face[i], face[j])
    
    # Find connected components using face connectivity
    connected_components_face = list(nx.connected_components(G_face))
    print(f"Found {len(connected_components_face)} connected orange components using face-based connectivity.")
    
    # If face-based connectivity found multiple components, use those
    if len(connected_components_face) > 1:
        # Filter out very small components (likely noise)
        filtered_components = []
        min_component_size = 50  # Minimum size for a valid component
        
        for comp in connected_components_face:
            if len(comp) >= min_component_size:
                filtered_components.append(comp)
        
        # Only use face-based connectivity if we still have multiple significant components
        if len(filtered_components) > 1:
            connected_components = filtered_components
            print(f"Using face-based connectivity for component separation (kept {len(filtered_components)} components of sufficient size)")
        else:
            print("Face-based connectivity found only small disconnected components, treating as a single component")
            connected_components = None
    
    # Otherwise, try clustering
    elif use_clustering and len(orange_indices) > min_cluster_size:
        print("Using spatial clustering...")
        
        # Try multiple clustering approaches
        connected_components = None
        
        # 1. Try DBSCAN with conservative eps values
        eps_values = [0.1, 0.15] if aggressive_mode else [0.15, 0.2]  # More conservative valuesre conservative values
        
        for eps in eps_values:
            clustering = DBSCAN(eps=eps, min_samples=30, n_jobs=-1).fit(orange_vertices)  # Increase min_samples  # Increase min_samples
            labels = clustering.labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            print(f"DBSCAN with eps={eps} found {num_clusters} clusters and {np.sum(labels == -1)} noise points")
            
            if num_clusters > 1:
                connected_components = []
                for i in range(num_clusters):
                    cluster_indices = np.where(labels == i)[0]
                    component = set(orange_indices[idx] for idx in cluster_indices)
                    connected_components.append(component)
                
                # Handle noise points
                noise_indices = np.where(labels == -1)[0]
                if len(noise_indices) > 0:
                    for idx in noise_indices:
                        v_pos = orange_vertices[idx]
                        
                        # Find nearest cluster
                        min_dist = float('inf')
                        nearest_cluster = 0
                        
                        for i, comp in enumerate(connected_components):
                            comp_vertices = vertices_np[[v for v in comp]]
                            distances = np.sum((comp_vertices - v_pos)**2, axis=1)
                            min_comp_dist = np.min(distances) if len(distances) > 0 else float('inf')
                            
                            if min_comp_dist < min_dist:
                                min_dist = min_comp_dist
                                nearest_cluster = i
                        
                        connected_components[nearest_cluster].add(orange_indices[idx])
                
                print(f"Using DBSCAN clustering with eps={eps}")
                break
        
        # 2. Try K-means clustering if DBSCAN failed and aggressive mode is on
        if connected_components is None and aggressive_mode:
            print("DBSCAN clustering failed. Trying K-means clustering...")
            
            # Try different K values
            for k in [2, 3]:
                kmeans = KMeans(n_clusters=k, random_state=42).fit(orange_vertices)
                labels = kmeans.labels_
                
                connected_components = []
                for i in range(k):
                    cluster_indices = np.where(labels == i)[0]
                    component = set(orange_indices[idx] for idx in cluster_indices)
                    connected_components.append(component)
                
                print(f"K-means with k={k} created {len(connected_components)} clusters")
                break
    
    # If clustering methods failed, try graph-based approach
    if connected_components is None or len(connected_components) <= 1:
        print("Clustering failed. Trying graph-based connectivity...")
        
        # Use more conservative threshold
        G = nx.Graph()
        G.add_nodes_from(orange_indices)
        
        threshold = 0.01 if aggressive_mode else 0.02  # More conservative threshold
        
        # Add edges between nearby orange vertices
        for i, idx_i in enumerate(orange_indices):
            v_i = vertices_np[idx_i]
            
            for j, idx_j in enumerate(orange_indices[i+1:], i+1):
                v_j = vertices_np[idx_j]
                
                # Only connect if close
                if np.sum((v_i - v_j) ** 2) < threshold:
                    G.add_edge(idx_i, idx_j)
        
        # Find connected components
        connected_components = list(nx.connected_components(G))
        print(f"Graph-based connectivity found {len(connected_components)} components")
    
    # If all else fails and force_split is enabled, split by position
    if (connected_components is None or len(connected_components) <= 1) and force_split:
        print("All detection methods failed. Forcing split by position...")
        
        # Get orange vertex coordinates
        orange_coords = vertices_np[orange_indices]
        
        # Force split along Y-axis (height)
        y_sorted = np.argsort(orange_coords[:, 1])
        split_point = len(y_sorted) // 2
        
        lower_indices = [orange_indices[y_sorted[i]] for i in range(split_point)]
        upper_indices = [orange_indices[y_sorted[i]] for i in range(split_point, len(y_sorted))]
        
        connected_components = [set(lower_indices), set(upper_indices)]
        print(f"Forced split created {len(connected_components)} components")
    
    if connected_components is None or len(connected_components) <= 1:
        print("All attempts to separate orange vertices failed. No fix needed.")
        return
    
    # Calculate component metrics for analysis
    component_metrics = []
    for comp in connected_components:
        comp_vertices = vertices_np[list(comp)]
        avg_x = np.mean(comp_vertices[:, 0])
        avg_y = np.mean(comp_vertices[:, 1])
        avg_z = np.mean(comp_vertices[:, 2])
        component_metrics.append({
            'component': comp,
            'size': len(comp),
            'avg_x': avg_x,
            'avg_y': avg_y,
            'avg_z': avg_z,
            'centroid': (avg_x, avg_y, avg_z)
        })
    
    # Sort components by y-coordinate (highest first)
    sorted_components = sorted(component_metrics, key=lambda x: x['avg_y'], reverse=True)
    
    print("\nOrange component details (sorted by height):")
    for i, comp in enumerate(sorted_components):
        position = "top" if i == 0 else ("middle" if i in [1, 2] else "bottom")
        print(f"Component {i+1}: {comp['size']} vertices, y={comp['avg_y']:.3f}, position={position}")
    
    # Color palette
    colors = {
        'orange': (1.0, 0.5, 0.0),  # For top component
        'red': (1.0, 0.0, 0.0),     # For middle components
        'teal': (0.0, 0.8, 0.8)     # For bottom components
    }
    
    # Assign new colors based on y-position
    new_colors = {}
    for i, comp_data in enumerate(sorted_components):
        if i == 0:  # Top component (highest y)
            color = colors['orange']
            color_name = 'orange'
        elif i in [1, 2]:  # Middle components
            color = colors['red'] 
            color_name = 'red'
        else:  # Bottom components
            color = colors['teal']
            color_name = 'teal'
        
        print(f"Coloring component {i+1} as {color_name}")
        for vertex_idx in comp_data['component']:
            new_colors[vertex_idx] = color
    
    # Check if there are existing red vertices and change them to teal if needed
    if len(sorted_components) > 1:  # Only if we have multiple components
        # Count existing red vertices
        existing_red_count = 0
        for i, (r, g, b) in enumerate(vertex_colors):
            if i not in new_colors and abs(r - 1.0) < 0.01 and abs(g) < 0.01 and abs(b) < 0.01:
                existing_red_count += 1
        
        # If there are existing red vertices, recolor them to teal
        if existing_red_count > 0:
            print(f"Found {existing_red_count} existing red vertices that will be changed to teal")
            for i, (r, g, b) in enumerate(vertex_colors):
                if i not in new_colors and abs(r - 1.0) < 0.01 and abs(g) < 0.01 and abs(b) < 0.01:
                    new_colors[i] = colors['teal']  # Change to teal
    
    # Write the fixed OBJ file
    vertex_index = 0
    with open(output_file_path, "w") as output_file:
        for line in lines:
            if line.startswith("v ") and len(line.strip().split()) >= 7:  # Vertex line with color
                parts = line.strip().split()
                x, y, z = parts[1:4]
                
                # Check if this vertex needs recoloring
                if vertex_index in new_colors:
                    r, g, b = new_colors[vertex_index]
                else:
                    # Keep original color
                    r, g, b = vertex_colors[vertex_index]
                
                output_file.write(f"v {x} {y} {z} {r} {g} {b}\n")
                vertex_index += 1
            else:
                # Keep other lines unchanged
                output_file.write(line)
    
    print(f"\nFixed OBJ file saved to: {output_file_path}")

def process_files(directory_pattern):
    """Process multiple files matching the given pattern"""
    files = glob.glob(directory_pattern)
    print(f"Found {len(files)} files to process")
    
    for file_path in files:
        check_and_fix_orange_components(file_path)

if __name__ == "__main__":
    # Example usage - process a single file
    obj_file_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/inner/00140_fixed_red2teal.obj"
    check_and_fix_orange_components(obj_file_path, 
                                   connectivity_threshold=30,         # Higher threshold - need more bridges to split
                                   bridge_ratio=0.005,               # Much higher bridge ratio - more conservative
                                   use_clustering=True,
                                   cluster_eps=0.15,                 # Larger eps - more inclusive clusters
                                   min_cluster_size=150,             # Larger minimum cluster size
                                   aggressive_mode=False,            # Disable aggressive mode
                                   force_split=False)                # Don't force split if methods fail