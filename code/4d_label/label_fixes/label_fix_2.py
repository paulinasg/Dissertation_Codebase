import pickle
import networkx as nx
import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN

def check_and_fix_red_components(obj_file_path, output_file_path=None, connectivity_threshold=5, bridge_ratio=0.001, 
                                use_clustering=True, cluster_eps=0.1, min_cluster_size=100):
    """
    Check if red vertices form a single connected component and fix if needed.
    Colors the highest component yellow.
    
    Args:
        obj_file_path: Path to the colored OBJ file
        output_file_path: Path to save the fixed OBJ file (defaults to original with _fixed suffix)
        connectivity_threshold: Minimum number of connections between components to consider them truly connected
        bridge_ratio: Maximum ratio of bridge vertices to total red vertices to consider components connected
        use_clustering: Whether to apply spatial clustering to separate components
        cluster_eps: Maximum distance between points in same cluster for DBSCAN
        min_cluster_size: Minimum number of points to form a cluster for DBSCAN
    """
    if output_file_path is None:
        output_file_path = obj_file_path.replace(".obj", "_fixed_red.obj")
    
    print(f"\nProcessing: {obj_file_path}")
    
    # Read the OBJ file
    with open(obj_file_path, "r") as obj_file:
        lines = obj_file.readlines()
    
    # Extract vertices with their colors
    vertices = []
    red_indices = []
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
                
                # Check if this is a red vertex (RGB: 1.0, 0.0, 0.0)
                if abs(r - 1.0) < 0.01 and abs(g) < 0.01 and abs(b) < 0.01:
                    red_indices.append(vertex_index)
                
                vertex_index += 1
    
    print(f"Found {len(red_indices)} red vertices.")
    
    # If no red vertices found, return
    if not red_indices:
        print("No red vertices found. No changes needed.")
        return
    
    # Convert to numpy array for faster operations
    vertices_np = np.array(vertices)
    red_vertices = vertices_np[red_indices]
    
    # Extract components using spatial clustering
    if use_clustering and len(red_indices) > min_cluster_size:
        print("Using spatial clustering to identify disconnected components...")
        
        # Use DBSCAN to cluster the red vertices
        clustering = DBSCAN(eps=cluster_eps, min_samples=min_cluster_size//10, n_jobs=-1).fit(red_vertices)
        labels = clustering.labels_
        
        # Get unique cluster labels (-1 is noise)
        unique_clusters = np.unique(labels)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        
        print(f"DBSCAN found {num_clusters} clusters and {np.sum(labels == -1)} noise points")
        
        # Create component sets from clusters
        connected_components = []
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            component = set(red_indices[idx] for idx in cluster_indices)
            connected_components.append(component)
            
        # Add noise points to the nearest cluster
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            v_pos = red_vertices[idx]
            
            # Find the nearest component
            min_dist = float('inf')
            nearest_comp_idx = 0
            
            for i, comp in enumerate(connected_components):
                comp_vertices = vertices_np[[v for v in comp]]
                distances = np.sum((comp_vertices - v_pos)**2, axis=1)
                min_comp_dist = np.min(distances) if len(distances) > 0 else float('inf')
                
                if min_comp_dist < min_dist:
                    min_dist = min_comp_dist
                    nearest_comp_idx = i
            
            # Add the noise point to nearest component
            if connected_components:  # Only if we have components
                connected_components[nearest_comp_idx].add(red_indices[idx])
    else:
        # Use the graph-based approach
        G = nx.Graph()
        G.add_nodes_from(red_indices)
        
        # Add edges between nearby red vertices
        threshold = 0.01  # Distance threshold for connectivity
        
        # Process in chunks to reduce memory usage
        chunk_size = 1000
        for i in range(0, len(red_indices), chunk_size):
            chunk_indices = red_indices[i:i+chunk_size]
            chunk_vertices = vertices_np[chunk_indices]
            
            # Calculate pairwise distances between this chunk and all red vertices
            for j, idx_j in enumerate(red_indices):
                v_j = vertices_np[idx_j]
                # Calculate squared distances
                distances = np.sum((chunk_vertices - v_j) ** 2, axis=1)
                
                # Find close vertices
                close_indices = np.where(distances < threshold)[0]
                
                # Add edges for close vertices
                for k in close_indices:
                    idx_i = red_indices[i + k] if i + k < len(red_indices) else red_indices[k]
                    if idx_i != idx_j:  # Avoid self-loops
                        G.add_edge(idx_i, idx_j)
        
        # Find connected components in the initial graph
        connected_components_initial = list(nx.connected_components(G))
        print(f"Found {len(connected_components_initial)} connected red components in initial graph.")
        
        # If multiple components already, no need for additional checks
        if len(connected_components_initial) > 1:
            connected_components = connected_components_initial
        else:
            # Find articulation points (bridge vertices)
            bridge_vertices = list(nx.articulation_points(G))
            print(f"Found {len(bridge_vertices)} bridge vertices.")
            
            # Calculate bridge ratio
            actual_bridge_ratio = len(bridge_vertices) / len(red_indices)
            print(f"Bridge ratio: {actual_bridge_ratio:.6f} (threshold: {bridge_ratio})")
            
            if len(bridge_vertices) > connectivity_threshold or actual_bridge_ratio > bridge_ratio:
                # Remove bridge vertices to find the true components
                G_no_bridges = G.copy()
                G_no_bridges.remove_nodes_from(bridge_vertices)
                connected_components = list(nx.connected_components(G_no_bridges))
                
                # Add each bridge vertex to the nearest component
                for vertex in bridge_vertices:
                    v_pos = vertices_np[vertex]
                    
                    # Find the nearest component
                    min_dist = float('inf')
                    nearest_comp_idx = 0
                    
                    for i, comp in enumerate(connected_components):
                        comp_vertices = vertices_np[[v for v in comp]]
                        distances = np.sum((comp_vertices - v_pos)**2, axis=1)
                        min_comp_dist = np.min(distances)
                        
                        if min_comp_dist < min_dist:
                            min_dist = min_comp_dist
                            nearest_comp_idx = i
                    
                    # Add the bridge vertex to the nearest component
                    connected_components[nearest_comp_idx].add(vertex)
                
                print(f"After removing bridge vertices: {len(connected_components)} components")
            else:
                # Keep the original components
                connected_components = connected_components_initial
                print("Insufficient bridge vertices. Treating as a single component.")
    
    if len(connected_components) <= 1:
        print("Red vertices still form a single component after analysis. Will try manual separation.")
        
        # Try to separate by coordinate clusters as a last resort
        red_coords = vertices_np[red_indices]
        
        # Try different eps values until we get multiple components
        for eps in [0.05, 0.1, 0.2, 0.5, 1.0]:
            clustering = DBSCAN(eps=eps, min_samples=30).fit(red_coords)
            labels = clustering.labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if num_clusters > 1:
                print(f"Successfully split into {num_clusters} components using DBSCAN with eps={eps}")
                
                connected_components = []
                for i in range(num_clusters):
                    component = set()
                    mask = (labels == i)
                    for idx, is_in_cluster in enumerate(mask):
                        if is_in_cluster:
                            component.add(red_indices[idx])
                    connected_components.append(component)
                
                # Handle noise points (-1 label) by assigning to nearest cluster
                noise_indices = np.where(labels == -1)[0]
                if len(noise_indices) > 0 and len(connected_components) > 0:
                    for idx in noise_indices:
                        v_pos = red_coords[idx]
                        
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
                        
                        connected_components[nearest_cluster].add(red_indices[idx])
                
                break
    
    if len(connected_components) <= 1:
        print("All attempts to separate red vertices failed. No fix needed.")
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
    
    print("\nRed component details (sorted by height):")
    for i, comp in enumerate(sorted_components):
        position = "top" if i == 0 else "lower"
        print(f"Component {i+1}: {comp['size']} vertices, y={comp['avg_y']:.3f}, position={position}")
    
    # Color palette
    colors = {
        'yellow': (1.0, 1.0, 0.0),  # For top component
        'red': (1.0, 0.0, 0.0),     # For remaining components
    }
    
    # Assign new colors - only the top component becomes yellow
    new_colors = {}
    for i, comp_data in enumerate(sorted_components):
        if i == 0:  # Top component (highest y)
            color = colors['yellow']
            color_name = 'yellow'
        else:  # All other components remain red
            color = colors['red'] 
            color_name = 'red'
        
        print(f"Coloring component {i+1} as {color_name}")
        for vertex_idx in comp_data['component']:
            new_colors[vertex_idx] = color
    
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

def check_and_fix_teal_components(obj_file_path, output_file_path=None, connectivity_threshold=5, bridge_ratio=0.001, 
                                use_clustering=True, cluster_eps=0.1, min_cluster_size=100):
    """
    Check if teal vertices form a single connected component and fix if needed.
    Colors the highest component yellow.
    
    Args:
        obj_file_path: Path to the colored OBJ file
        output_file_path: Path to save the fixed OBJ file (defaults to original with _fixed suffix)
        connectivity_threshold: Minimum number of connections between components to consider them truly connected
        bridge_ratio: Maximum ratio of bridge vertices to total teal vertices to consider components connected
        use_clustering: Whether to apply spatial clustering to separate components
        cluster_eps: Maximum distance between points in same cluster for DBSCAN
        min_cluster_size: Minimum number of points to form a cluster for DBSCAN
    """
    if output_file_path is None:
        output_file_path = obj_file_path.replace(".obj", "_fixed_teal.obj")
    
    print(f"\nProcessing: {obj_file_path}")
    
    # Read the OBJ file
    with open(obj_file_path, "r") as obj_file:
        lines = obj_file.readlines()
    
    # Extract vertices with their colors
    vertices = []
    teal_indices = []
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
                
                # Check if this is a teal vertex (RGB: 0.0, 0.8, 0.8)
                if abs(r) < 0.01 and abs(g - 0.8) < 0.01 and abs(b - 0.8) < 0.01:
                    teal_indices.append(vertex_index)
                
                vertex_index += 1
    
    print(f"Found {len(teal_indices)} teal vertices.")
    
    # If no teal vertices found, return
    if not teal_indices:
        print("No teal vertices found. No changes needed.")
        return
    
    # Convert to numpy array for faster operations
    vertices_np = np.array(vertices)
    teal_vertices = vertices_np[teal_indices]
    
    # Extract components using spatial clustering
    if use_clustering and len(teal_indices) > min_cluster_size:
        print("Using spatial clustering to identify disconnected components...")
        
        # Use DBSCAN to cluster the teal vertices
        clustering = DBSCAN(eps=cluster_eps, min_samples=min_cluster_size//10, n_jobs=-1).fit(teal_vertices)
        labels = clustering.labels_
        
        # Get unique cluster labels (-1 is noise)
        unique_clusters = np.unique(labels)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        
        print(f"DBSCAN found {num_clusters} clusters and {np.sum(labels == -1)} noise points")
        
        # Create component sets from clusters
        connected_components = []
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            component = set(teal_indices[idx] for idx in cluster_indices)
            connected_components.append(component)
            
        # Add noise points to the nearest cluster
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            v_pos = teal_vertices[idx]
            
            # Find the nearest component
            min_dist = float('inf')
            nearest_comp_idx = 0
            
            for i, comp in enumerate(connected_components):
                comp_vertices = vertices_np[[v for v in comp]]
                distances = np.sum((comp_vertices - v_pos)**2, axis=1)
                min_comp_dist = np.min(distances) if len(distances) > 0 else float('inf')
                
                if min_comp_dist < min_dist:
                    min_dist = min_comp_dist
                    nearest_comp_idx = i
            
            # Add the noise point to nearest component
            if connected_components:  # Only if we have components
                connected_components[nearest_comp_idx].add(teal_indices[idx])
    else:
        # Use the graph-based approach
        G = nx.Graph()
        G.add_nodes_from(teal_indices)
        
        # Add edges between nearby teal vertices
        threshold = 0.01  # Distance threshold for connectivity
        
        # Process in chunks to reduce memory usage
        chunk_size = 1000
        for i in range(0, len(teal_indices), chunk_size):
            chunk_indices = teal_indices[i:i+chunk_size]
            chunk_vertices = vertices_np[chunk_indices]
            
            # Calculate pairwise distances between this chunk and all teal vertices
            for j, idx_j in enumerate(teal_indices):
                v_j = vertices_np[idx_j]
                # Calculate squared distances
                distances = np.sum((chunk_vertices - v_j) ** 2, axis=1)
                
                # Find close vertices
                close_indices = np.where(distances < threshold)[0]
                
                # Add edges for close vertices
                for k in close_indices:
                    idx_i = teal_indices[i + k] if i + k < len(teal_indices) else teal_indices[k]
                    if idx_i != idx_j:  # Avoid self-loops
                        G.add_edge(idx_i, idx_j)
        
        # Find connected components in the initial graph
        connected_components_initial = list(nx.connected_components(G))
        print(f"Found {len(connected_components_initial)} connected teal components in initial graph.")
        
        # If multiple components already, no need for additional checks
        if len(connected_components_initial) > 1:
            connected_components = connected_components_initial
        else:
            # Find articulation points (bridge vertices)
            bridge_vertices = list(nx.articulation_points(G))
            print(f"Found {len(bridge_vertices)} bridge vertices.")
            
            # Calculate bridge ratio
            actual_bridge_ratio = len(bridge_vertices) / len(teal_indices)
            print(f"Bridge ratio: {actual_bridge_ratio:.6f} (threshold: {bridge_ratio})")
            
            if len(bridge_vertices) > connectivity_threshold or actual_bridge_ratio > bridge_ratio:
                # Remove bridge vertices to find the true components
                G_no_bridges = G.copy()
                G_no_bridges.remove_nodes_from(bridge_vertices)
                connected_components = list(nx.connected_components(G_no_bridges))
                
                # Add each bridge vertex to the nearest component
                for vertex in bridge_vertices:
                    v_pos = vertices_np[vertex]
                    
                    # Find the nearest component
                    min_dist = float('inf')
                    nearest_comp_idx = 0
                    
                    for i, comp in enumerate(connected_components):
                        comp_vertices = vertices_np[[v for v in comp]]
                        distances = np.sum((comp_vertices - v_pos)**2, axis=1)
                        min_comp_dist = np.min(distances)
                        
                        if min_comp_dist < min_dist:
                            min_dist = min_comp_dist
                            nearest_comp_idx = i
                    
                    # Add the bridge vertex to the nearest component
                    connected_components[nearest_comp_idx].add(vertex)
                
                print(f"After removing bridge vertices: {len(connected_components)} components")
            else:
                # Keep the original components
                connected_components = connected_components_initial
                print("Insufficient bridge vertices. Treating as a single component.")
    
    if len(connected_components) <= 1:
        print("Teal vertices still form a single component after analysis. Will try manual separation.")
        
        # Try to separate by coordinate clusters as a last resort
        teal_coords = vertices_np[teal_indices]
        
        # Try different eps values until we get multiple components
        for eps in [0.05, 0.1, 0.2, 0.5, 1.0]:
            clustering = DBSCAN(eps=eps, min_samples=30).fit(teal_coords)
            labels = clustering.labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if num_clusters > 1:
                print(f"Successfully split into {num_clusters} components using DBSCAN with eps={eps}")
                
                connected_components = []
                for i in range(num_clusters):
                    component = set()
                    mask = (labels == i)
                    for idx, is_in_cluster in enumerate(mask):
                        if is_in_cluster:
                            component.add(teal_indices[idx])
                    connected_components.append(component)
                
                # Handle noise points (-1 label) by assigning to nearest cluster
                noise_indices = np.where(labels == -1)[0]
                if len(noise_indices) > 0 and len(connected_components) > 0:
                    for idx in noise_indices:
                        v_pos = teal_coords[idx]
                        
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
                        
                        connected_components[nearest_cluster].add(teal_indices[idx])
                
                break
    
    if len(connected_components) <= 1:
        print("All attempts to separate teal vertices failed. No fix needed.")
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
    
    print("\nTeal component details (sorted by height):")
    for i, comp in enumerate(sorted_components):
        position = "top" if i == 0 else "lower"
        print(f"Component {i+1}: {comp['size']} vertices, y={comp['avg_y']:.3f}, position={position}")
    
    # Color palette
    colors = {
        'yellow': (1.0, 1.0, 0.0),  # For top component
        'teal': (0.0, 0.8, 0.8),    # For remaining components
    }
    
    # Assign new colors - only the top component becomes yellow
    new_colors = {}
    for i, comp_data in enumerate(sorted_components):
        if i == 0:  # Top component (highest y)
            color = colors['yellow']
            color_name = 'yellow'
        else:  # All other components remain teal
            color = colors['teal'] 
            color_name = 'teal'
        
        print(f"Coloring component {i+1} as {color_name}")
        for vertex_idx in comp_data['component']:
            new_colors[vertex_idx] = color
    
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

if __name__ == "__main__":
    # Example usage - process a single file
    obj_file_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/inner/00129_fixed.obj"
    check_and_fix_red_components(obj_file_path, 
                               connectivity_threshold=20,
                               bridge_ratio=0.001,
                               use_clustering=True,
                               cluster_eps=0.1,
                               min_cluster_size=100)
    check_and_fix_teal_components(obj_file_path, 
                               connectivity_threshold=20,
                               bridge_ratio=0.001,
                               use_clustering=True,
                               cluster_eps=0.1,
                               min_cluster_size=100)
    
    # To process multiple files (uncomment and modify as needed):
    # def process_files(directory_pattern):
    #     files = glob.glob(directory_pattern)
    #     print(f"Found {len(files)} files to process")
    #     for file_path in files:
    #         check_and_fix_red_components(file_path)
    #         check_and_fix_teal_components(file_path)
    # 
    # process_files("/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/inner/*.obj")