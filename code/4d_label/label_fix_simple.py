import numpy as np
import os
import glob
from sklearn.cluster import KMeans

def separate_orange_into_three(obj_file_path, output_file_path=None):
    """
    Simplified function to separate orange vertices into exactly 3 components by height.
    
    Args:
        obj_file_path: Path to the colored OBJ file
        output_file_path: Path to save the fixed OBJ file (defaults to original with _fixed suffix)
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
    
    # Convert to numpy array for faster operations
    vertices_np = np.array(vertices)
    orange_vertices = vertices_np[orange_indices]
    
    # Use K-means to split into exactly 3 clusters
    print("Applying K-means clustering with k=3...")
    kmeans = KMeans(n_clusters=3, random_state=42).fit(orange_vertices)
    labels = kmeans.labels_
    
    # Create components from clusters
    components = []
    for i in range(3):
        cluster_indices = np.where(labels == i)[0]
        component = set(orange_indices[idx] for idx in cluster_indices)
        components.append(component)
    
    # Calculate component metrics
    component_metrics = []
    for comp in components:
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
        position = "top" if i == 0 else ("middle" if i == 1 else "bottom")
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
        elif i == 1:  # Middle component
            color = colors['red'] 
            color_name = 'red'
        else:  # Bottom component
            color = colors['teal']
            color_name = 'teal'
        
        print(f"Coloring component {i+1} as {color_name}")
        for vertex_idx in comp_data['component']:
            new_colors[vertex_idx] = color
    
    # Check if there are existing red vertices and change them to teal if needed
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

if __name__ == "__main__":
    # Process specific file
    obj_file_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/inner/00140_fixed_red2teal.obj"
    separate_orange_into_three(obj_file_path)