import pickle
import networkx as nx
import numpy as np
import os
import glob

def color_model(obj_file_path, labels_file_path, output_obj_file_path):
    """Color a 3D model based on its segmentation labels, exactly matching colour_model_custom.py"""
    # Define a base color mapping for labels other than 0 (RGB format, values in [0, 1])
    # Exactly matching the original script's colors
    label_colors = {
       1: (0.0, 1.0, 0.0),  # Green
       2: (0.0, 0.0, 1.0),  # Blue
       3: (1.0, 1.0, 0.0),  # Yellow
       4: (1.0, 0.0, 1.0),  # Magenta
       5: (1.0, 1.0, 0.0),  # Cyan - using exact value from original script
    }

    # Step 1: Load labels from the .pkl file
    with open(labels_file_path, "rb") as f:
        labels_data = pickle.load(f)
    labels = labels_data['scan_labels']  # Adjust this key if necessary

    # Step 2: Read the OBJ file and parse vertices
    with open(obj_file_path, "r") as obj_file:
        lines = obj_file.readlines()

    vertices = []
    for line in lines:
        if line.startswith("v "):  # Vertex line
            parts = line.strip().split()
            vertices.append(tuple(map(float, parts[1:4])))

    # Convert to numpy array for faster operations
    vertices = np.array(vertices)

    # Step 3: Identify unconnected components for label 0 using NetworkX
    label_0_indices = [i for i, label in enumerate(labels) if label == 0]

    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(label_0_indices)

    # Vectorized distance calculation
    threshold = 0.01  # Adjust this value if needed
    vertices_0 = vertices[label_0_indices]

    # Process edges in chunks to reduce memory usage
    chunk_size = 1000
    for i in range(0, len(label_0_indices), chunk_size):
        chunk_end = min(i + chunk_size, len(label_0_indices))
        chunk_vertices = vertices_0[i:chunk_end]
        
        # Calculate distances for this chunk to all vertices
        for j in range(i, len(label_0_indices)):
            v2 = vertices_0[j]
            distances = np.sum((chunk_vertices - v2) ** 2, axis=1)
            
            # Find close vertices
            close_indices = np.where(distances < threshold)[0]
            
            # Add edges for close vertices
            for k in close_indices:
                if i + k < j:  # Avoid duplicate edges and self-loops
                    G.add_edge(label_0_indices[i + k], label_0_indices[j])

    # Find connected components
    connected_components = list(nx.connected_components(G))

    # Same get_component_metrics function as in the original script
    def get_component_metrics(component):
        component_vertices = vertices[list(component)]
        avg_x = np.mean(component_vertices[:, 0])
        avg_y = np.mean(component_vertices[:, 1])  # Add y-coordinate calculation
        return {
            'component': component,
            'avg_x': avg_x,
            'avg_y': avg_y
        }

    # Calculate metrics for all components
    component_metrics = []
    for component in connected_components:
        metrics = get_component_metrics(component)
        component_metrics.append(metrics)

    # Replace predefined_colors with new specific colors
    specific_colors = {
        'orange': (1.0, 0.5, 0.0),    # Orange for top component (head)
        'red': (1.0, 0.0, 0.0),    # Red for middle components (hands)
        'teal': (0.0, 0.8, 0.8)       # Teal for bottom components (legs)
    }

    # Sort components by y-coordinate (height)
    sorted_components = sorted(
        component_metrics,
        key=lambda x: x['avg_y'],
        reverse=True  # Highest y first
    )

    # Create color mapping based on vertical position
    component_colors = {}
    for i, component_data in enumerate(sorted_components):
        if i == 0:  # Top component
            color = specific_colors['orange']
        elif i in [1, 2]:  # Middle two components
            color = specific_colors['red']
        else:  # Bottom components
            color = specific_colors['teal']
        
        for vertex in component_data['component']:
            component_colors[vertex] = color

    # Print component information for verification - exactly as in original script
    print("\nComponent positions and colors:")
    for i, component_data in enumerate(sorted_components):
        component = component_data['component']
        component_vertices = vertices[list(component)]
        centroid = np.mean(component_vertices, axis=0)
        color_name = "orange" if i == 0 else ("violet" if i in [1, 2] else "teal")
        print(f"Component {i}: x={centroid[0]:.3f}, y={centroid[1]:.3f}, z={centroid[2]:.3f}, color={color_name}")

    # Step 4: Modify the OBJ file lines with colors
    colored_lines = []
    vertex_index = 0

    for line in lines:
        if line.startswith("v "):  # Vertex line
            parts = line.strip().split()
            x, y, z = parts[1:4]
            
            # Get the corresponding label and its color
            label = labels[vertex_index]
            
            if label == 0:
                # Use the color assigned to this specific component
                r, g, b = component_colors.get(vertex_index, (1.0, 1.0, 1.0))  # Default to white
            else:
                # Use the pre-defined label color
                r, g, b = label_colors.get(label, (1.0, 1.0, 1.0))  # Default to white
            
            # Write the vertex with color
            colored_lines.append(f"v {x} {y} {z} {r} {g} {b}\n")
            vertex_index += 1
        else:
            # Keep other lines unchanged
            colored_lines.append(line)

    # Step 5: Write the new OBJ file
    with open(output_obj_file_path, "w") as output_file:
        output_file.writelines(colored_lines)


def main():
    """Process all matching model pairs between obj_4ddress_files and label_files/outer directories."""
    # Define directories
    obj_dir = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_files"
    labels_dir = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/label_files/outer"
    
    # Create output directory for colored models
    output_dir = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Colored models will be saved to: {output_dir}")
    
    # Get all .obj files in the obj_4ddress_files directory
    obj_files = glob.glob(os.path.join(obj_dir, "*.obj"))
    
    print(f"Found {len(obj_files)} .obj files")
    
    # Process each .obj file
    for i, obj_file_path in enumerate(sorted(obj_files)):
        # Get the basename (e.g., "00163" from "00163.obj")
        basename = os.path.splitext(os.path.basename(obj_file_path))[0]
        
        # Construct the path to the corresponding .pkl file
        labels_file_path = os.path.join(labels_dir, f"{basename}.pkl")
        
        # Construct the output path
        output_obj_file_path = os.path.join(output_dir, f"{basename}.obj")
        
        # Check if the label file exists
        if not os.path.exists(labels_file_path):
            print(f"Warning: Label file not found for {basename}, skipping")
            continue
        
        print(f"\nProcessing file pair {i+1}/{len(obj_files)}: {basename}")
        print(f"OBJ file: {obj_file_path}")
        print(f"Label file: {labels_file_path}")
        print(f"Output file: {output_obj_file_path}")
        
        try:
            # Apply the coloring function
            color_model(obj_file_path, labels_file_path, output_obj_file_path)
            print(f"Successfully colored and saved: {basename}.obj")
        except Exception as e:
            print(f"Error processing {basename}: {str(e)}")
    
    print("\nColoring process completed!")


if __name__ == "__main__":
    main()