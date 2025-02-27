import pickle
import networkx as nx
import numpy as np

# Paths to files
obj_file_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_files/00134.obj"
labels_file_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/label_files/00134.pkl"
output_obj_file_path = "00134.obj"

# Define a base color mapping for labels other than 0 (RGB format, values in [0, 1])
label_colors = {
   1: (0.0, 1.0, 0.0),  # Green
   2: (0.0, 0.0, 1.0),  # Blue
   3: (1.0, 1.0, 0.0),  # Yellow
   4: (1.0, 0.0, 1.0),  # Magenta
   5: (1.0, 1.0, 0.0),  # Cyan
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

# Function to get component metrics with focus on average x position
def get_component_metrics(component):
   component_vertices = vertices[list(component)]
   # Calculate average x-coordinate for left/right determination
   avg_x = np.mean(component_vertices[:, 0])
   return {
       'component': component,
       'avg_x': avg_x
   }

# Calculate metrics for all components
component_metrics = []
for component in connected_components:
   metrics = get_component_metrics(component)
   component_metrics.append(metrics)

# Sort components simply by x coordinate (left to right)
sorted_components = sorted(
   component_metrics,
   key=lambda x: x['avg_x']  # Sort only by left to right
)

# Print centroids of all components
print("\nComponent centroids from left to right:")
for i, component_data in enumerate(sorted_components):
   component = component_data['component']
   component_vertices = vertices[list(component)]
   centroid = np.mean(component_vertices, axis=0)
   print(f"Component {i}: x={centroid[0]:.3f}, y={centroid[1]:.3f}, z={centroid[2]:.3f}")

# Predefined colors for components
predefined_colors = [
   (1.0, 0.5, 0.0),  # Orange
   (0.5, 0.0, 1.0),  # Purple
   (1.0, 0.0, 0.0),  # Red
   (0.5, 1.0, 0.0),  # Lime Green
   (0.0, 0.5, 1.0),  # Sky Blue
   (0.7, 0.3, 0.0),  # Brown
   (0.8, 0.4, 0.6),  # Pink
   (0.4, 0.7, 0.3),  # Forest Green
   (0.6, 0.6, 0.2),  # Olive
   (0.2, 0.5, 0.7),  # Steel Blue
]

# Create efficient color mapping
component_colors = {}
for i, component_data in enumerate(sorted_components):
   color = predefined_colors[i % len(predefined_colors)]
   for vertex in component_data['component']:
       component_colors[vertex] = color

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

print(f"\nColored OBJ file saved to: {output_obj_file_path}")