import pickle

# Paths to files
obj_file_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_files/00134.obj"
labels_file_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/label_files/00134.pkl"
output_obj_file_path = "00134_test.obj"



# Define a color mapping for labels 0-5 (RGB format, values in [0, 1])
label_colors = {
    0: (1.0, 0.0, 0.0),  # Red
    1: (0.0, 1.0, 0.0),  # Green
    2: (0.0, 0.0, 1.0),  # Blue
    3: (1.0, 1.0, 0.0),  # Yellow
    4: (1.0, 0.0, 1.0),  # Magenta
    5: (0.0, 1.0, 1.0),  # Cyan
}

# The actual segmentation I want is hair (1), head (0), clothed torso (5 and 3), unclothed arms/hands (0 to the side), clothed legs (4), unclothed legs (lowest 0), shoes (2)
# So I want to separate the areas labelled 0 into distinct areas and label them different colours
# And I want to group 3 and 5 together into one colour

# label_colors = {
#     0: (1.0, 0.0, 0.0),  # Red - head
#     1: (0.0, 1.0, 0.0),  # Green - hair
#     2: (0.0, 0.0, 1.0),  # Blue - shoes
#     3: (1.0, 1.0, 0.0),  # Yellow - clothed torso
#     4: (1.0, 0.0, 1.0)  # Magenta
# }


# Step 1: Load labels from the .pkl file
with open(labels_file_path, "rb") as f:
    labels_data = pickle.load(f)

labels = labels_data['scan_labels']  # Adjust this key if necessary

# Step 2: Read the OBJ file and modify vertices
with open(obj_file_path, "r") as obj_file:
    lines = obj_file.readlines()

colored_lines = []
vertex_index = 0  # Track vertex index to map with labels

for line in lines:
    if line.startswith("v "):  # Vertex line
        # Parse the original vertex position
        parts = line.strip().split()
        x, y, z = parts[1:4]

        # Get the corresponding label and its color
        label = labels[vertex_index]
        color = label_colors[label]
        r, g, b = color

        # Write the vertex with color
        colored_lines.append(f"v {x} {y} {z} {r} {g} {b}\n")
        vertex_index += 1
    else:
        # Keep other lines unchanged
        colored_lines.append(line)

# Step 3: Write the new OBJ file
with open(output_obj_file_path, "w") as output_file:
    output_file.writelines(colored_lines)

print(f"Colored OBJ file saved to: {output_obj_file_path}")
