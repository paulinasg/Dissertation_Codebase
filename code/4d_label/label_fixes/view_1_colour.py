import numpy as np
import os
import sys

def extract_orange_vertices_with_faces(obj_file_path, output_file_path=None):
    """
    Extract only orange vertices from an OBJ file along with their faces and save as a new OBJ file.
    
    Args:
        obj_file_path: Path to the original colored OBJ file
        output_file_path: Path to save the orange-only OBJ file (defaults to original with _orange suffix)
    """
    if output_file_path is None:
        output_file_path = obj_file_path.replace(".obj", "_orange.obj")
    
    print(f"Processing: {obj_file_path}")
    
    # Read the OBJ file
    with open(obj_file_path, "r") as obj_file:
        lines = obj_file.readlines()
    
    # Extract vertex lines and identify orange vertices
    vertex_lines = []
    orange_indices = set()
    old_to_new_idx = {}
    vertex_count = 0
    
    for i, line in enumerate(lines):
        if line.startswith("v "):  # Vertex line
            vertex_count += 1
            parts = line.strip().split()
            
            if len(parts) >= 7:  # Vertex with color
                r, g, b = map(float, parts[4:7])
                
                # Check if this is an orange vertex (RGB: 1.0, 0.5, 0.0)
                if abs(r - 1.0) < 0.01 and abs(g - 0.5) < 0.01 and abs(b) < 0.01:
                    orange_indices.add(vertex_count)
                    old_to_new_idx[vertex_count] = len(old_to_new_idx) + 1
                    vertex_lines.append(line)
    
    print(f"Found {len(orange_indices)} orange vertices out of {vertex_count} total vertices.")
    
    if not orange_indices:
        print("No orange vertices found. No output file created.")
        return
    
    # Extract faces that only reference orange vertices
    face_lines = []
    for line in lines:
        if line.startswith("f "):  # Face line
            parts = line.strip().split()
            face_indices = []
            
            # Parse face vertex indices
            for part in parts[1:]:
                # Handle formats like "v", "v/vt", "v/vt/vn", "v//vn"
                vertex_idx = int(part.split('/')[0])
                face_indices.append(vertex_idx)
            
            # Check if all vertices in this face are orange
            if all(idx in orange_indices for idx in face_indices):
                # Remap vertex indices to new indices
                new_face_parts = ["f"]
                for idx_str in parts[1:]:
                    components = idx_str.split('/')
                    old_idx = int(components[0])
                    new_idx = old_to_new_idx[old_idx]
                    
                    # Reconstruct the face vertex reference with new index
                    if len(components) == 1:
                        new_face_parts.append(str(new_idx))
                    elif len(components) == 2:
                        new_face_parts.append(f"{new_idx}/{components[1]}")
                    else:  # len(components) == 3
                        new_face_parts.append(f"{new_idx}/{components[1]}/{components[2]}")
                
                face_lines.append(" ".join(new_face_parts) + "\n")
    
    print(f"Extracted {len(face_lines)} faces that connect only orange vertices.")
    
    # Write the new OBJ file with only orange vertices and their faces
    with open(output_file_path, "w") as out_file:
        # Add a comment header
        out_file.write(f"# OBJ file with only orange vertices from {os.path.basename(obj_file_path)}\n")
        out_file.write(f"# Original file had {vertex_count} vertices\n")
        out_file.write(f"# This file contains {len(orange_indices)} orange vertices and {len(face_lines)} faces\n\n")
        
        # Write all orange vertices
        for vertex_line in vertex_lines:
            out_file.write(vertex_line)
        
        # Write a separator comment
        out_file.write("\n# Faces\n")
        
        # Write all faces that connect only orange vertices
        for face_line in face_lines:
            out_file.write(face_line)
    
    print(f"Orange vertices and faces saved to: {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use command line argument if provided
        obj_file_path = sys.argv[1]
    else:
        # Default file path
        obj_file_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/inner/00140_fixed_red2teal.obj"
    
    extract_orange_vertices_with_faces(obj_file_path)