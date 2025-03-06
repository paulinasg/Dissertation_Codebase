import numpy as np
import trimesh
import pyvista as pv
import os
from pathlib import Path

def load_obj_with_colors(file_path):
    """
    Load an OBJ file as a PyVista mesh.
    """
    try:
        mesh = trimesh.load(file_path)
        vertices = mesh.vertices
        faces = mesh.faces
        if faces.shape[1] == 3:  # Triangular mesh
            faces = np.column_stack((np.full(len(faces), 3), faces))
        pv_mesh = pv.PolyData(vertices, faces)
        return pv_mesh
    except Exception as e:
        print(f"Error loading mesh: {e}")
        raise

def compute_normals(mesh):
    """
    Compute vertex normals for a mesh.
    """
    # Use PyVista to compute normals
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    normals = mesh.point_normals
    
    # Normalize the normals (just to be sure)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    # Avoid division by zero
    norm[norm == 0] = 1.0
    normals = normals / norm
    
    return normals

def normals_to_colors(normals):
    """
    Convert normal vectors to RGB colors.
    Each XYZ component maps to RGB, remapped from [-1,1] to [0,1].
    """
    # Map from [-1, 1] to [0, 1] range
    colors = (normals + 1) * 0.5
    
    return np.clip(colors, 0, 1)

def save_colored_obj(mesh, colors, output_path):
    """
    Save mesh as an OBJ file with vertex colors.
    """
    print(f"Saving colored mesh to {output_path}...")
    
    vertices = mesh.points
    
    # Convert PyVista face data to standard face indices
    face_data = mesh.faces
    faces = []
    i = 0
    while i < len(face_data):
        n_vertices = face_data[i]
        if n_vertices == 3:  # Only handle triangular faces
            faces.append([face_data[i+1]+1, face_data[i+2]+1, face_data[i+3]+1])  # OBJ is 1-indexed
        i += n_vertices + 1
    
    # Write OBJ file with vertex colors
    with open(output_path, 'w') as f:
        # Write vertices with colors
        for i, (v, c) in enumerate(zip(vertices, colors)):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Successfully saved colored mesh with {len(vertices)} vertices and {len(faces)} faces")

def color_mesh_by_normals(input_path, output_path=None):
    """
    Main function to color a mesh by its vertex normals.
    """
    print(f"Loading mesh from {input_path}...")
    
    # Load the mesh
    mesh = load_obj_with_colors(input_path)
    
    print(f"Mesh loaded: {mesh.n_points} points, {mesh.n_cells} faces")
    
    # Compute normals
    print("Computing normals...")
    normals = compute_normals(mesh)
    
    # Convert normals to colors
    print("Creating colors from normals...")
    colors = normals_to_colors(normals)
    
    # Default output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_normal_colored.obj"
    
    # Save colored mesh
    save_colored_obj(mesh, colors, output_path)
    
    return output_path

if __name__ == "__main__":
    import sys
    
    # Default file paths to use if no command line arguments are provided
    default_input_file = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_files/00122.obj"
    default_output_file = "00122_normal_colored.obj"
    
    # Use command-line arguments if provided, otherwise use defaults
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        print("Using files from command line arguments.")
    else:
        input_file = default_input_file
        output_file = default_output_file
        print(f"Using default files:\nInput: {input_file}\nOutput: {output_file}")
    
    try:
        result_path = color_mesh_by_normals(input_file, output_file)
        print(f"Colored mesh saved to: {result_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)