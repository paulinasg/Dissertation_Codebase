import os
import sys
import pyvista as pv
import trimesh
import numpy as np
from pathlib import Path

# Define directories
PIFUHD_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_aligned_files'
DRESS_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files'

def load_mesh(file_path):
    """Load a mesh file using trimesh"""
    try:
        return trimesh.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def trimesh_to_pyvista(tm_mesh):
    """Convert a trimesh mesh to a PyVista mesh"""
    vertices = tm_mesh.vertices
    faces = np.column_stack((np.ones(len(tm_mesh.faces), dtype=np.int64) * 3, tm_mesh.faces))
    pv_mesh = pv.PolyData(vertices, faces)
    return pv_mesh

def visualize_meshes(pifuhd_file, dress_file):
    """Visualize two meshes overlapping with different colors"""
    # Load meshes with trimesh
    pifuhd_mesh = load_mesh(pifuhd_file)
    dress_mesh = load_mesh(dress_file)
    
    if pifuhd_mesh is None or dress_mesh is None:
        print("Failed to load one or both meshes.")
        return
    
    # Convert to PyVista meshes
    pv_pifuhd = trimesh_to_pyvista(pifuhd_mesh)
    pv_dress = trimesh_to_pyvista(dress_mesh)
    
    # Create a plotter
    plotter = pv.Plotter(window_size=[1024, 768])
    
    # Add meshes with colors
    plotter.add_mesh(pv_pifuhd, color='red', opacity=0.7, label="PiFUHD")
    plotter.add_mesh(pv_dress, color='blue', opacity=0.7, label="4DDRESS")
    
    # Add legend
    plotter.add_legend(size=(0.2, 0.2), loc='upper right')
    
    # Enable shadows for better depth perception
    plotter.enable_shadows()
    
    # Use a preset camera position for better view
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.2)
    
    # Show the plotter
    plotter.show()

def main():
    # SPECIFY YOUR FILES HERE - change the file names as needed
    file_basename = "00174"  # Change this to the file you want to visualize
    
    # Construct full file paths
    pifuhd_file = os.path.join(PIFUHD_DIR, f"{file_basename}.obj")
    dress_file = os.path.join(DRESS_DIR, f"{file_basename}.obj")
    
    print(f"Comparing:\n1. {pifuhd_file}\n2. {dress_file}")
    
    # Visualize the meshes
    visualize_meshes(pifuhd_file, dress_file)

if __name__ == "__main__":
    main()