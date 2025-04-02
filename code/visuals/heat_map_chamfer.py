import numpy as np
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from pathlib import Path

def load_obj_with_colors(file_path):
    """
    Custom loader for OBJ files with vertex colors.
    
    Args:
        file_path: Path to the OBJ file
    
    Returns:
        pv.PolyData: PyVista mesh
    """
    # First try using trimesh which handles vertex colors better
    try:
        mesh = trimesh.load(file_path)
        # Convert to PyVista
        vertices = mesh.vertices
        faces = mesh.faces
        if faces.shape[1] == 3:  # Triangular mesh
            faces = np.column_stack((np.full(len(faces), 3), faces))
        pv_mesh = pv.PolyData(vertices, faces)
        
        # If there are vertex colors, add them
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Normalize to 0-1
            pv_mesh['colors'] = colors
            
        return pv_mesh
    except Exception as e:
        print(f"Warning: Trimesh failed to load the mesh properly: {e}")
        print("Falling back to manual OBJ parsing...")
    
    # Manual parsing as fallback
    vertices = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            if parts[0] == 'v':
                # Extract just the vertex coordinates (first 3 values after 'v')
                try:
                    coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                    vertices.append(coords)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid vertex line: {line.strip()}")
                    
            elif parts[0] == 'f':
                # Extract face indices
                try:
                    # Handle different face formats (v, v/vt, v/vt/vn)
                    face_indices = []
                    for p in parts[1:]:
                        idx = p.split('/')[0]  # Get just the vertex index
                        face_indices.append(int(idx) - 1)  # OBJ indices are 1-based
                    
                    if len(face_indices) == 3:  # Triangle
                        faces.append(face_indices)
                    elif len(face_indices) == 4:  # Quad - convert to two triangles
                        faces.append(face_indices[:3])
                        faces.append([face_indices[0], face_indices[2], face_indices[3]])
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid face line: {line.strip()}")
    
    if not vertices:
        raise ValueError("No vertices found in the OBJ file")
    
    # Convert to numpy arrays
    vertices = np.array(vertices)
    
    # Create PyVista mesh
    if faces:
        faces = np.array(faces)
        # Add count to the beginning of each face for PyVista
        face_list = np.column_stack((np.full(len(faces), 3), faces))
        pv_mesh = pv.PolyData(vertices, face_list)
    else:
        # If no faces, create a point cloud
        pv_mesh = pv.PolyData(vertices)
    
    return pv_mesh

def pyvista_to_trimesh(pv_mesh):
    """
    Convert a PyVista mesh to a Trimesh mesh.
    
    Args:
        pv_mesh: PyVista PolyData mesh
    
    Returns:
        trimesh.Trimesh: Converted mesh
    """
    # Extract vertices
    vertices = pv_mesh.points
    
    # Extract faces - need to convert from VTK format to simple indices
    if pv_mesh.faces.size > 0:
        # VTK format includes count at the beginning of each face
        # Need to reshape and skip these counts
        faces_with_counts = pv_mesh.faces.reshape(-1, 4)
        faces = faces_with_counts[:, 1:4]
    else:
        faces = None
    
    # Create trimesh
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def compute_chamfer_distance(mesh_a, mesh_b):
    """
    Compute Chamfer Distance between two meshes.
    
    Args:
        mesh_a: First mesh (PyVista PolyData)
        mesh_b: Second mesh (PyVista PolyData)
    
    Returns:
        (a_to_b_distances, b_to_a_distances): Two arrays of distances
            a_to_b_distances: Distance from each vertex in mesh_a to mesh_b
            b_to_a_distances: Distance from each vertex in mesh_b to mesh_a
    """
    # Convert PyVista meshes to trimesh
    trimesh_a = pyvista_to_trimesh(mesh_a)
    trimesh_b = pyvista_to_trimesh(mesh_b)
    
    # Get vertices
    vertices_a = mesh_a.points
    vertices_b = mesh_b.points
    
    print(f"Computing Chamfer Distance:")
    print(f" - From {len(vertices_a)} points in mesh A to mesh B...")
    # A -> B distances
    _, a_to_b_distances, _ = trimesh.proximity.closest_point(trimesh_b, vertices_a)
    
    print(f" - From {len(vertices_b)} points in mesh B to mesh A...")
    # B -> A distances
    _, b_to_a_distances, _ = trimesh.proximity.closest_point(trimesh_a, vertices_b)
    
    return np.array(a_to_b_distances), np.array(b_to_a_distances)

def compute_mesh_distances(target_mesh, source_mesh):
    """
    Compute Chamfer Distance from target mesh to source mesh.
    """
    # Compute both directions of the Chamfer Distance
    target_to_source, source_to_target = compute_chamfer_distance(target_mesh, source_mesh)
    
    # Store the mean as a field with the correct size
    target_mesh["source_to_target"] = np.full(target_mesh.n_points, np.mean(source_to_target))
    
    return target_to_source

def visualize_distance_heatmap(target_mesh, source_mesh, cmap="coolwarm", clim=[0, 0.1], 
                              show_edges=False, point_size=5, save_path=None):
    """
    Visualize point-to-surface distances as a heatmap with fixed scale.
    
    Args:
        target_mesh: The mesh to visualize distances on
        source_mesh: The reference mesh
        cmap: Colormap for the heatmap
        clim: Fixed limits for the colormap [min, max], default [0, 0.1]
        show_edges: Whether to show mesh edges
        point_size: Size of points in visualization
        save_path: Path to save screenshot (optional)
    """
    # Compute distances
    distances = compute_mesh_distances(target_mesh, source_mesh)
    
    # Add distances as a scalar field to the target mesh
    target_mesh["distances"] = distances
    
    # Create a custom colormap with normalized scale
    custom_cmap = LinearSegmentedColormap.from_list(
        "FixedScale", 
        [
            (0.0, '#00DDFF'),    # Blue for minimum
            (0.5, '#EEFF00'),    # Yellow for middle
            (1.0, '#FF0000')     # Red for maximum
        ]
    )
    
    # Create the visualization with fixed scale
    p = pv.Plotter()
    p.add_mesh(target_mesh, scalars="distances", cmap=custom_cmap, 
              clim=clim,  # This enforces the fixed scale
              show_edges=show_edges, point_size=point_size)
    
    # Add a colorbar with fixed scale
    p.add_scalar_bar(title="Chamfer Distance (m)", fmt="%.3f")
    
    # Show the plot
    p.show(screenshot=save_path)
    
    return distances, p

def main(target_path=None, source_path=None, output_path=None):
    """
    Main function to load meshes and visualize distances using Chamfer Distance.
    
    Args:
        target_path: Path to the target mesh OBJ file (optional)
        source_path: Path to the source mesh OBJ file (optional)
        output_path: Path to save the visualization (optional)
    """
    # Default paths if not provided
    if target_path is None:
        # Specify your default target mesh path here
        target_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_aligned_files/inner/00129.obj"
        print(f"Using default target mesh: {target_path}")
    
    if source_path is None:
        # Specify your default source mesh path here
        source_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_files/inner/00129.obj"
        print(f"Using default source mesh: {source_path}")
    
    print(f"Loading meshes...")
    
    # Check if the input paths are strings/paths or already loaded meshes
    if isinstance(target_path, (str, Path)):
        target_mesh = load_obj_with_colors(target_path)
    else:
        target_mesh = target_path
    
    if isinstance(source_path, (str, Path)):
        source_mesh = load_obj_with_colors(source_path)
    else:
        source_mesh = source_path
    
    print(f"Target mesh: {target_mesh.n_points} points, {target_mesh.n_cells} faces")
    print(f"Source mesh: {source_mesh.n_points} points, {source_mesh.n_cells} faces")
    
    print("Computing and visualizing Chamfer distances...")
    distances, plotter = visualize_distance_heatmap(
        target_mesh, 
        source_mesh,
        save_path=output_path
    )
    
    # Calculate combined Chamfer Distance
    source_to_target_mean = target_mesh["source_to_target"][0]
    target_to_source_mean = np.mean(distances)
    chamfer_distance = (target_to_source_mean + source_to_target_mean) / 2
    
    # Print statistics
    print(f"Distance statistics:")
    print(f"  Target → Source Mean: {target_to_source_mean:.6f}")
    print(f"  Source → Target Mean: {source_to_target_mean:.6f}")
    print(f"  Chamfer Distance: {chamfer_distance:.6f}")
    print(f"  Min: {distances.min():.6f}")
    print(f"  Max: {distances.max():.6f}")
    print(f"  Mean: {distances.mean():.6f}")
    print(f"  Median: {np.median(distances):.6f}")
    print(f"  Std Dev: {distances.std():.6f}")
    
    return distances, chamfer_distance

if __name__ == "__main__":
    # This code will run when the script is executed directly
    main()