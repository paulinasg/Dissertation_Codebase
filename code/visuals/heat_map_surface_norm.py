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

def compute_normal_differences(target_mesh, source_mesh):
    """
    Compute surface normal differences between target mesh vertices and closest points on source mesh.
    
    Args:
        target_mesh: The mesh whose surface normals we're measuring from (PyVista PolyData)
        source_mesh: The reference mesh we're comparing to (PyVista PolyData)
    
    Returns:
        normal_diffs: Array of normal differences for each vertex in target_mesh (normalized 0-1)
    """
    # Calculate surface normals for target mesh
    target_mesh.compute_normals(inplace=True)
    target_normals = target_mesh["Normals"]
    
    # Convert PyVista mesh to trimesh for faster closest point queries
    source_trimesh = pyvista_to_trimesh(source_mesh)
    
    # Calculate face normals for source mesh if not already computed
    source_trimesh.face_normals
    
    # Get vertices from target mesh
    vertices = target_mesh.points
    
    print(f"Computing normal differences from {len(vertices)} points to reference mesh...")
    
    # For each vertex in target mesh, find the closest point on source mesh
    closest_points, _, triangle_ids = trimesh.proximity.closest_point(source_trimesh, vertices)
    
    # Get face normals for the closest triangles on source mesh
    source_normals = source_trimesh.face_normals[triangle_ids]
    
    # Ensure target_normals matches the number of vertices
    if len(target_normals) != len(vertices):
        print(f"Warning: Normal count mismatch. Using vertex normals directly...")
        # Alternative: compute vertex normals differently
        target_mesh.point_data["Normals"] = target_mesh.point_normals
        target_normals = target_mesh.point_data["Normals"]
    
    # Compute the dot product between normalized normals (dot product = cos(angle))
    # Ensure normals are normalized
    target_normals_normalized = target_normals / np.linalg.norm(target_normals, axis=1)[:, np.newaxis]
    source_normals_normalized = source_normals / np.linalg.norm(source_normals, axis=1)[:, np.newaxis]
    
    dot_products = np.sum(target_normals_normalized * source_normals_normalized, axis=1)
    
    # Clip dot products to [-1, 1] to account for numerical instability
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Compute angle in degrees
    angles = np.arccos(dot_products) * 180 / np.pi
    
    # FLIPPED: Normalize to [0, 1] range where 1 is perfect alignment and 0 is maximum difference
    normal_diffs = 1.0 - (angles / 180.0)
    
    return normal_diffs

def visualize_normal_heatmap(target_mesh, source_mesh, cmap="coolwarm", clim=[0, 1], 
                             show_edges=False, point_size=5, save_path=None):
    """
    Visualize surface normal differences as a heatmap.
    
    Args:
        target_mesh: The mesh to visualize normal differences on
        source_mesh: The reference mesh
        cmap: Colormap for the heatmap
        clim: Fixed limits for the colormap [min, max], default [0, 1]
        show_edges: Whether to show mesh edges
        point_size: Size of points in visualization
        save_path: Path to save screenshot (optional)
    """
    # Compute normal differences
    normal_diffs = compute_normal_differences(target_mesh, source_mesh)
    
    # Add normal differences as a scalar field to the target mesh
    target_mesh["normal_diffs"] = normal_diffs
    
    # Create a custom colormap with normalized scale
    custom_cmap = LinearSegmentedColormap.from_list(
        "FixedScale", 
        [
            (0.0, '#FF0000'),    # Red for minimum (opposite normals)
            (0.5, '#EEFF00'),    # Yellow for middle 
            (1.0, '#00DDFF')     # Blue for maximum (perfect alignment)
        ]
    )
    
    # Create the visualization with fixed scale
    p = pv.Plotter()
    p.add_mesh(target_mesh, scalars="normal_diffs", cmap=custom_cmap, 
              clim=clim,  # This enforces the fixed scale
              show_edges=show_edges, point_size=point_size)
    
    # Add a colorbar with fixed scale
    p.add_scalar_bar(title="Normal Alignment (1=aligned, 0=opposite)", fmt="%.2f")
    
    # Show the plot
    p.show(screenshot=save_path)
    
    return normal_diffs, p

def compute_mesh_distances(target_mesh, source_mesh):
    """
    Compute point-to-surface distances from vertices in target_mesh to the surface of source_mesh.
    
    Args:
        target_mesh: The mesh whose vertices we're measuring from (PyVista PolyData)
        source_mesh: The reference mesh we're measuring to (PyVista PolyData)
    
    Returns:
        distances: Array of distances for each vertex in target_mesh
    """
    # Convert PyVista mesh to trimesh
    source_trimesh = pyvista_to_trimesh(source_mesh)
    
    # Get vertices from target mesh
    vertices = target_mesh.points
    
    print(f"Computing distances from {len(vertices)} points to reference mesh...")
    
    # Use trimesh's proximity query for efficient distance computation
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(source_trimesh, vertices)
    
    return np.array(distances)

def visualize_distance_heatmap(target_mesh, source_mesh, cmap="coolwarm", clim=[0, 0.07], 
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

def main(target_path, source_path, output_path=None):
    """
    Main function to load meshes and visualize normal differences.
    
    Args:
        target_path: Path to the target mesh OBJ file
        source_path: Path to the source mesh OBJ file
        output_path: Path to save the visualization (optional)
    """
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
    
    print("Computing and visualizing normal differences...")
    normal_diffs, plotter = visualize_normal_heatmap(
        target_mesh, 
        source_mesh,
        save_path=output_path
    )
    
    # Print statistics
    print(f"Normal difference statistics (0=aligned, 1=opposite):")
    print(f"  Min: {normal_diffs.min():.6f}")
    print(f"  Max: {normal_diffs.max():.6f}")
    print(f"  Mean: {normal_diffs.mean():.6f}")
    print(f"  Median: {np.median(normal_diffs):.6f}")
    print(f"  Std Dev: {normal_diffs.std():.6f}")
    
    return normal_diffs

if __name__ == "__main__":

    target_mesh = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_aligned_files/inner/00129.obj"
    source_mesh = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_files/inner/00129.obj"
    output = "00129.png"

    main(target_mesh, source_mesh, output)