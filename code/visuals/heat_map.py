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
        face_count = pv_mesh.n_faces
        faces_flat = pv_mesh.faces
        
        # Initialize faces array
        faces = np.empty((face_count, 3), dtype=np.int64)
        
        # Extract the face indices while skipping the count values
        i_face = 0
        i_flat = 0
        while i_face < face_count:
            n_points = faces_flat[i_flat]
            if n_points == 3:  # We only support triangular faces
                faces[i_face] = faces_flat[i_flat+1:i_flat+4]
                i_face += 1
            i_flat += n_points + 1
    else:
        faces = None
    
    # Create trimesh
    return trimesh.Trimesh(vertices=vertices, faces=faces)

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

def get_distance_colors(distances, cmap_name="brighter_heatmap", clim=None):
    """
    Convert distances to RGB colors using a custom, brighter colormap.
    
    Args:
        distances: Array of distance values
        cmap_name: Name of colormap to use
        clim: Optional [min, max] limits for the colormap
    
    Returns:
        colors: Nx3 array of RGB values (0-1)
    """
    if clim is None:
        clim = [0, np.percentile(distances, 95)]  # Use 95th percentile to avoid outliers
    
    # Create a brighter custom colormap (blue to green to red)
    # Increasing saturation and brightness compared to default
    brighter_cmap = LinearSegmentedColormap.from_list(
        "brighter_heatmap", 
        [(0, (0.0, 0.86, 1.0)),       # Bright blue
         (0.5, (0.91, 1.0, 0.0)),     # Bright yellow
         (1, (1.0, 0.0, 0.0))]       # Bright red
    )
    
    # Normalize distances to 0-1 range based on limits
    norm_distances = np.clip((distances - clim[0]) / (clim[1] - clim[0]), 0, 1)
    
    # Convert to RGB colors
    colors = brighter_cmap(norm_distances)[:, :3]  # Get RGB, discard alpha
    
    return colors

def save_colored_obj(mesh, colors, output_path):
    """
    Save mesh as an OBJ file with vertex colors.
    
    Args:
        mesh: PyVista PolyData mesh
        colors: Nx3 array of RGB values (0-1)
        output_path: Path to save the OBJ file
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
    
    # Ensure colors are in the right format (Nx3 array of values 0-1)
    colors = np.clip(colors, 0, 1)
    
    # Write OBJ file with vertex colors
    with open(output_path, 'w') as f:
        # Write vertices with colors
        for i, (v, c) in enumerate(zip(vertices, colors)):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Successfully saved colored mesh with {len(vertices)} vertices and {len(faces)} faces")
    return output_path

def normalize_mesh_orientation(mesh):
    """
    Normalize mesh orientation to face forward while preserving vertical orientation.
    """
    # Create a copy of the mesh to work with
    mesh_normalized = mesh.copy()
    
    # Get the mesh center
    center = mesh.center
    
    # Center the mesh by subtracting the center coordinates
    mesh_normalized.points -= center
    
    # Compute PCA to find principal axes
    points = mesh_normalized.points
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # The Y-axis (up) should align with the second principal component
    # Create a rotation matrix that preserves Y-axis
    y_axis = np.array([0, 1, 0])
    
    # Project the first principal component onto the XZ plane
    forward = eigenvectors[:, 0]
    forward[1] = 0  # Remove Y component
    forward = forward / np.linalg.norm(forward)  # Normalize
    
    # Compute right vector using cross product
    right = np.cross(y_axis, forward)
    right = right / np.linalg.norm(right)
    
    # Create rotation matrix
    rotation = np.column_stack((right, y_axis, forward))
    
    # Create full transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    
    # Apply the rotation
    mesh_normalized.transform(transform)
    
    return mesh_normalized

def save_distance_visualization(target_mesh, source_mesh, output_obj_path, clim=None):
    """
    Create and save a colored mesh based on distances.
    
    Args:
        target_mesh: Target mesh (PyVista PolyData)
        source_mesh: Reference mesh (PyVista PolyData)
        output_obj_path: Path to save the colored OBJ
        clim: Optional [min, max] limits for the color scale
    
    Returns:
        distances: Array of distances
    """
    # Compute distances
    distances = compute_mesh_distances(target_mesh, source_mesh)
    
    # Add distances as a scalar field to the target mesh
    target_mesh["distances"] = distances
    
    # Determine colormap limits if not provided
    if clim is None:
        clim = [0, np.percentile(distances, 95)]  # Use 95th percentile to avoid outliers
    
    print(f"Distance range: {clim[0]:.6f} to {clim[1]:.6f}")
    
    # Get colors based on distances
    colors = get_distance_colors(distances, clim=clim)
    
    # Add normalization step before saving
    normalized_mesh = normalize_mesh_orientation(target_mesh)
    
    # Save the normalized mesh with colors
    save_colored_obj(normalized_mesh, colors, output_obj_path)
    
    # Print statistics
    print(f"Distance statistics:")
    print(f"  Min: {distances.min():.6f}")
    print(f"  Max: {distances.max():.6f}")
    print(f"  Mean: {distances.mean():.6f}")
    print(f"  Median: {np.median(distances):.6f}")
    print(f"  Std Dev: {distances.std():.6f}")
    
    return distances

def main(target_path, source_path, output_path=None):
    """
    Main function to load meshes and save colored distance visualization.
    
    Args:
        target_path: Path to the target mesh OBJ file
        source_path: Path to the source mesh OBJ file
        output_path: Path to save the colored OBJ (optional)
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
    
    # Default output path if not provided
    if output_path is None:
        if isinstance(target_path, str):
            base_name = os.path.splitext(os.path.basename(target_path))[0]
            output_path = f"{base_name}_distance_colored.obj"
        else:
            output_path = "distance_colored.obj"
    
    # Process and save the visualization
    distances = save_distance_visualization(
        target_mesh, 
        source_mesh,
        output_path
    )
    
    print(f"Colored mesh saved to {output_path}")
    return distances

if __name__ == "__main__":

    target_mesh = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/code/align/aligned_result_man.obj"
    source_mesh = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_files/00122.obj"
    output = "00122.obj"

    main(target_mesh, source_mesh, output)