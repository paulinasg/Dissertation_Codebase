import os
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# Reuse constants from surface_norm.py
BODY_PART_LABELS = {
    (0.0, 0.8, 0.8): "Legs",
    (1.0, 0.0, 1.0): "Bottom-Half Clothes",
    (0.0, 0.0, 1.0): "Shoes",
    (1.0, 1.0, 0.0): "Top-Half Clothes", 
    (1.0, 0.5, 0.0): "Head",
    (0.0, 1.0, 0.0): "Hair",
    (1.0, 0.0, 0.0): "Hands"
}

COLOR_NAMES = {
    (0.0, 0.8, 0.8): "Cyan",
    (1.0, 0.0, 1.0): "Magenta",
    (0.0, 0.0, 1.0): "Blue",
    (1.0, 1.0, 0.0): "Yellow", 
    (1.0, 0.5, 0.0): "Orange",
    (0.0, 1.0, 0.0): "Green",
    (1.0, 0.0, 0.0): "Red"
}

# Define directories
SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files'
NON_SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_labelled_files'
OUTPUT_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/normal_maps'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_obj_with_face_groups(file_path):
    """
    Load OBJ file and extract vertices, faces, and face groups by color
    """
    vertices = []
    faces = []
    vertex_colors = []
    face_groups = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Get vertex coordinates
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
                
                # Get color values if they exist
                if len(parts) >= 7:
                    color = tuple(float(parts[i]) for i in range(4, 7))
                    vertex_colors.append(color)
                else:
                    vertex_colors.append(None)
                    
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                # OBJ indices start from 1, so subtract 1 for 0-based indexing
                face = [int(part.split('/')[0]) - 1 for part in parts]
                faces.append(face)
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Assign faces to color groups based on vertex colors
    for face_idx, face in enumerate(faces):
        # Get the most common color for this face's vertices
        face_colors = [vertex_colors[v_idx] for v_idx in face if vertex_colors[v_idx] is not None]
        if not face_colors:
            continue
            
        # Simple majority vote for face color
        unique_colors, counts = np.unique(face_colors, axis=0, return_counts=True)
        face_color = tuple(unique_colors[np.argmax(counts)])
        
        if face_color not in face_groups:
            face_groups[face_color] = []
        face_groups[face_color].append(face_idx)
    
    # Convert lists to numpy arrays for each group
    for color in face_groups:
        face_groups[color] = np.array(face_groups[color])
    
    return vertices, faces, face_groups

def compute_bounding_box(vertices, padding=0.05):
    """
    Compute bounding box with optional padding
    """
    if len(vertices) == 0:
        return np.array([0, 0, 0]), np.array([0, 0, 0])
    
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    
    # Add padding
    bbox_size = max_vals - min_vals
    min_vals -= bbox_size * padding
    max_vals += bbox_size * padding
    
    return min_vals, max_vals

def render_normal_map(mesh, height=512, width=512, camera_position=None):
    """
    Render a normal map from a fixed perspective
    Returns: RGB normal map as numpy array where RGB channels correspond to XYZ normal components
    """
    # Create a scene
    scene = pyrender.Scene()
    
    # Create a mesh for rendering with normals
    mesh.vertex_normals  # Force computation of vertex normals
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(render_mesh)
    
    # Set up camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
    
    # Position camera to look at the mesh's center
    if camera_position is None:
        # Default camera position
        s = np.max(np.abs(mesh.bounds)) * 2.5
        camera_position = [0.0, 0.0, s]  # Front view
    
    camera_pose = np.array([
        [1.0, 0.0, 0.0, camera_position[0]],
        [0.0, 1.0, 0.0, camera_position[1]],
        [0.0, 0.0, 1.0, camera_position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    
    # Render
    r = pyrender.OffscreenRenderer(width, height)
    
    # Render the normal map directly
    # This sets up a special shader to visualize normals as colors
    normal_map, depth = r.render(scene, flags=pyrender.RenderFlags.FLAT)
    
    # Convert to normalized normal map (from 0-255 to 0-1 range)
    normal_map = normal_map / 255.0
    
    # Create a valid mask from the depth buffer
    valid_mask = depth > 0
    
    # Clean up
    r.delete()
    
    return normal_map, valid_mask

def create_body_part_mask(mesh, face_indices, height=512, width=512, camera_position=None):
    """
    Create a mask for a specific body part by rendering only those faces
    """
    # Create a new mesh containing only the faces for this body part
    part_mesh = mesh.submesh([face_indices])[0]
    
    # Render only this body part
    _, mask = render_normal_map(part_mesh, height, width, camera_position)
    
    return mask

def compute_l2_distance(normal_map1, normal_map2, mask=None):
    """
    Compute L2 distance between two normal maps
    If mask is provided, only calculates distance within the masked area
    """
    if mask is not None:
        # Apply the mask to both normal maps
        pixels1 = normal_map1[mask]
        pixels2 = normal_map2[mask]
        
        if len(pixels1) == 0:  # No pixels in mask
            return float('nan')
    else:
        # Use all pixels
        pixels1 = normal_map1.reshape(-1, 3)
        pixels2 = normal_map2.reshape(-1, 3)
    
    # Calculate the squared Euclidean (L2) distance
    squared_diff = np.sum((pixels1 - pixels2) ** 2, axis=1)
    mean_squared_diff = np.mean(squared_diff)
    
    # Return RMSE (Root Mean Squared Error)
    return math.sqrt(mean_squared_diff)

def process_file_pair(segmented_file, non_segmented_file, render_height=512, render_width=512):
    """
    Process a pair of files and compute L2 distance between normal maps
    """
    # Load segmented mesh with face groups
    seg_vertices, seg_faces, seg_face_groups = load_obj_with_face_groups(segmented_file)
    seg_mesh = trimesh.Trimesh(vertices=seg_vertices, faces=seg_faces)
    
    # Load non-segmented mesh
    non_seg_mesh = trimesh.load(non_segmented_file)
    
    # Make sure both meshes are centered and aligned
    # This ensures fair comparison between the normal maps
    seg_mesh.vertices -= seg_mesh.centroid
    non_seg_mesh.vertices -= non_seg_mesh.centroid
    
    # Scale both meshes to similar size
    seg_scale = 1.0 / np.max(np.abs(seg_mesh.vertices))
    non_seg_scale = 1.0 / np.max(np.abs(non_seg_mesh.vertices))
    seg_mesh.vertices *= seg_scale
    non_seg_mesh.vertices *= non_seg_scale
    
    # Define camera position for consistent rendering
    camera_position = [0.0, 0.0, 2.5]  # Front view
    
    # Render normal maps from fixed perspective
    seg_normal_map, seg_valid_mask = render_normal_map(
        seg_mesh, render_height, render_width, camera_position)
    non_seg_normal_map, non_seg_valid_mask = render_normal_map(
        non_seg_mesh, render_height, render_width, camera_position)
    
    # Combined mask of valid pixels in both renderings
    combined_mask = seg_valid_mask & non_seg_valid_mask
    
    # Save normal maps for visualization
    basename = os.path.basename(segmented_file).split('.')[0]
    seg_map_path = os.path.join(OUTPUT_DIR, f"{basename}_segmented_normal.png")
    non_seg_map_path = os.path.join(OUTPUT_DIR, f"{basename}_nonsegmented_normal.png")
    plt.imsave(seg_map_path, seg_normal_map)
    plt.imsave(non_seg_map_path, non_seg_normal_map)
    
    # Compute overall L2 distance
    overall_distance = compute_l2_distance(seg_normal_map, non_seg_normal_map, combined_mask)
    
    print(f"\n{basename}")
    print(f"Overall normal map L2 distance: {overall_distance:.4f} (0.0 is perfect)")
    
    # Results dictionary
    results = {'overall': overall_distance}
    
    # Calculate distance for each body part
    for color, face_indices in seg_face_groups.items():
        if color in BODY_PART_LABELS:
            part_name = BODY_PART_LABELS[color]
            color_name = COLOR_NAMES.get(color, "Unknown")
            
            # Create mask for this body part
            body_part_mask = create_body_part_mask(
                seg_mesh, face_indices, render_height, render_width, camera_position)
            
            # Combined mask for this body part (part & valid in both renderings)
            part_mask = body_part_mask & combined_mask
            
            # Compute L2 distance for this body part
            part_distance = compute_l2_distance(seg_normal_map, non_seg_normal_map, part_mask)
            results[color] = part_distance
            
            print(f"{part_name}: Normal map L2 distance: {part_distance:.4f} (0.0 is perfect)")
            
            # Save masked normal map difference for visualization
            diff_map = np.zeros_like(seg_normal_map)
            diff_map[part_mask] = np.abs(seg_normal_map[part_mask] - non_seg_normal_map[part_mask])
            diff_path = os.path.join(OUTPUT_DIR, f"{basename}_{color_name}_diff.png")
            plt.imsave(diff_path, diff_map)
    
    # Create and save overall difference map
    diff_map = np.zeros_like(seg_normal_map)
    diff_map[combined_mask] = np.abs(seg_normal_map[combined_mask] - non_seg_normal_map[combined_mask])
    diff_path = os.path.join(OUTPUT_DIR, f"{basename}_overall_diff.png")
    plt.imsave(diff_path, diff_map)
    
    return results

def main():
    # Get list of files in both directories
    segmented_files = [f for f in os.listdir(SEGMENTED_DIR) if f.endswith('.obj')]
    non_segmented_files = [f for f in os.listdir(NON_SEGMENTED_DIR) if f.endswith('.obj')]
    
    # Find common file names
    seg_basenames = {os.path.splitext(f)[0] for f in segmented_files}
    non_seg_basenames = {os.path.splitext(f)[0] for f in non_segmented_files}
    common_basenames = seg_basenames.intersection(non_seg_basenames)
    
    print(f"Found {len(common_basenames)} matching file pairs")
    
    # Process each file pair
    all_results = {}
    for basename in tqdm(sorted(common_basenames)):
        segmented_file = os.path.join(SEGMENTED_DIR, f"{basename}.obj")
        non_segmented_file = os.path.join(NON_SEGMENTED_DIR, f"{basename}.obj")
        
        try:
            results = process_file_pair(segmented_file, non_segmented_file)
            all_results[basename] = results
        except Exception as e:
            print(f"Error processing {basename}: {e}")
    
    # Compute average results across all files
    print("\n----- SUMMARY -----")
    print(f"Processed {len(all_results)} file pairs successfully")
    
    # Calculate overall average first
    overall_distances = [results.get('overall', float('nan')) for results in all_results.values()]
    valid_overall = [d for d in overall_distances if not np.isnan(d)]
    if valid_overall:
        avg_overall = np.mean(valid_overall)
        print(f"Average overall normal map L2 distance: {avg_overall:.4f} (0.0 is perfect)")
    
    # Aggregate results by body part
    for color, part_name in BODY_PART_LABELS.items():
        distances = [results.get(color, float('nan')) for results in all_results.values()]
        valid_distances = [d for d in distances if not np.isnan(d)]
        
        if valid_distances:
            avg_distance = np.mean(valid_distances)
            color_name = COLOR_NAMES.get(color, "Unknown")
            print(f"Average normal map L2 distance for {part_name} ({color_name}): {avg_distance:.4f}")

if __name__ == "__main__":
    main()