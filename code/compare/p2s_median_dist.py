import os
import numpy as np
import trimesh
from tqdm import tqdm

# Body part color mappings
BODY_PART_LABELS = {
    (0.0, 0.8, 0.8): "Legs",
    (1.0, 0.0, 1.0): "Bottom-Half Clothes",
    (0.0, 0.0, 1.0): "Shoes",
    (1.0, 1.0, 0.0): "Top-Half Clothes", 
    (1.0, 0.5, 0.0): "Hands",
    (0.0, 1.0, 0.0): "Hair",
    (1.0, 0.0, 0.0): "Head"
}

# Define directories directly in the script
SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files'
NON_SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_aligned_files'

def extract_labeled_parts(file_path):
    """
    Extract vertices for each labeled part based on RGB values from OBJ file
    """
    labeled_parts = {}
    all_vertices = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Get vertex coordinates
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                all_vertices.append(vertex)  # Store all vertices regardless of color
                
                # Get color values
                if len(parts) >= 7:  # Make sure color values exist
                    color = tuple(float(x) for x in parts[4:7])
                    
                    # Add vertex to corresponding color group
                    if color not in labeled_parts:
                        labeled_parts[color] = []
                    labeled_parts[color].append(vertex)
    
    # Convert lists to numpy arrays
    for color in labeled_parts:
        labeled_parts[color] = np.array(labeled_parts[color])
    
    return labeled_parts, np.array(all_vertices)

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

def compute_point_to_surface_distance(points, mesh):
    """
    Compute the minimum distance from each point to the mesh surface,
    excluding 5% of outliers (2.5% on each end)
    """
    if len(points) == 0:
        return float('inf')
    
    # Use trimesh's built-in function to find closest points on the mesh
    distances, _, _ = trimesh.proximity.closest_point(mesh, points)
    
    # Calculate 95% distribution by removing 5% from far end
    lower_bound = np.percentile(distances, 0)
    upper_bound = np.percentile(distances, 80)
    
    # Filter distances to exclude outliers
    filtered_distances = distances[(distances >= lower_bound) & (distances <= upper_bound)]
    
    # Return mean of the filtered distances (95% of data)
    return np.mean(filtered_distances)

def process_file_pair(segmented_file, non_segmented_file):
    """
    Process a pair of files and compute point-to-surface distances for each body part
    """
    # Extract labeled parts from segmented file
    labeled_parts, segmented_all_vertices = extract_labeled_parts(segmented_file)
    
    # Load non-segmented mesh
    non_segmented_mesh = trimesh.load(non_segmented_file)
    
    # Print only the filename
    print(f"\n{os.path.basename(segmented_file)}")
    print("Point-to-surface distances by body part:")
    
    # Calculate overall point-to-surface distance first
    overall_p2s_dist = compute_point_to_surface_distance(segmented_all_vertices, non_segmented_mesh)
    print(f"Overall shape: {len(segmented_all_vertices)} vertices, "
          f"Point-to-surface distance: {overall_p2s_dist:.6f} units")
    
    # Calculate point-to-surface distance for each labeled part
    results = {'overall': overall_p2s_dist}
    for color, part_vertices in labeled_parts.items():
        if color in BODY_PART_LABELS:
            part_name = BODY_PART_LABELS[color]
            
            # Compute point-to-surface distance for this part
            p2s_dist = compute_point_to_surface_distance(part_vertices, non_segmented_mesh)
            
            results[color] = p2s_dist
            print(f"{part_name}: {len(part_vertices)} vertices, "
                  f"Point-to-surface distance: {p2s_dist:.6f} units")
    
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
    for basename in tqdm(common_basenames):
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
    valid_overall_distances = [d for d in overall_distances if not np.isnan(d) and d != float('inf')]
    if valid_overall_distances:
        avg_overall_distance = np.mean(valid_overall_distances)
        print(f"Average overall point-to-surface distance: {avg_overall_distance:.6f} units")
    
    # Aggregate results by body part
    body_part_results = {}
    for color, part_name in BODY_PART_LABELS.items():
        distances = [results.get(color, float('nan')) for results in all_results.values()]
        valid_distances = [d for d in distances if not np.isnan(d) and d != float('inf')]
        
        if valid_distances:
            avg_distance = np.mean(valid_distances)
            body_part_results[part_name] = avg_distance
            print(f"Average point-to-surface distance for {part_name}: {avg_distance:.6f} units")

if __name__ == "__main__":
    main()