import os
import numpy as np
import trimesh
import pyvista as pv
import csv
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

COLOR_NAMES = {
    (0.0, 0.8, 0.8): "Cyan",
    (1.0, 0.0, 1.0): "Magenta",
    (0.0, 0.0, 1.0): "Blue",
    (1.0, 1.0, 0.0): "Yellow", 
    (1.0, 0.5, 0.0): "Orange",
    (0.0, 1.0, 0.0): "Green",
    (1.0, 0.0, 0.0): "Red"
}

# Define directories directly in the script
SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/inner'
NON_SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_aligned_files/inner'

def load_obj_with_colors(file_path):
    """
    Custom loader for OBJ files with vertex colors.
    
    Args:
        file_path: Path to the OBJ file
    
    Returns:
        vertices: Numpy array of vertices
        colors: Dictionary mapping colors to arrays of vertices
    """
    vertices = []
    colors_dict = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Get vertex coordinates
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
                
                # Get color values if they exist
                if len(parts) >= 7:
                    color = tuple(float(x) for x in parts[4:7])
                    
                    # Add vertex to corresponding color group
                    if color not in colors_dict:
                        colors_dict[color] = []
                    colors_dict[color].append(vertex)
    
    # Convert lists to numpy arrays
    vertices = np.array(vertices)
    for color in colors_dict:
        colors_dict[color] = np.array(colors_dict[color])
    
    return vertices, colors_dict

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

def compute_mesh_distances(target_vertices, source_mesh):
    """
    Compute point-to-surface distances from vertices to the surface of source_mesh.
    Using the same approach as in the first script.
    
    Args:
        target_vertices: Array of vertices to measure from
        source_mesh: The reference mesh we're measuring to (trimesh.Trimesh)
    
    Returns:
        distances: Array of distances for each vertex
        stats: Dictionary with statistical metrics
    """
    if len(target_vertices) == 0:
        return np.array([]), {"min": float('inf'), "max": 0, "mean": 0, "median": 0, "std": 0}
    
    # Use trimesh's proximity query for efficient distance computation
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(source_mesh, target_vertices)
    
    # Calculate statistics
    stats = {
        "min": distances.min(),
        "max": distances.max(),
        "mean": distances.mean(),
        "median": np.median(distances),
        "std": distances.std()
    }
    
    return distances, stats

def process_file_pair(segmented_file, non_segmented_file):
    """
    Process a pair of files and compute point-to-surface distances for each body part
    using the same methodology as the first script.
    """
    print(f"\nProcessing: {os.path.basename(segmented_file)}")
    
    try:
        # Load the segmented mesh with colors
        all_vertices, labeled_parts = load_obj_with_colors(segmented_file)
        
        # Load non-segmented mesh using trimesh for efficient distance calculations
        non_segmented_mesh = trimesh.load(non_segmented_file)
        
        # Calculate overall point-to-surface distance first
        overall_distances, overall_stats = compute_mesh_distances(all_vertices, non_segmented_mesh)
        
        print(f"Overall shape: {len(all_vertices)} vertices")
        print(f"  Min: {overall_stats['min']:.6f}")
        print(f"  Max: {overall_stats['max']:.6f}")
        print(f"  Mean: {overall_stats['mean']:.6f}")
        print(f"  Median: {overall_stats['median']:.6f}")
        print(f"  Std Dev: {overall_stats['std']:.6f}")
        
        # Calculate point-to-surface distance for each labeled part
        results = {'overall': overall_stats}
        
        for color, part_vertices in labeled_parts.items():
            if color in BODY_PART_LABELS:
                part_name = BODY_PART_LABELS[color]
                
                # Compute point-to-surface distance for this part
                part_distances, part_stats = compute_mesh_distances(part_vertices, non_segmented_mesh)
                
                results[color] = part_stats
                print(f"{part_name}: {len(part_vertices)} vertices")
                print(f"  Min: {part_stats['min']:.6f}")
                print(f"  Max: {part_stats['max']:.6f}")
                print(f"  Mean: {part_stats['mean']:.6f}")
                print(f"  Median: {part_stats['median']:.6f}")
                print(f"  Std Dev: {part_stats['std']:.6f}")
        
        return results
    
    except Exception as e:
        print(f"Error processing {os.path.basename(segmented_file)}: {e}")
        return None

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
        
        results = process_file_pair(segmented_file, non_segmented_file)
        if results:
            all_results[basename] = results
    
    # Compute average results across all files
    print("\n----- SUMMARY -----")
    print(f"Processed {len(all_results)} file pairs successfully")
    
    # Calculate overall average statistics
    overall_means = [results['overall']['mean'] for results in all_results.values() if results and 'overall' in results]
    overall_medians = [results['overall']['median'] for results in all_results.values() if results and 'overall' in results]
    overall_stds = [results['overall']['std'] for results in all_results.values() if results and 'overall' in results]

    if overall_means:
        print(f"Average overall mean distance: {np.mean(overall_means):.6f} units")
        print(f"Average overall median distance: {np.mean(overall_medians):.6f} units")
        print(f"Average overall standard deviation: {np.mean(overall_stds):.6f} units")
        print(f"Standard deviation of mean distances: {np.std(overall_means):.6f} units")
    
    # Aggregate results by body part
    print("\n----- BODY PART STATISTICS -----")
    for color, part_name in BODY_PART_LABELS.items():
        part_means = [results.get(color, {}).get('mean', float('nan')) for results in all_results.values()]
        part_medians = [results.get(color, {}).get('median', float('nan')) for results in all_results.values()]
        part_stds = [results.get(color, {}).get('std', float('nan')) for results in all_results.values()]
        
        valid_means = [d for d in part_means if not np.isnan(d) and d != float('inf')]
        valid_medians = [d for d in part_medians if not np.isnan(d) and d != float('inf')]
        valid_stds = [d for d in part_stds if not np.isnan(d) and d != float('inf')]
        
        if valid_means and valid_medians:
            print(f"{part_name}:")
            print(f"  Average mean distance: {np.mean(valid_means):.6f} units")
            print(f"  Average median distance: {np.mean(valid_medians):.6f} units") 
            print(f"  Average standard deviation: {np.mean(valid_stds):.6f} units")
            print(f"  Standard deviation of mean distances: {np.std(valid_means):.6f} units")
    
    # Find models with min/max point-to-surface distances
    print("\n----- MIN/MAX VALUES -----")
    
    # Overall min/max
    min_overall = {'value': float('inf'), 'file': None}
    max_overall = {'value': float('-inf'), 'file': None}
    
    for basename, results in all_results.items():
        if 'overall' in results:
            mean_value = results['overall'].get('mean', float('nan'))
            if not (np.isnan(mean_value) or mean_value == float('inf') or mean_value == float('-inf')):
                if mean_value < min_overall['value']:
                    min_overall['value'] = mean_value
                    min_overall['file'] = basename
                if mean_value > max_overall['value']:
                    max_overall['value'] = mean_value
                    max_overall['file'] = basename
    
    if min_overall['file'] is not None:
        print(f"Lowest overall mean distance: {min_overall['value']:.6f} units (Model: {min_overall['file']})")
    if max_overall['file'] is not None:
        print(f"Highest overall mean distance: {max_overall['value']:.6f} units (Model: {max_overall['file']})")
    
    # Body part min/max
    for color, part_name in BODY_PART_LABELS.items():
        min_part = {'value': float('inf'), 'file': None}
        max_part = {'value': float('-inf'), 'file': None}
        
        for basename, results in all_results.items():
            if color in results:
                part_value = results[color].get('mean', float('nan'))
                if not (np.isnan(part_value) or part_value == float('inf') or part_value == float('-inf')):
                    if part_value < min_part['value']:
                        min_part['value'] = part_value
                        min_part['file'] = basename
                    if part_value > max_part['value']:
                        max_part['value'] = part_value
                        max_part['file'] = basename
        
        if min_part['file'] is not None and max_part['file'] is not None:
            color_name = COLOR_NAMES.get(color, "Unknown")
            print(f"{part_name} ({color_name}):")
            print(f"  Lowest mean distance: {min_part['value']:.6f} units (Model: {min_part['file']})")
            print(f"  Highest mean distance: {max_part['value']:.6f} units (Model: {max_part['file']})")
    
    # Save all results to CSV files
    save_results_to_csv(all_results)

def save_results_to_csv(all_results):
    """Save the distance measurement results to CSV files."""
    output_dir = os.path.join(os.path.dirname(SEGMENTED_DIR), "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV for detailed per-model, per-part results
    detailed_csv_path = os.path.join(output_dir, "detailed_distance_measurements.csv")
    with open(detailed_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model_id', 'body_part', 'color', 'min', 'max', 'mean', 'median', 'std_dev', 'vertex_count'])
        
        # Write row for each model and body part
        for model_id, results in all_results.items():
            # Write overall stats first
            if 'overall' in results:
                writer.writerow([
                    model_id, 
                    'overall', 
                    'NA',
                    results['overall'].get('min', 'NA'),
                    results['overall'].get('max', 'NA'),
                    results['overall'].get('mean', 'NA'),
                    results['overall'].get('median', 'NA'),
                    results['overall'].get('std', 'NA'),
                    'NA'  # We don't track vertex count for overall
                ])
            
            # Write stats for each body part
            for color, part_stats in results.items():
                if color == 'overall':
                    continue
                
                part_name = BODY_PART_LABELS.get(color, "Unknown")
                color_str = f"{color[0]},{color[1]},{color[2]}"  # Convert tuple to string
                
                writer.writerow([
                    model_id,
                    part_name,
                    color_str,
                    part_stats.get('min', 'NA'),
                    part_stats.get('max', 'NA'),
                    part_stats.get('mean', 'NA'),
                    part_stats.get('median', 'NA'),
                    part_stats.get('std', 'NA'),
                    part_stats.get('count', 'NA')  # This might be NA if not tracked
                ])
    
    # CSV for aggregate statistics by body part
    aggregate_csv_path = os.path.join(output_dir, "aggregate_body_part_statistics.csv")
    with open(aggregate_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['body_part', 'avg_mean', 'avg_median', 'avg_std_dev', 'std_of_means', 'min_model', 'min_value', 'max_model', 'max_value'])
        
        # First write overall statistics
        overall_means = [results['overall']['mean'] for results in all_results.values() if results and 'overall' in results]
        overall_medians = [results['overall']['median'] for results in all_results.values() if results and 'overall' in results]
        overall_stds = [results['overall']['std'] for results in all_results.values() if results and 'overall' in results]
        
        valid_means = [d for d in overall_means if not np.isnan(d) and d != float('inf')]
        valid_medians = [d for d in overall_medians if not np.isnan(d) and d != float('inf')]
        valid_stds = [d for d in overall_stds if not np.isnan(d) and d != float('inf')]
        
        # Find min/max models for overall
        min_overall = {'value': float('inf'), 'file': None}
        max_overall = {'value': float('-inf'), 'file': None}
        
        for basename, results in all_results.items():
            if 'overall' in results:
                mean_value = results['overall'].get('mean', float('nan'))
                if not (np.isnan(mean_value) or mean_value == float('inf') or mean_value == float('-inf')):
                    if mean_value < min_overall['value']:
                        min_overall['value'] = mean_value
                        min_overall['file'] = basename
                    if mean_value > max_overall['value']:
                        max_overall['value'] = mean_value
                        max_overall['file'] = basename
        
        if valid_means:
            writer.writerow([
                'overall',
                np.mean(valid_means),
                np.mean(valid_medians),
                np.mean(valid_stds),
                np.std(valid_means),
                min_overall['file'],
                min_overall['value'],
                max_overall['file'],
                max_overall['value']
            ])
        
        # Then write statistics for each body part
        for color, part_name in BODY_PART_LABELS.items():
            part_means = [results.get(color, {}).get('mean', float('nan')) for results in all_results.values()]
            part_medians = [results.get(color, {}).get('median', float('nan')) for results in all_results.values()]
            part_stds = [results.get(color, {}).get('std', float('nan')) for results in all_results.values()]
            
            valid_means = [d for d in part_means if not np.isnan(d) and d != float('inf')]
            valid_medians = [d for d in part_medians if not np.isnan(d) and d != float('inf')]
            valid_stds = [d for d in part_stds if not np.isnan(d) and d != float('inf')]
            
            # Find min/max models for this part
            min_part = {'value': float('inf'), 'file': None}
            max_part = {'value': float('-inf'), 'file': None}
            
            for basename, results in all_results.items():
                if color in results:
                    part_value = results[color].get('mean', float('nan'))
                    if not (np.isnan(part_value) or part_value == float('inf') or part_value == float('-inf')):
                        if part_value < min_part['value']:
                            min_part['value'] = part_value
                            min_part['file'] = basename
                        if part_value > max_part['value']:
                            max_part['value'] = part_value
                            max_part['file'] = basename
            
            if valid_means:
                writer.writerow([
                    part_name,
                    np.mean(valid_means),
                    np.mean(valid_medians),
                    np.mean(valid_stds),
                    np.std(valid_means),
                    min_part['file'],
                    min_part['value'],
                    max_part['file'],
                    max_part['value']
                ])
    
    print(f"\nResults saved to CSV files:")
    print(f"- Detailed measurements: {detailed_csv_path}")
    print(f"- Aggregate statistics: {aggregate_csv_path}")

if __name__ == "__main__":
    main()