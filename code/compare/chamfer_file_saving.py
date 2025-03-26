import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm
import csv

# Body part color mappings
BODY_PART_LABELS = {
    (0.0, 0.8, 0.8): "Legs",
    (1.0, 0.0, 1.0): "Bottom-Half Clothes",
    (0.0, 0.0, 1.0): "Shoes",
    (1.0, 1.0, 0.0): "Top-Half Clothes", 
    (1.0, 0.5, 0.0): "Head",
    (0.0, 1.0, 0.0): "Hair",
    (1.0, 0.0, 0.0): "Hands"
}

# Add this dictionary after the BODY_PART_LABELS definition
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
SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/outer'
NON_SEGMENTED_DIR = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_pifuhd_aligned_files/outer'

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

def filter_vertices_in_bbox(vertices, bbox_min, bbox_max):
    """
    Filter vertices that fall within a bounding box
    """
    mask = np.all((vertices >= bbox_min) & (vertices <= bbox_max), axis=1)
    return vertices[mask]

def compute_chamfer_distance(points1, points2):
    """
    Compute chamfer distance between two point clouds
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # Build kd-trees for both point sets
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Find nearest neighbors in both directions
    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)
    
    # Compute chamfer distance
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return chamfer_dist / 2.0

def process_file_pair(segmented_file, non_segmented_file):
    """
    Process a pair of files and compute chamfer distances for each body part
    """
    # Extract labeled parts from segmented file
    labeled_parts, segmented_all_vertices = extract_labeled_parts(segmented_file)
    
    # Load non-segmented mesh
    non_segmented_mesh = trimesh.load(non_segmented_file)
    non_segmented_vertices = np.array(non_segmented_mesh.vertices)
    
    # Print only the filename
    print(f"\n{os.path.basename(segmented_file)}")
    print("Chamfer distances by body part:")
    
    # Calculate overall chamfer distance first
    overall_chamfer_dist = compute_chamfer_distance(segmented_all_vertices, non_segmented_vertices)
    print(f"Overall shape: {len(segmented_all_vertices)} vertices, {len(non_segmented_vertices)} vertices, "
          f"Chamfer distance: {overall_chamfer_dist:.6f} units")
    
    # Calculate chamfer distance for each labeled part
    results = {'overall': overall_chamfer_dist}
    for color, part_vertices in labeled_parts.items():
        if color in BODY_PART_LABELS:
            part_name = BODY_PART_LABELS[color]
            
            # Compute bounding box for this part
            bbox_min, bbox_max = compute_bounding_box(part_vertices)
            
            # Filter non-segmented vertices to those within the bounding box
            filtered_vertices = filter_vertices_in_bbox(non_segmented_vertices, bbox_min, bbox_max)
            
            # Skip if no vertices in bounding box
            if len(filtered_vertices) == 0:
                continue
            
            # Compute chamfer distance between part and filtered vertices
            chamfer_dist = compute_chamfer_distance(part_vertices, filtered_vertices)
            
            results[color] = chamfer_dist
            print(f"{part_name}: {len(part_vertices)} vertices, {len(filtered_vertices)} in bbox, "
                  f"Chamfer distance: {chamfer_dist:.6f} units")
    
    return results

def save_results_to_csv(all_results):
    """Save the chamfer distance results to CSV files."""
    output_dir = os.path.join(os.path.dirname(SEGMENTED_DIR), "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV for detailed per-model, per-part results
    detailed_csv_path = os.path.join(output_dir, "detailed_chamfer_measurements_outer.csv")
    with open(detailed_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model_id', 'body_part', 'color', 'chamfer_distance'])
        
        # Write row for each model and body part
        for model_id, results in all_results.items():
            # Write overall value first
            writer.writerow([
                model_id, 
                'overall', 
                'NA',
                results.get('overall', 'NA')
            ])
            
            # Write values for each body part
            for color, chamfer_dist in results.items():
                if color == 'overall':
                    continue
                
                part_name = BODY_PART_LABELS.get(color, "Unknown")
                color_str = f"{color[0]},{color[1]},{color[2]}"  # Convert tuple to string
                
                writer.writerow([
                    model_id,
                    part_name,
                    color_str,
                    chamfer_dist
                ])
    
    # CSV for aggregate statistics by body part
    aggregate_csv_path = os.path.join(output_dir, "aggregate_chamfer_statistics_outer.csv")
    with open(aggregate_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['body_part', 'avg_distance', 'std_dev', 'min_model', 'min_value', 'max_model', 'max_value'])
        
        # First write overall statistics
        overall_distances = [results.get('overall', float('nan')) for results in all_results.values()]
        valid_overall_distances = [d for d in overall_distances if not np.isnan(d) and d != float('inf')]
        
        # Find min/max models for overall
        min_overall = {'value': float('inf'), 'file': None}
        max_overall = {'value': float('-inf'), 'file': None}
        
        for basename, results in all_results.items():
            overall_value = results.get('overall', float('nan'))
            if not (np.isnan(overall_value) or overall_value == float('inf') or overall_value == float('-inf')):
                if overall_value < min_overall['value']:
                    min_overall['value'] = overall_value
                    min_overall['file'] = basename
                if overall_value > max_overall['value']:
                    max_overall['value'] = overall_value
                    max_overall['file'] = basename
        
        if valid_overall_distances:
            writer.writerow([
                'overall',
                np.mean(valid_overall_distances),
                np.std(valid_overall_distances),
                min_overall['file'],
                min_overall['value'],
                max_overall['file'],
                max_overall['value']
            ])
        
        # Then write statistics for each body part
        for color, part_name in BODY_PART_LABELS.items():
            distances = [results.get(color, float('nan')) for results in all_results.values()]
            valid_distances = [d for d in distances if not np.isnan(d) and d != float('inf')]
            
            # Find min/max models for this part
            min_part = {'value': float('inf'), 'file': None}
            max_part = {'value': float('-inf'), 'file': None}
            
            for basename, results in all_results.items():
                part_value = results.get(color, float('nan'))
                if not (np.isnan(part_value) or part_value == float('inf') or part_value == float('-inf')):
                    if part_value < min_part['value']:
                        min_part['value'] = part_value
                        min_part['file'] = basename
                    if part_value > max_part['value']:
                        max_part['value'] = part_value
                        max_part['file'] = basename
            
            if valid_distances:
                writer.writerow([
                    part_name,
                    np.mean(valid_distances),
                    np.std(valid_distances),
                    min_part['file'],
                    min_part['value'],
                    max_part['file'],
                    max_part['value']
                ])
    
    print(f"\nResults saved to CSV files:")
    print(f"- Detailed measurements: {detailed_csv_path}")
    print(f"- Aggregate statistics: {aggregate_csv_path}")

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
    
    # Calculate overall average and standard deviation first
    overall_distances = [results.get('overall', float('nan')) for results in all_results.values()]
    valid_overall_distances = [d for d in overall_distances if not np.isnan(d) and d != float('inf')]
    if valid_overall_distances:
        avg_overall_distance = np.mean(valid_overall_distances)
        std_overall_distance = np.std(valid_overall_distances)
        print(f"Average overall chamfer distance: {avg_overall_distance:.6f} units, Std Dev: {std_overall_distance:.6f} units")
    
    # Aggregate results by body part
    body_part_results = {}
    for color, part_name in BODY_PART_LABELS.items():
        distances = [results.get(color, float('nan')) for results in all_results.values()]
        valid_distances = [d for d in distances if not np.isnan(d) and d != float('inf')]
        
        if valid_distances:
            avg_distance = np.mean(valid_distances)
            std_distance = np.std(valid_distances)
            body_part_results[part_name] = {'avg': avg_distance, 'std': std_distance}
            color_name = COLOR_NAMES.get(color, "Unknown")
            print(f"Average chamfer distance for {part_name} ({color_name}, RGB: {color}): {avg_distance:.6f} units, Std Dev: {std_distance:.6f} units")
    
    # Find models with min/max chamfer distances
    print("\n----- MIN/MAX VALUES -----")
    
    # Overall min/max
    min_overall = {'value': float('inf'), 'file': None}
    max_overall = {'value': float('-inf'), 'file': None}
    
    for basename, results in all_results.items():
        overall_value = results.get('overall', float('nan'))
        if not (np.isnan(overall_value) or overall_value == float('inf') or overall_value == float('-inf')):
            if overall_value < min_overall['value']:
                min_overall['value'] = overall_value
                min_overall['file'] = basename
            if overall_value > max_overall['value']:
                max_overall['value'] = overall_value
                max_overall['file'] = basename
    
    if min_overall['file'] is not None:
        print(f"Lowest overall chamfer distance: {min_overall['value']:.6f} units (Model: {min_overall['file']})")
    if max_overall['file'] is not None:
        print(f"Highest overall chamfer distance: {max_overall['value']:.6f} units (Model: {max_overall['file']})")
    
    # Body part min/max
    for color, part_name in BODY_PART_LABELS.items():
        min_part = {'value': float('inf'), 'file': None}
        max_part = {'value': float('-inf'), 'file': None}
        
        for basename, results in all_results.items():
            part_value = results.get(color, float('nan'))
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
            print(f"  Lowest chamfer distance: {min_part['value']:.6f} units (Model: {min_part['file']})")
            print(f"  Highest chamfer distance: {max_part['value']:.6f} units (Model: {max_part['file']})")

    # Save results to CSV
    save_results_to_csv(all_results)

if __name__ == "__main__":
    main()