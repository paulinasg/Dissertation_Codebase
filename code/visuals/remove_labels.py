import os

def remove_color_labels(input_file_path, output_file_path):
    """
    Remove color labels from OBJ file vertices while preserving geometry.
    
    Args:
        input_file_path (str): Path to input OBJ file
        output_file_path (str): Path to output OBJ file
    """
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            if line.startswith('v '):
                # Split the line into parts
                parts = line.strip().split()
                # Keep only the vertex coordinates (first 4 parts: 'v' and xyz)
                new_line = ' '.join(parts[:4]) + '\n'
                outfile.write(new_line)
            else:
                # Write all other lines as they are
                outfile.write(line)

def main():
    # Example usage
    input_dir = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files'
    output_dir = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/visuals'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all OBJ files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.obj'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                remove_color_labels(input_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()