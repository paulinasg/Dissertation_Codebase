import os
import glob
import sys

def convert_red_to_teal(obj_file_path, output_file_path=None):
    """
    Simple utility to convert all red vertices to teal in an OBJ file.
    
    Args:
        obj_file_path: Path to the colored OBJ file
        output_file_path: Path to save the modified OBJ file (defaults to original with _fixed suffix)
    """
    if output_file_path is None:
        output_file_path = obj_file_path.replace(".obj", "_red2teal.obj")
    
    print(f"\nProcessing: {obj_file_path}")
    
    # Define colors
    red = (1.0, 0.0, 0.0)
    teal = (0.0, 0.8, 0.8)
    
    # Read the OBJ file
    with open(obj_file_path, "r") as obj_file:
        lines = obj_file.readlines()
    
    # Count red vertices
    red_count = 0
    modified_lines = []
    
    for line in lines:
        if line.startswith("v "):  # Vertex line
            parts = line.strip().split()
            
            if len(parts) >= 7:  # Vertex with color
                x, y, z = parts[1:4]
                r, g, b = map(float, parts[4:7])
                
                # Check if this is a red vertex (RGB: 1.0, 0.0, 0.0)
                if abs(r - red[0]) < 0.01 and abs(g - red[1]) < 0.01 and abs(b - red[2]) < 0.01:
                    # Replace with teal
                    modified_line = f"v {x} {y} {z} {teal[0]} {teal[1]} {teal[2]}\n"
                    red_count += 1
                else:
                    modified_line = line
            else:
                modified_line = line
        else:
            modified_line = line
            
        modified_lines.append(modified_line)
    
    # Write the modified OBJ file
    with open(output_file_path, "w") as out_file:
        out_file.writelines(modified_lines)
    
    print(f"Converted {red_count} red vertices to teal")
    print(f"Modified file saved to: {output_file_path}")

def process_files(directory_pattern):
    """Process multiple files matching the given pattern"""
    files = glob.glob(directory_pattern)
    print(f"Found {len(files)} files to process")
    
    for file_path in files:
        convert_red_to_teal(file_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process file or pattern provided as command line argument
        input_path = sys.argv[1]
        if '*' in input_path:  # It's a pattern
            process_files(input_path)
        else:  # It's a single file
            convert_red_to_teal(input_path)
    else:
        # Default file path
        obj_file_path = "/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_labelled_files/inner/00140_fixed.obj"
        convert_red_to_teal(obj_file_path)