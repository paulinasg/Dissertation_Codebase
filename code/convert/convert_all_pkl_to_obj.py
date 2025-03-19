import os
import pickle
from pkl_to_obj import pkl_to_obj

def convert_all_pkl_to_obj(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    successful = 0
    failed = 0
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pkl'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.obj'
            output_path = os.path.join(output_folder, output_filename)

            try:
                # Load the .pkl file
                with open(input_path, 'rb') as f:
                    pkl_data = pickle.load(f)

                # Convert to .obj
                pkl_to_obj(pkl_data, output_path)
                print(f"Converted {input_path} to {output_path}")
                successful += 1
                
            except Exception as e:
                print(f"Error converting {input_path}: {str(e)}")
                failed += 1
    
    print(f"\nConversion complete. Successfully converted: {successful}, Failed: {failed}")

# Example usage:
input_folder = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/pkl_meshes/inner'
output_folder = '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_files/inner'
convert_all_pkl_to_obj(input_folder, output_folder)
