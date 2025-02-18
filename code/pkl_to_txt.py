import pickle
import numpy as np
import json

def convert_pkl_to_text(input_file, output_file):
    """
    Convert a pickle file to a human-readable text file.
    
    Args:
        input_file (str): Path to the input pickle file
        output_file (str): Path to the output text file
    """
    def numpy_to_list(obj):
        """
        Convert numpy arrays to lists for JSON serialization
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    try:
        # Read the pickle file
        with open(input_file, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # Convert numpy arrays to lists recursively
        converted_data = {k: numpy_to_list(v) for k, v in pkl_data.items()}
        
        # Convert to JSON with indentation for readability
        json_output = json.dumps(converted_data, indent=2)
        
        # Write to output text file
        with open(output_file, 'w') as f:
            f.write(json_output)
        
        print(f"Successfully converted {input_file} to {output_file}")
    
    except Exception as e:
        print(f"Error converting pickle file: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace with the actual path to your pickle file
    input_pickle_file = 'Semantic/labels/label-f00042.pkl'
    
    # Specify the desired output text file path
    output_text_file = 'label-f00042.txt'
    
    # Convert the file
    convert_pkl_to_text(input_pickle_file, output_text_file)