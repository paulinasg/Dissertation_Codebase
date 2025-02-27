import numpy as np

def pkl_to_obj(pkl_data, obj_filename):
    # Extract the relevant data from the dictionary
    vertices = pkl_data['vertices']
    faces = pkl_data['faces']
    normals = pkl_data['normals']
    uvs = pkl_data['uvs']

    with open(obj_filename, 'w') as f:
        # Write vertices (v)
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write normals (vn), if available
        for normal in normals:
            f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
        
        # Write texture coordinates (vt), if available
        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")
        
        # Write faces (f)
        for face in faces:
            # OBJ face format uses 1-based indexing, while your data seems to use 0-based indexing
            f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n")

# Example usage:
# Assuming pkl_data is the dictionary you provided (the content of the pkl file)
# Load your .pkl file first, e.g., using pickle.load
import pickle
with open('/Users/paulinagerchuk/Downloads/dataset-segment-analyse/pkl_meshes/00163.pkl', 'rb') as f:
    pkl_data = pickle.load(f)

# Convert to .obj
pkl_to_obj(pkl_data, '/Users/paulinagerchuk/Downloads/dataset-segment-analyse/obj_4ddress_files/00134.obj')