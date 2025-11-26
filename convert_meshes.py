
import trimesh
import os
import glob

# Path to the visual meshes
mesh_dir = "models/panda_description/meshes/visual"
files = glob.glob(os.path.join(mesh_dir, "*.dae"))

print(f"Found {len(files)} DAE files in {mesh_dir}")

for file_path in files:
    try:
        print(f"Converting {file_path}...")
        # Load the mesh
        mesh = trimesh.load(file_path)
        
        # Create new filename with .stl extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_path = os.path.join(mesh_dir, base_name + ".stl")
        
        # Export
        mesh.export(new_path)
        print(f"Saved to {new_path}")
        
    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")
