import xml.etree.ElementTree as ET
import os
import glob

class ObjectLibrary:
    def __init__(self, objects_dir):
        self.objects_dir = objects_dir
        self.objects = {}
        self.available_objects, self.surfaces = self.get_available_objects()

    
    def get_available_objects(self):
        # Find all xml files in the directory
        xml_files = glob.glob(os.path.join(self.objects_dir, "*.xml"))
        available_objects = []
        available_surfaces = []
        for file_path in xml_files:
            with open(file_path, 'r') as f:
                content = f.read()
                wrapped_content = f"<root>{content}</root>"
                try:
                    root = ET.fromstring(wrapped_content)
                except ET.ParseError as e:
                    print(f"Failed to parse {file_path}: {e}")
                    continue
        
            # Find all bodies
            for body in root.findall(".//body"):
                name = body.get("name")
                class_name = body.get("class")
                if name:
                    if class_name == "object":
                        available_objects.append(name)
                        self.objects[name] = body
                    elif class_name == "surface":
                        available_surfaces.append(name)
                        self.objects[name] = body
        return available_objects, available_surfaces
    
    def get_object(self, name):
        return self.objects.get(name)


if __name__ == "__main__":
    obj_lib = ObjectLibrary("./objects")
    print("Available objects:", obj_lib.available_objects)
    print("Available surfaces:", obj_lib.surfaces)
    print("All loaded objects:", obj_lib.objects)