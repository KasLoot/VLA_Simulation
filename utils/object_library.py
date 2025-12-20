import xml.etree.ElementTree as ET
import os
import glob

class ObjectLibrary:
    def __init__(self, objects_dir):
        self.objects_dir = objects_dir
        self.objects = {}
        self._load_objects()
        print(f"self.objects: {self.objects}")

    def _load_objects(self):
        # Find all xml files in the directory
        xml_files = glob.glob(os.path.join(self.objects_dir, "*.xml"))
        print(f"xml_files: {xml_files}")
        
        for file_path in xml_files:
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # If the root is a body, add it (though usually root is mujoco or similar, 
                # but here the snippets seem to be just lists of bodies? 
                # The provided attachments show snippets like <body ...> </body> <body ...> </body>
                # These are not valid XML documents on their own if they have multiple roots without a single parent.
                # I will assume they might be wrapped or I need to handle multiple root elements if it's a fragment file.
                # Actually, standard XML parsers fail on multiple roots. 
                # I'll assume the user provided files are wrapped in a root tag or I should wrap them.
                # Let's check the file content provided in attachments.
                # They look like fragments. I might need to wrap them to parse.
                pass
            except ET.ParseError:
                # Try wrapping in a fake root
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
                if name:
                    self.objects[name] = body

    def get_object(self, name):
        return self.objects.get(name)
