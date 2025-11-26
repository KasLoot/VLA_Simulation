import xml.etree.ElementTree as ET
import copy
import os
import mujoco

class SceneBuilder:
    def __init__(self, panda_xml_path, scene_xml_path):
        self.panda_xml_path = panda_xml_path
        self.scene_xml_path = scene_xml_path
        self.robots = []
        
        # Parse the base XMLs
        self.panda_tree = ET.parse(panda_xml_path)
        self.scene_tree = ET.parse(scene_xml_path)
        
    def add_robot(self, name, pos):
        """
        Add a robot to the scene.
        name: Unique prefix for the robot (e.g., "robot1")
        pos: "x y z" string or list of floats
        """
        if isinstance(pos, (list, tuple, np.ndarray)):
            pos = f"{pos[0]} {pos[1]} {pos[2]}"
        self.robots.append({'name': name, 'pos': pos})

    def _add_prefix(self, name, prefix):
        if name:
            return f"{prefix}_{name}"
        return name

    def build(self, save_built_scene_xml=False):
        """
        Constructs the combined XML.
        If save_built_scene_xml is True, saves to a file and loads from it.
        Otherwise, loads directly from the XML string.
        """
        panda_root = self.panda_tree.getroot()
        scene_root = copy.deepcopy(self.scene_tree.getroot()) # Work on a copy

        # Remove existing include of panda.xml if present
        for child in scene_root.findall('include'):
            if 'panda.xml' in child.get('file', ''):
                scene_root.remove(child)

        # Merge assets, defaults, compiler, option from panda.xml
        for tag in ['asset', 'default', 'compiler', 'option']:
            element = panda_root.find(tag)
            if element is not None:
                element = copy.deepcopy(element) # Deepcopy to avoid modifying original
                
                if tag in ['compiler', 'option']:
                    # These should be unique and at the top
                    existing = scene_root.find(tag)
                    if existing is None:
                        scene_root.insert(0, element)
                elif tag == 'asset':
                    # Merge assets
                    scene_asset = scene_root.find('asset')
                    if scene_asset is None:
                        scene_root.append(element)
                    else:
                        for asset_child in element:
                            scene_asset.append(asset_child)
                elif tag == 'default':
                    scene_root.append(element)

        # Update meshdir to absolute path to support loading from string
        compiler = scene_root.find('compiler')
        if compiler is not None and 'meshdir' in compiler.attrib:
            meshdir = compiler.get('meshdir')
            if not os.path.isabs(meshdir):
                abs_meshdir = os.path.abspath(os.path.join(os.path.dirname(self.panda_xml_path), meshdir))
                compiler.set('meshdir', abs_meshdir)

        # Prepare worldbody
        worldbody = scene_root.find('worldbody')
        if worldbody is None:
            worldbody = ET.SubElement(scene_root, 'worldbody')

        # Robot specific tags to clone
        robot_specific_tags = ['actuator', 'tendon', 'equality', 'contact', 'keyframe']
        
        # Find robot root body (assuming first body in panda.xml worldbody)
        panda_worldbody = panda_root.find('worldbody')
        robot_root_body = panda_worldbody.find('body')

        for robot in self.robots:
            prefix = robot['name']
            pos = robot['pos']
            
            # 1. Clone and rename body hierarchy
            new_body = copy.deepcopy(robot_root_body)
            new_body.set('pos', pos)
            
            self._recursive_rename(new_body, prefix)
            worldbody.append(new_body)

            # 2. Clone and rename other sections
            for tag in robot_specific_tags:
                section = panda_root.find(tag)
                if section is not None:
                    scene_section = scene_root.find(tag)
                    if scene_section is None:
                        scene_section = ET.SubElement(scene_root, tag)
                    
                    for item in section:
                        new_item = copy.deepcopy(item)
                        self._rename_attributes(new_item, prefix, tag)
                        scene_section.append(new_item)

        if save_built_scene_xml:
            # Save to a temporary file in the same directory as panda.xml 
            output_dir = os.path.dirname(self.panda_xml_path)
            output_path = os.path.join(output_dir, 'generated_multi_scene.xml')
            
            tree = ET.ElementTree(scene_root)
            if hasattr(ET, 'indent'):
                ET.indent(tree, space="  ", level=0)
                
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            print(f"Generated temporary scene file: {output_path}")
            
            try:
                model = mujoco.MjModel.from_xml_path(output_path)
                return model
            except Exception as e:
                print(f"Failed to load generated model from file: {e}")
                raise e
        else:
            # Load directly from string
            xml_string = ET.tostring(scene_root, encoding='unicode')
            try:
                model = mujoco.MjModel.from_xml_string(xml_string)
                return model
            except Exception as e:
                print(f"Failed to load generated model from string: {e}")
                raise e

    def _recursive_rename(self, element, prefix):
        if 'name' in element.attrib:
            element.set('name', self._add_prefix(element.get('name'), prefix))
        for child in element:
            self._recursive_rename(child, prefix)

    def _rename_attributes(self, element, prefix, tag):
        # Rename 'name'
        if 'name' in element.attrib:
            element.set('name', self._add_prefix(element.get('name'), prefix))
        
        # Update references
        if 'joint' in element.attrib:
            element.set('joint', self._add_prefix(element.get('joint'), prefix))
        if 'tendon' in element.attrib:
            element.set('tendon', self._add_prefix(element.get('tendon'), prefix))
        
        # Tag specific references
        if tag == 'tendon':
            for joint_ref in element.findall('joint'):
                if 'joint' in joint_ref.attrib:
                    joint_ref.set('joint', self._add_prefix(joint_ref.get('joint'), prefix))
        
        if tag == 'equality':
            if 'joint1' in element.attrib:
                element.set('joint1', self._add_prefix(element.get('joint1'), prefix))
            if 'joint2' in element.attrib:
                element.set('joint2', self._add_prefix(element.get('joint2'), prefix))
                
        if tag == 'contact':
            if 'body1' in element.attrib:
                element.set('body1', self._add_prefix(element.get('body1'), prefix))
            if 'body2' in element.attrib:
                element.set('body2', self._add_prefix(element.get('body2'), prefix))

import numpy as np
