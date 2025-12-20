import xml.etree.ElementTree as ET
import copy
import os
import mujoco
import numpy as np
import json
import random
from utils.objects import ObjectLibrary

class EnvironmentBuilder:
    def __init__(self, 
                 robot_xml_path, 
                 env_xml_path, 
                 robot_config, 
                 env_config, 
                 scene_json_path=None, 
                 objects_dir=None,
                 seed=None):
        self.robot_xml_path = robot_xml_path
        self.env_xml_path = env_xml_path
        self.robot_config = robot_config
        self.env_config = env_config
        self.scene_json_path = scene_json_path
        self.objects_dir = objects_dir
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Parse the base XMLs
        self.robot_tree = ET.parse(robot_xml_path)
        self.env_tree = ET.parse(env_xml_path)
        
        # Load Object Library if provided
        self.object_library = None
        if objects_dir:
            self.object_library = ObjectLibrary(objects_dir)
            
        # Load Scene JSON if provided
        self.scene_data = {}
        if scene_json_path and os.path.exists(scene_json_path):
            with open(scene_json_path, 'r') as f:
                data = json.load(f)
                # Convert list to dict keyed by robot_index
                for item in data:
                    self.scene_data[item['robot_index']] = item

        
    def build(self, save_path: str = None):
        robot_root = self.robot_tree.getroot()
        env_root = self.env_tree.getroot()
        
        # 1. Merge Assets and Defaults
        self._merge_section(env_root, robot_root, 'asset')
        self._merge_section(env_root, robot_root, 'default')
        self._merge_compiler(env_root, robot_root)
        
        # 2. Remove existing includes that might conflict
        for include in env_root.findall('include'):
            if 'panda.xml' in include.get('file', ''):
                env_root.remove(include)

        # 3. Get the robot body
        robot_worldbody = robot_root.find('worldbody')
        if robot_worldbody is None:
            raise ValueError("Robot XML must have a worldbody")
        
        # Find the main robot body. It might be the first body.
        robot_body = robot_worldbody.find('body')
        if robot_body is None:
             raise ValueError("Robot XML worldbody must have a body")

        env_worldbody = env_root.find('worldbody')
        if env_worldbody is None:
            env_worldbody = ET.SubElement(env_root, 'worldbody')

        # 4. Loop to create robots
        num_robots = self.robot_config.quantities[0]
        max_per_row = self.env_config.maximum_robots_per_row
        row_spacing = self.env_config.row_spacing
        col_spacing = self.env_config.column_spacing
        
        # Handle spacing if it's a list or float
        if isinstance(row_spacing, list): row_spacing = row_spacing[0]
        if isinstance(col_spacing, list): col_spacing = col_spacing[0]

        for i in range(num_robots):
            # Calculate position
            row = i // max_per_row
            col = i % max_per_row
            x = col * col_spacing
            y = row * row_spacing
            
            # Create container body
            container_name = f"robot_{i}"
            container_body = ET.Element('body', {'name': container_name, 'pos': f"{x} {y} 0"})
            
            # Copy childclass from robot body to container if it exists
            if 'childclass' in robot_body.attrib:
                container_body.set('childclass', robot_body.get('childclass'))

            # Clone body
            new_body = copy.deepcopy(robot_body)
            prefix = f"robot_{i}/"
            
            # Rename body and children
            self._rename_recursively(new_body, prefix)
            
            # Reset position of the robot base to 0 0 0 (or keep original if it wasn't 0 0 0)
            # The container handles the grid position now.
            # If the original body had a pos, we keep it (it's a local offset).
            # If it didn't, we don't add one (defaults to 0 0 0).
            # In the previous version, we were overwriting it.
            # Here we just leave it as is (which is the cloned value).
            
            container_body.append(new_body)
            
            env_worldbody.append(container_body)

            # Add Scene Objects if configured
            if i in self.scene_data and self.object_library:
                # Calculate robot global position
                robot_pos = [x, y, 0.0]
                self._add_scene_to_robot(container_body, env_worldbody, robot_pos, self.scene_data[i], prefix)
            
            # Clone and rename actuators, sensors, etc.
            for tag in ['actuator', 'sensor', 'tendon', 'equality', 'contact']:
                section = robot_root.find(tag)
                if section is not None:
                    env_section = env_root.find(tag)
                    if env_section is None:
                        env_section = ET.SubElement(env_root, tag)
                    
                    for item in section:
                        new_item = copy.deepcopy(item)
                        self._rename_attributes(new_item, prefix)
                        env_section.append(new_item)

        if save_path:
            tree = ET.ElementTree(env_root)
            if hasattr(ET, 'indent'):
                ET.indent(tree, space="  ", level=0)
            tree.write(save_path, encoding="utf-8", xml_declaration=True)
            print(f"Environment saved to {save_path}")

        return env_root

    def _merge_section(self, target_root, source_root, tag):
        source_section = source_root.find(tag)
        if source_section is not None:
            target_section = target_root.find(tag)
            if target_section is None:
                target_section = ET.SubElement(target_root, tag)
            # Append children from source to target
            for child in source_section:
                target_section.append(copy.deepcopy(child))

    def _merge_compiler(self, target_root, source_root):
        source_compiler = source_root.find('compiler')
        if source_compiler is not None:
            target_compiler = target_root.find('compiler')
            if target_compiler is None:
                # Insert compiler at the beginning (usually better style)
                target_compiler = ET.Element('compiler')
                target_root.insert(0, target_compiler)
            
            # Copy attributes
            for key, value in source_compiler.attrib.items():
                if key == 'meshdir':
                    # Resolve meshdir relative to robot xml
                    robot_dir = os.path.dirname(self.robot_xml_path)
                    abs_meshdir = os.path.abspath(os.path.join(robot_dir, value))
                    target_compiler.set(key, abs_meshdir)
                else:
                    if key not in target_compiler.attrib:
                        target_compiler.set(key, value)

    def _rename_recursively(self, element, prefix):
        # Rename 'name' attribute
        if 'name' in element.attrib:
            element.set('name', prefix + element.get('name'))
        
        for child in element:
            self._rename_recursively(child, prefix)

    def _rename_attributes(self, element, prefix):
        # Rename 'name' attribute
        if 'name' in element.attrib:
            element.set('name', prefix + element.get('name'))
            
        # Attributes that need prefixing because they refer to robot parts
        refs = ['joint', 'body', 'tendon', 'body1', 'body2', 'joint1', 'joint2']
        for attr in refs:
            if attr in element.attrib:
                element.set(attr, prefix + element.get(attr))
        
        for child in element:
            self._rename_attributes(child, prefix)

    def _set_class_on_geoms(self, element, class_name):
        if 'geom' in element.tag:
            element.set('class', class_name)
        for child in element:
            self._set_class_on_geoms(child, class_name)

    def _add_scene_to_robot(self, container_body, env_worldbody, robot_pos, scene_config, prefix):
        surface_name = scene_config.get('surface')
        object_names = scene_config.get('objects', [])
        collision = scene_config.get('collision', True)
        
        surface_body = None
        surface_size = [0.3, 0.3, 0.3] # Default size
        
        # Get surface position from config, default to [0.5, 0, 0] if not present
        surface_pos = scene_config.get('surface_position', [0.5, 0, 0])
        
        # 1. Add Surface
        if surface_name:
            original_surface = self.object_library.get_object(surface_name)
            if original_surface is not None:
                surface_body = copy.deepcopy(original_surface)
                if 'class' in surface_body.attrib:
                    del surface_body.attrib['class']

                # Rename
                self._rename_recursively(surface_body, prefix)
                
                # Position
                # Check if it has a pos, otherwise set default
                if 'pos' not in surface_body.attrib:
                    surface_body.set('pos', f"{surface_pos[0]} {surface_pos[1]} {surface_pos[2]}")
                else:
                    # If it has a pos, we might want to override it or respect it.
                    # For now, let's respect it if it's not 0 0 0, otherwise set default?
                    # Actually, the user's desks.xml has no pos. So we set it.
                    pass
                    
                # Try to guess size from geom
                geom = surface_body.find('geom')
                if geom is not None and 'size' in geom.attrib:
                    try:
                        s = [float(x) for x in geom.get('size').split()]
                        if len(s) >= 3:
                            surface_size = s
                    except:
                        pass
                
                # If the surface is a box, its top is at pos.z + size.z
                # But usually pos is 0.5, top is pos.z + size.z
                # If pos.z is 0, top is size.z.
                # But we want the desk to be on the floor?
                # If desk is 0.6m tall (size=0.3), and we want top at 0.6m?
                # Then pos.z should be 0.3.
                # Let's assume we want the surface top to be at some height, say 0.0 (floor) + height.
                # If we place it at z=0, it sinks.
                # Let's place it such that it sits on the floor.
                # z = surface_size[2]
                surface_body.set('pos', f"{surface_pos[0]} {surface_pos[1]} {surface_size[2]}")
                
                # Check if surface has freejoint (unlikely for desk, but possible)
                if surface_body.find('freejoint') is not None:
                    # Move to worldbody with global pos
                    local_pos = [float(x) for x in surface_body.get('pos').split()]
                    global_pos = [robot_pos[0] + local_pos[0], robot_pos[1] + local_pos[1], robot_pos[2] + local_pos[2]]
                    surface_body.set('pos', f"{global_pos[0]} {global_pos[1]} {global_pos[2]}")
                    env_worldbody.append(surface_body)
                else:
                    container_body.append(surface_body)
                
                # Apply collision setting to surface as well
                self._set_collision(surface_body, collision)
        
        # 2. Add Objects
        if surface_body is not None:
            # Place objects on top of the surface
            surface_top_z = surface_size[2] * 2 # If pos is at size[2], top is at size[2]*2? No.
            # If pos is at Z=h, and size is h, then top is at 2h.
            # Wait. Box size is half-extents.
            # If I place center at Z=h, top is at Z=h+h = 2h.
            # If I want top at H_desk, and size is h, then center should be at H_desk - h.
            # If I want bottom at 0, center is at h. Top is at 2h.
            # So surface_top_z = surface_size[2] * 2.
            
            for obj_name in object_names:
                original_obj = self.object_library.get_object(obj_name)
                if original_obj is not None:
                    obj_body = copy.deepcopy(original_obj)
                    if 'class' in obj_body.attrib:
                        del obj_body.attrib['class']
                    
                    # Apply collision setting
                    self._set_collision(obj_body, collision)

                    # Rename after setting collision and before adding to parent
                    self._rename_recursively(obj_body, prefix)
                    
                    
                    # Get object size to avoid falling off
                    obj_size = [0.03, 0.03, 0.03]
                    geom = obj_body.find('geom')
                    if geom is not None and 'size' in geom.attrib:
                        try:
                            s = [float(x) for x in geom.get('size').split()]
                            if len(s) >= 3: obj_size = s
                            elif len(s) == 2: obj_size = [s[0], s[0], s[1]] # Cylinder radius, height
                        except:
                            pass
                    
                    # Random offset
                    margin = 0.05
                    x_range = surface_size[0] - obj_size[0] - margin
                    y_range = surface_size[1] - obj_size[1] - margin
                    
                    if x_range < 0: x_range = 0
                    if y_range < 0: y_range = 0
                    
                    dx = random.uniform(-x_range, x_range)
                    dy = random.uniform(-y_range, y_range)
                    
                    # Object Z
                    # If object is box, pos is center. We want bottom at surface_top_z.
                    # So pos.z = surface_top_z + obj_size[2]
                    dz = surface_top_z + obj_size[2] + 0.001 # Add a tiny bit of drop height
                    
                    # Absolute pos relative to container
                    obj_x = surface_pos[0] + dx
                    obj_y = surface_pos[1] + dy
                    obj_z = dz
                    
                    obj_body.set('pos', f"{obj_x} {obj_y} {obj_z}")
                    
                    # Check if object has freejoint
                    has_freejoint = False
                    for child in obj_body:
                        if 'freejoint' in child.tag:
                            has_freejoint = True
                            break

                    if has_freejoint:
                        # Move to worldbody with global pos
                        global_pos = [robot_pos[0] + obj_x, robot_pos[1] + obj_y, robot_pos[2] + obj_z]
                        obj_body.set('pos', f"{global_pos[0]} {global_pos[1]} {global_pos[2]}")
                        env_worldbody.append(obj_body)
                    else:
                        container_body.append(obj_body)

    def _set_collision(self, element, enable):
        if element.tag == 'geom':
            if not enable:
                # If collision with robot is disabled:
                # Set scene objects to group 2 (bit 1).
                # Robot is default group 1 (bit 0).
                # Scene objects will collide with each other (2 & 2 != 0)
                # But not with robot (1 & 2 == 0)
                element.set('contype', '2')
                element.set('conaffinity', '2')
            else:
                # If collision is enabled, ensure they are in default group 1
                # or just remove the attributes to let them default
                # But to be safe, let's set them to 1 if they were modified or just leave them.
                # If we want to enforce collision, we can set to 1.
                # But existing objects might have their own settings.
                # For now, we only modify if enable is False.
                pass
        
        for child in element:
            self._set_collision(child, enable)