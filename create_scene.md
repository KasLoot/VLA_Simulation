# Creating and Extending Scenes

This document explains how the scene generation system works and how to extend it.

## Overview

The simulation environment is built dynamically using `EnvironmentBuilder`. It combines:
1.  **Robot XML**: The base robot definition (e.g., Panda arm).
2.  **Environment XML**: The base environment (lighting, floor, cameras).
3.  **Scene JSON**: A configuration file defining specific tasks/scenes for each robot.
4.  **Object Library**: A collection of XML files defining available objects (desks, cubes, baskets).

## How it Works

1.  **`SceneGenerator`**: Generates a JSON file (e.g., `pick_and_place_scene.json`) that assigns a surface (like a desk) and a list of objects to each robot index.
2.  **`ObjectLibrary`**: Scans the `test/objects/` directory for XML files and loads object definitions.
3.  **`EnvironmentBuilder`**:
    *   Creates a grid of robots.
    *   For each robot, checks the JSON configuration.
    *   Loads the specified surface and objects from the `ObjectLibrary`.
    *   Places the surface relative to the robot using `surface_position` from JSON (default `0.5 0 0`).
    *   Randomly scatters the objects on top of the surface.
    *   Wraps everything in a container body `robot_{i}` to maintain local coordinates.

## Extending the System

### Adding New Objects

1.  Create a new XML file in `test/objects/` (e.g., `tools.xml`).
2.  Define the object body inside the XML. Ensure it has a unique name (e.g., `hammer`).
    ```xml
    <body name="hammer">
        <freejoint/> <!-- If it's movable -->
        <geom ... />
    </body>
    ```
3.  The `ObjectLibrary` will automatically pick it up.

### Creating New Task Setups

1.  Modify `test/scene_generator.py` or create a new generator script.
2.  Define a new logic to assign objects to robots.
    ```python
    scene_data.append({
        "robot_index": i,
        "surface": "desk_2",
        "surface_position": [0.6, 0, 0],
        "objects": ["hammer", "nail"]
    })
    ```
3.  Run the generator to update the JSON file.

### Customizing Placement Logic

Currently, objects are placed randomly on the surface. To customize this:
1.  Edit `test/scene_builder.py`.
2.  Modify the `_add_scene_to_robot` method.
3.  You can implement specific placement logic based on object names or add position parameters to the JSON configuration.

## File Structure

*   `test/run.py`: Main entry point.
*   `test/scene_builder.py`: Core logic for building the XML.
*   `test/scene_generator.py`: Generates the scene configuration JSON.
*   `test/object_library.py`: Manages available objects.
*   `test/objects/`: Directory containing object XML definitions.
*   `test/environments/`: Directory containing base environment XMLs.
