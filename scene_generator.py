import json
import random
import os

class SceneGenerator:
    def __init__(self, output_path, num_robots=None):
        self.output_path = output_path
        self.num_robots = num_robots
        assert isinstance(num_robots, int), "num_robots must be specified as an integer"

    def generate_scene(self, task, surface_position=[0.6, 0, 0]):
        if task == "pick_and_place":
            self.generate_pick_and_place_scene(self.num_robots, surface_position)
        else:
            raise ValueError(f"Unsupported task type: {task}")

    def generate_pick_and_place_scene(self, num_robots=2, surface_position=[0.6, 0, 0]):
        scene_data = []
        
        # Fill the rest with random assignments if num_robots > 2
        for i in range(num_robots):
            scene_data.append({
                "robot_index": i,
                "surface": random.choice(["desk_1"]),
                "surface_position": surface_position,
                "objects": random.sample(["cube_1", "cube_2", "cube_3", "cube_4", "basket_1"], k=random.randint(1, 3))
            })

        self.save_scene(scene_data)

    def save_scene(self, data):
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Scene JSON saved to {self.output_path}")

if __name__ == "__main__":
    generator = SceneGenerator("pick_and_place", "test/scene/pick_and_place_scene.json", num_robots=10)
    generator.generate_scene()