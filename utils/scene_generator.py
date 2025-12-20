import json
import random
import os

class SceneGenerator:
    def __init__(self, output_path, num_robots=None, seed: int | None = None):
        self.output_path = output_path
        self.num_robots = num_robots
        if seed is not None:
            random.seed(seed)
        assert isinstance(num_robots, int), "num_robots must be specified as an integer"

    def generate_scene(self, task, surface_position=[0.6, 0, 0], min_objects=1, max_objects=5, collision=True):
        if task == "pick_and_place":
            self.generate_pick_and_place_scene(self.num_robots, surface_position, min_objects, max_objects, collision)
        else:
            raise ValueError(f"Unsupported task type: {task}")

    def generate_pick_and_place_scene(self, num_robots=2, surface_position=[0.6, 0, 0], min_objects=1, max_objects=5, collision=True):
        scene_data = []
        available_objects = ["cube_1", "cube_2", "cube_3", "cube_4", "basket_1"]
        
        # Fill the rest with random assignments if num_robots > 2
        for i in range(num_robots):
            # Determine number of objects to spawn
            lower_bound = min_objects
            upper_bound = max_objects
            
            k = random.randint(lower_bound, upper_bound)
            
            scene_data.append({
                "robot_index": i,
                "surface": random.choice(["desk_1"]),
                "surface_position": surface_position,
                "objects": random.sample(available_objects, k=k),
                "collision": collision
            })

        self.save_scene(scene_data)

    def save_scene(self, data):
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Scene JSON saved to {self.output_path}")

if __name__ == "__main__":
    generator = SceneGenerator("pick_and_place", "test/scene/pick_and_place_scene.json", num_robots=10)
    generator.generate_scene()