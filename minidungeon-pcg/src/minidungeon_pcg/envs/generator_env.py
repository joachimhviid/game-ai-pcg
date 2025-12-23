import gymnasium as gym
import numpy as np
from minidungeon_pcg.pcg.generator import Generator
from minidungeon_pcg.envs.md_env import MdEnv
import os

class GeneratorEnv(gym.Env):
    """
    A gym environment for training the level generator.
    - The 'agent' is PPO, which learns to tune the generator's parameters.
    - An 'action' is a change to the generator's parameters.
    - An 'observation' is the current state of the generator's parameters.
    - The 'reward' is the total reward obtained by a game-playing agent on the generated level.
    """
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.generator = Generator()
        self.stage_name_prefix = "ppo_generated_level"
        self.step_count = 0

        # Define the action space: delta for [min_path_length, target_monster_count, target_potion_count, target_treasure_count]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Define the observation space: current values of the parameters
        self.observation_space = gym.spaces.Box(
            low=np.array([5, 1, 0, 1]), # Min values for params
            high=np.array([20, 10, 5, 10]), # Max values for params
            dtype=np.float32
        )

    def step(self, action):
        # 1. Apply action to generator parameters
        current_params = np.array([
            self.generator.config["min_path_length"],
            self.generator.config["target_monster_count"],
            self.generator.config["target_potion_count"],
            self.generator.config["target_treasure_count"]
        ])

        # Scale action to a reasonable step size
        # Let's say an action of 1.0 corresponds to a change of 1 unit
        scaled_action = action
        new_params = current_params + scaled_action

        # Clip parameters to be within the observation space bounds
        low_bounds = self.observation_space.low
        high_bounds = self.observation_space.high
        new_params = np.clip(new_params, low_bounds, high_bounds)
        
        # Update generator config, rounding int params
        self.generator.config["min_path_length"] = int(round(new_params[0]))
        self.generator.config["target_monster_count"] = int(round(new_params[1]))
        self.generator.config["target_potion_count"] = int(round(new_params[2]))
        self.generator.config["target_treasure_count"] = int(round(new_params[3]))
        self.generator.update_internal_params()

        if self.debug:
            print(f"GeneratorEnv: New params: {self.generator.config}")

        # 2. Generate a level with new parameters
        stage_name = f"{self.stage_name_prefix}_{self.step_count}"
        self.step_count += 1
        try:
            self.generator.generate_dungeon(stage_name=stage_name)
        except Exception as e:
            if self.debug:
                print(f"GeneratorEnv: Failed to generate dungeon: {e}")
            # Penalize for failing to generate a valid level
            return self._get_obs(), -50.0, True, False, {}

        # 3. Simulate agent on the level
        total_reward = 0
        try:
            sim_env = MdEnv(stage_name=stage_name, debug=self.debug)
            obs, info = sim_env.reset()
            done = False
            sim_step_count = 0
            max_steps = 100 # Prevent infinite loops
            while not done and sim_step_count < max_steps:
                # The MdTreasureAgent inside MdEnv ignores this action
                sim_action = np.zeros(sim_env.action_space.shape)
                obs, reward, terminated, truncated, info = sim_env.step(sim_action)
                total_reward += reward
                done = terminated or truncated
                sim_step_count += 1
            sim_env.close()
        except Exception as e:
             if self.debug:
                print(f"GeneratorEnv: Error during simulation: {e}")
             return self._get_obs(), -100.0, True, False, {}


        if self.debug:
            print(f"GeneratorEnv: Simulation finished. Total reward: {total_reward}")

        # 4. Return results
        # Episode is done after one generation/simulation cycle
        return self._get_obs(), total_reward, True, False, {}

    def reset(self, *, seed=None, options=None):
        # Reset generator to default config
        self.generator.config = {
            "min_path_length": 8,
            "target_monster_count": 3,
            "target_potion_count": 1,
            "target_treasure_count": 3,
            "difficulty_level": 1
        }
        self.generator.update_internal_params()
        if self.debug:
            print("GeneratorEnv: Reset to default parameters.")
        return self._get_obs(), {}

    def render(self, mode='human'):
        # This environment does not have a visual representation
        pass

    def close(self):
        # Clean up any resources if needed
        pass

    def _get_obs(self):
        """Get the current observation from the generator's config."""
        obs = np.array([
            self.generator.config["min_path_length"],
            self.generator.config["target_monster_count"],
            self.generator.config["target_potion_count"],
            self.generator.config["target_treasure_count"]
        ], dtype=np.float32)
        return obs
