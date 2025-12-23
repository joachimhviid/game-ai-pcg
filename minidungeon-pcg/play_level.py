
import argparse
import time
import numpy as np
from minidungeon_pcg.envs.md_env import MdEnv

def play(stage_name: str):
    """
    Plays a level with rendering enabled.
    """
    print(f"--- Playing Level: {stage_name} ---")
    
    try:
        # 1. Create the environment with human rendering
        env = MdEnv(stage_name=stage_name, render_mode="human")
        
        # 2. Reset the environment
        obs, info = env.reset()
        env.render()
        time.sleep(1) # Pause to see initial state

        done = False
        total_reward = 0
        step_count = 0
        max_steps = 100 # Safety break

        # 3. Run the simulation loop
        while not done and step_count < max_steps:
            # The agent logic is inside MdEnv, so we can pass a dummy action
            action = np.zeros(env.action_space.shape) 
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Render the frame
            env.render()
            
            step_count += 1
            time.sleep(0.1) # Slow down for visibility

        print(f"--- Playback Complete ---")
        print(f"Total reward: {total_reward}")
        print(f"Steps: {step_count}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure the stage name is correct and the file exists in 'minidungeon-pcg/src/minidungeon_pcg/pcg/stages/'.")
        print("Example: 'ppo_generated-0' (without the .json extension)")
    finally:
        if 'env' in locals() and env:
            env.close()


def main():
    parser = argparse.ArgumentParser(description="Play back a generated Minidungeon level with visual rendering.")
    parser.add_argument("stage_name", type=str, help="The name of the stage file to play (e.g., 'ppo_generated-0').")
    args = parser.parse_args()
    
    play(args.stage_name)

if __name__ == "__main__":
    main()
