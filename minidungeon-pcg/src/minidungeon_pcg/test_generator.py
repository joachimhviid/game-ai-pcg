import sys
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, "src")

from minidungeon_pcg.pcg.generator import Generator

def main():
    print("=" * 60)
    print("Testing GA Dungeon Generator (Batch & Adaptive)")
    print("=" * 60)

    # 1. Initialize Generator
    generator = Generator(
        width=9,
        height=9,
        population_size=50,  # Smaller for testing speed
        generations=50,      # Fewer gens for testing speed
        mutation_rate=0.15,
        elite_size=5,
    )
    
    # 2. Test Single Generation (Backward Compatibility)
    print("\n[Test 1] Generating Single Dungeon...")
    dungeon = generator.generate_dungeon(stage_name="ga_single_test")
    generator.print_dungeon(dungeon)
    print("✓ Single dungeon saved to: ga_single_test.txt")

    # 3. Test Batch Generation
    print("\n" + "=" * 60)
    print("[Test 2] Generating Batch of 3 Levels...")
    batch_files = generator.generate_batch(batch_size=10, stage_name_prefix="ga_batch_test")
    
    print(f"✓ Generated {len(batch_files)} files: {batch_files}")

    # 4. Test Adaptive Difficulty (Simulating feedback)
    print("\n" + "=" * 60)
    print("[Test 3] Testing Adaptive Difficulty Update...")
    
    print(f"Current Difficulty: {generator.config['difficulty_level']}")
    print(f"Current Monsters: {generator.config['target_monster_count']}")

    # Simulate a high win rate (too easy) -> Should increase difficulty
    print("\n>> Simulating High Win Rate (100%)...")
    generator.update_difficulty(avg_reward=20.0, win_rate=1.0)
    
    print(f"New Difficulty: {generator.config['difficulty_level']}")
    print(f"New Monsters: {generator.config['target_monster_count']}")

    if generator.config['difficulty_level'] > 1:
        print("✓ Difficulty successfully increased!")
    else:
        print("x Difficulty did not increase (Check thresholds).")

    # 5. Test Saving/Loading Model
    print("\n" + "=" * 60)
    print("[Test 4] Saving & Loading Generator State...")
    model_path = Path("test_gen_config.json")
    generator.save_model(model_path)
    
    # Create new generator and load settings
    gen2 = Generator()
    gen2.load_model(model_path)
    
    if gen2.config['target_monster_count'] == generator.config['target_monster_count']:
        print("✓ Model saved and loaded correctly.")
    else:
        print("x Model load failed.")

    # Cleanup test files (optional, comment out if you want to inspect them)
    if model_path.exists():
        model_path.unlink()

    print("\n" + "=" * 60)
    print("All Tests Complete.")

if __name__ == "__main__":
    main()