import sys
sys.path.insert(0, 'src')

from minidungeon_pcg.pcg.generator import Generator

def main():
    print("="*60)
    print("Testing GA Dungeon Generator")
    print("="*60)
    
    generator = Generator(
        width=9,
        height=9,
        population_size=50,
        generations=100,
        mutation_rate=0.15,
        elite_size=5
    )
    
    dungeon = generator.generate_dungeon(stage_name="ga_generated")
    
    print("\n" + "="*60)
    print("Generated Dungeon:")
    print("="*60)
    generator.print_dungeon(dungeon)
    
    print("✓ Dungeon saved to: src/minidungeon_pcg/pcg/stages/ga_generated.txt")
    print("\nTo play this dungeon, run:")
    print("  poetry run start ga_generated")

if __name__ == "__main__":
    main()