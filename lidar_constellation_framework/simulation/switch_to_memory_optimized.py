"""
Switch to Memory-Optimized Configuration

This script switches to the memory-optimized configuration and provides
instructions for running the memory-optimized training.
"""

import os
import shutil
import sys

def switch_to_memory_optimized():
    """Switch to the memory-optimized configuration."""
    print("🔄 Switching to MEMORY-OPTIMIZED configuration...")
    
    # Backup current config
    if os.path.exists("config.py"):
        shutil.copy("config.py", "config_current_backup.py")
        print("✅ Backed up current config as config_current_backup.py")
    
    # Copy memory-optimized config
    if os.path.exists("config_memory_optimized.py"):
        shutil.copy("config_memory_optimized.py", "config.py")
        print("✅ Switched to memory-optimized configuration")
        print("\n📊 Memory-Optimized Configuration Features:")
        print("   • 40% shorter episodes (30 timesteps vs 50)")
        print("   • 25% fewer satellites (15 vs 20)")
        print("   • 40% fewer training episodes (300 vs 500)")
        print("   • 50% smaller memory buffer (5000 vs 10000)")
        print("   • 50% smaller batch size (16 vs 32)")
        print("   • 50% smaller neural network (64 vs 128 hidden dims)")
        print("   • Memory monitoring and cleanup")
        print("   • Expected completion time: 1-2 hours")
        print("   • Expected memory usage: <2GB")
    else:
        print("❌ config_memory_optimized.py not found!")
        return False
    
    return True

def show_memory_optimization_tips():
    """Show tips for memory optimization."""
    print("\n💡 Memory Optimization Tips:")
    print("   1. Close other applications to free up RAM")
    print("   2. Monitor memory usage during training")
    print("   3. If memory issues occur, reduce BATCH_SIZE further")
    print("   4. Consider using CPU if GPU memory is limited")
    print("   5. The system will automatically clean up memory every 25 episodes")

def main():
    """Main function to handle configuration switching."""
    print("=" * 60)
    print("LiDAR Constellation Memory Optimization")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python switch_to_memory_optimized.py switch  # Switch to memory-optimized config")
        print("  python switch_to_memory_optimized.py tips     # Show memory optimization tips")
        print("  python switch_to_memory_optimized.py run      # Switch and show run instructions")
        return
    
    command = sys.argv[1].lower()
    
    if command == "switch":
        switch_to_memory_optimized()
    elif command == "tips":
        show_memory_optimization_tips()
    elif command == "run":
        if switch_to_memory_optimized():
            print("\n🚀 Ready to run memory-optimized training!")
            print("\nTo start training, run:")
            print("  python main_memory_optimized.py")
            print("\nThis will:")
            print("  • Use memory-optimized PPO agent")
            print("  • Monitor memory usage in real-time")
            print("  • Perform automatic memory cleanup")
            print("  • Complete training in 1-2 hours")
            print("  • Use <2GB of RAM")
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: switch, tips, run")

if __name__ == "__main__":
    main() 