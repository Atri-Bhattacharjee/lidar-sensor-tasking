"""
Configuration Switcher for LiDAR Constellation Framework

This script allows you to easily switch between different training configurations:
- Standard: Full-featured training (longer, more comprehensive)
- Fast: Optimized for quick training (2-3 hours)
"""

import os
import shutil
import sys

def switch_to_fast_config():
    """Switch to the fast training configuration."""
    print("🔄 Switching to FAST training configuration...")
    
    # Backup current config
    if os.path.exists("config.py"):
        shutil.copy("config.py", "config_standard_backup.py")
        print("✅ Backed up current config as config_standard_backup.py")
    
    # Copy fast config
    if os.path.exists("config_fast.py"):
        shutil.copy("config_fast.py", "config.py")
        print("✅ Switched to fast configuration")
        print("\n📊 Fast Configuration Features:")
        print("   • 50% shorter episodes (50 timesteps vs 100)")
        print("   • Full constellation (40 satellites)")
        print("   • 60% fewer training episodes (400 vs 1000)")
        print("   • Reduced noise and clutter for easier learning")
        print("   • Optimized learning rate for faster convergence")
        print("   • Expected completion time: 1-3 hours")
    else:
        print("❌ config_fast.py not found!")
        return False
    
    return True

def switch_to_standard_config():
    """Switch to the standard training configuration."""
    print("🔄 Switching to STANDARD training configuration...")
    
    # Restore from backup
    if os.path.exists("config_standard_backup.py"):
        shutil.copy("config_standard_backup.py", "config.py")
        print("✅ Restored standard configuration")
        print("\n📊 Standard Configuration Features:")
        print("   • Full episodes (100 timesteps)")
        print("   • Full constellation (40 satellites)")
        print("   • Full training (1000 episodes)")
        print("   • Realistic noise and clutter levels")
        print("   • Larger neural network for better performance")
        print("   • Expected completion time: 6-8 hours")
    else:
        print("❌ No backup found! Please restore manually.")
        return False
    
    return True

def show_current_config():
    """Show which configuration is currently active."""
    if os.path.exists("config.py"):
        with open("config.py", "r") as f:
            first_line = f.readline().strip()
        
        if "Optimized Configuration for Fast Training" in first_line:
            print("🚀 Currently using: FAST configuration")
            print("   Expected completion: 2-3 hours")
        else:
            print("📚 Currently using: STANDARD configuration")
            print("   Expected completion: 6-8 hours")
    else:
        print("❌ No configuration file found!")

def main():
    """Main function to handle configuration switching."""
    print("=" * 60)
    print("LiDAR Constellation Configuration Switcher")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python switch_config.py fast     # Switch to fast training")
        print("  python switch_config.py standard # Switch to standard training")
        print("  python switch_config.py status   # Show current configuration")
        return
    
    command = sys.argv[1].lower()
    
    if command == "fast":
        switch_to_fast_config()
    elif command == "standard":
        switch_to_standard_config()
    elif command == "status":
        show_current_config()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: fast, standard, status")

if __name__ == "__main__":
    main() 