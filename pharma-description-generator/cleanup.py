#!/usr/bin/env python3
"""
Cleanup script to remove all output files and temporary data.
Run this to clean the workspace of any generated files.
"""

import os
import shutil
import glob

def cleanup_workspace():
    """Remove all output files and temporary data from the workspace."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Directories to clean
    dirs_to_clean = ['output', 'uploads', '__pycache__']
    
    # File patterns to remove
    patterns_to_remove = [
        '*.xlsx',
        '*.xls', 
        'test_output.*',
        'output_*.xlsx',
        '*.tmp',
        '*.temp',
        '*.log'
    ]
    
    print("🧹 Cleaning up workspace...")
    
    # Remove directories
    for dir_name in dirs_to_clean:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"✅ Removed directory: {dir_name}/")
            except Exception as e:
                print(f"❌ Could not remove {dir_name}/: {str(e)}")
    
    # Remove files by pattern
    for pattern in patterns_to_remove:
        files = glob.glob(os.path.join(base_dir, pattern))
        for file_path in files:
            try:
                os.remove(file_path)
                print(f"✅ Removed file: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Could not remove {os.path.basename(file_path)}: {str(e)}")
    
    # Recreate necessary directories
    necessary_dirs = ['uploads']
    for dir_name in necessary_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Recreated directory: {dir_name}/")
    
    print("\n🎉 Workspace cleanup complete!")
    print("📝 Note: Only source code files remain")

if __name__ == "__main__":
    cleanup_workspace()
