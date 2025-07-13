#!/usr/bin/env python3
"""
Setup script for Isaac Gym and MANO dependencies
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
from termcolor import cprint

def run_command(cmd, check=True, shell=True):
    """Run shell command with error handling"""
    cprint(f"🔄 Running: {cmd}", "yellow")
    try:
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        cprint(f"❌ Command failed: {e}", "red")
        if e.stderr:
            print(e.stderr)
        return False

def check_isaac_gym():
    """Check if Isaac Gym is properly installed"""
    cprint("🔍 Checking Isaac Gym installation...", "blue")
    
    # Check if Isaac Gym directory exists
    isaac_paths = [
        "/workspace/isaacgym",
        "./isaacgym", 
        os.path.expanduser("~/isaacgym"),
        "/opt/isaacgym"
    ]
    
    for path in isaac_paths:
        if os.path.exists(path):
            cprint(f"✅ Found Isaac Gym at: {path}", "green")
            return path
    
    cprint("⚠️ Isaac Gym not found in standard locations", "yellow")
    return None

def setup_isaac_gym():
    """Setup Isaac Gym"""
    cprint("🚀 Setting up Isaac Gym...", "blue")
    
    isaac_path = check_isaac_gym()
    
    if not isaac_path:
        cprint("📥 Isaac Gym needs to be manually downloaded from NVIDIA", "yellow")
        cprint("1. Go to: https://developer.nvidia.com/isaac-gym", "cyan")
        cprint("2. Download Isaac Gym Preview Release", "cyan")
        cprint("3. Extract to /workspace/isaacgym", "cyan")
        return False
    
    # Install Isaac Gym Python package
    python_dir = os.path.join(isaac_path, "python")
    if os.path.exists(python_dir):
        cprint(f"📦 Installing Isaac Gym from {python_dir}", "blue")
        
        # Install in development mode
        cmd = f"cd {python_dir} && pip install -e ."
        if run_command(cmd):
            cprint("✅ Isaac Gym installed successfully", "green")
            return True
        else:
            cprint("❌ Failed to install Isaac Gym", "red")
            return False
    else:
        cprint(f"❌ Python directory not found in {isaac_path}", "red")
        return False

def setup_mano():
    """Setup MANO model"""
    cprint("🚀 Setting up MANO...", "blue")
    
    # Check if MANO is already installed
    try:
        import manopth
        cprint("✅ MANO (manopth) already installed", "green")
        return True
    except ImportError:
        pass
    
    # Install manopth
    cprint("📦 Installing manopth...", "blue")
    cmd = "pip install git+https://github.com/hassony2/manopth.git"
    if not run_command(cmd):
        cprint("⚠️ Failed to install manopth, trying alternative...", "yellow")
        
        # Try alternative installation
        cmd = "pip install manopth"
        if not run_command(cmd):
            cprint("❌ Failed to install MANO", "red")
            return False
    
    # Download MANO models
    cprint("📥 Setting up MANO model files...", "blue")
    
    mano_dir = "./mano"
    os.makedirs(mano_dir, exist_ok=True)
    
    cprint("📋 MANO model setup:", "cyan")
    cprint("1. Register at: https://mano.is.tue.mpg.de/", "cyan")
    cprint("2. Download MANO_RIGHT.pkl and MANO_LEFT.pkl", "cyan")
    cprint(f"3. Place them in: {os.path.abspath(mano_dir)}", "cyan")
    
    # Check if model files exist
    mano_files = ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]
    all_exist = True
    for file in mano_files:
        filepath = os.path.join(mano_dir, file)
        if os.path.exists(filepath):
            cprint(f"✅ Found {file}", "green")
        else:
            cprint(f"⚠️ Missing {file}", "yellow")
            all_exist = False
    
    if all_exist:
        cprint("✅ MANO setup complete", "green")
        return True
    else:
        cprint("⚠️ MANO models need to be downloaded manually", "yellow")
        return False

def setup_pytorch_kinematics():
    """Setup pytorch_kinematics"""
    cprint("🚀 Setting up pytorch_kinematics...", "blue")
    
    try:
        import pytorch_kinematics
        cprint("✅ pytorch_kinematics already installed", "green")
        return True
    except ImportError:
        pass
    
    cmd = "pip install pytorch-kinematics"
    if run_command(cmd):
        cprint("✅ pytorch_kinematics installed", "green")
        return True
    else:
        # Try alternative
        cmd = "pip install git+https://github.com/UM-ARM-Lab/pytorch_kinematics.git"
        if run_command(cmd):
            cprint("✅ pytorch_kinematics installed from git", "green")
            return True
        else:
            cprint("❌ Failed to install pytorch_kinematics", "red")
            return False

def create_isaac_gym_import_wrapper():
    """Create a wrapper to handle Isaac Gym import order"""
    wrapper_content = '''# Isaac Gym Import Wrapper
# This module handles the correct import order for Isaac Gym
import os
import sys

# Set environment variables for Isaac Gym
os.environ["ISAAC_GYM_HEADLESS"] = "1"  # Default to headless mode

# Import Isaac Gym FIRST before any other imports
try:
    from isaacgym import gymapi, gymtorch, gymutil
    import pytorch_kinematics as pk
    ISAAC_AVAILABLE = True
    print("✅ Isaac Gym imported successfully")
except ImportError as e:
    print(f"⚠️ Isaac Gym not available: {e}")
    ISAAC_AVAILABLE = False

# Now safe to import other modules
import torch
import numpy as np

def get_isaac_gym():
    """Get Isaac Gym modules if available"""
    if ISAAC_AVAILABLE:
        return gymapi, gymtorch, gymutil
    else:
        return None, None, None

def get_pytorch_kinematics():
    """Get pytorch_kinematics if available"""
    if ISAAC_AVAILABLE:
        return pk
    else:
        return None
'''
    
    with open("isaac_gym_wrapper.py", "w") as f:
        f.write(wrapper_content)
    
    cprint("✅ Created isaac_gym_wrapper.py", "green")

def test_imports():
    """Test if everything imports correctly"""
    cprint("🧪 Testing imports...", "blue")
    
    # Test Isaac Gym import order
    test_script = '''
import os
import sys
sys.path.append('.')

# Test Isaac Gym import
try:
    from isaac_gym_wrapper import get_isaac_gym, get_pytorch_kinematics, ISAAC_AVAILABLE
    if ISAAC_AVAILABLE:
        print("✅ Isaac Gym wrapper works")
    else:
        print("⚠️ Isaac Gym not available via wrapper")
except Exception as e:
    print(f"❌ Isaac Gym wrapper failed: {e}")

# Test MANO import
try:
    import manopth
    print("✅ MANO imports successfully")
except Exception as e:
    print(f"⚠️ MANO import failed: {e}")

# Test pytorch_kinematics
try:
    import pytorch_kinematics
    print("✅ pytorch_kinematics imports successfully")
except Exception as e:
    print(f"⚠️ pytorch_kinematics import failed: {e}")
'''
    
    with open("test_imports.py", "w") as f:
        f.write(test_script)
    
    cprint("🧪 Running import test...", "yellow")
    run_command("python test_imports.py")
    
    # Clean up
    os.remove("test_imports.py")

def main():
    """Main setup function"""
    cprint("🚀 Setting up dependencies for MANO to Inspire Hand optimization", "blue")
    
    success_count = 0
    total_steps = 4
    
    # 1. Setup Isaac Gym
    if setup_isaac_gym():
        success_count += 1
    
    # 2. Setup MANO
    if setup_mano():
        success_count += 1
    
    # 3. Setup pytorch_kinematics
    if setup_pytorch_kinematics():
        success_count += 1
    
    # 4. Create Isaac Gym wrapper
    create_isaac_gym_import_wrapper()
    success_count += 1
    
    # Test everything
    test_imports()
    
    # Summary
    cprint(f"\n📋 Setup Summary: {success_count}/{total_steps} steps completed", "cyan")
    
    if success_count == total_steps:
        cprint("🎉 All dependencies set up successfully!", "green")
        cprint("\n📖 Next steps:", "blue")
        cprint("1. Run: python run_shape_optimization.py --check_deps", "cyan")
        cprint("2. Run: python run_shape_optimization.py --mode advanced", "cyan")
    else:
        cprint("⚠️ Some dependencies need manual setup", "yellow")
        cprint("\n📋 Manual steps needed:", "blue")
        cprint("• Download Isaac Gym from NVIDIA Developer website", "cyan")
        cprint("• Download MANO models from official website", "cyan")

if __name__ == "__main__":
    main() 