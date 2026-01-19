#!/usr/bin/env python3
"""
Environment setup script to prevent meta tensor issues with embedding models.
Run this script before using AgentOccam with memory retrieval functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

def set_environment_variables():
    """Set environment variables to prevent meta tensor and other common issues."""
    print("🔧 Setting up environment variables...")
    
    env_vars = {
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_NO_ADVISORY_WARNINGS': 'true',
        'PYTHONPATH': str(Path(__file__).parent.absolute()),
        'HF_HOME': str(Path.home() / '.cache' / 'huggingface'),
        'TRANSFORMERS_CACHE': str(Path.home() / '.cache' / 'huggingface' / 'transformers'),
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  ✅ {key}={value}")
    
    # Create cache directories if they don't exist
    cache_dirs = [
        Path.home() / '.cache' / 'huggingface',
        Path.home() / '.cache' / 'huggingface' / 'transformers',
        Path(__file__).parent / 'cache'
    ]
    
    for cache_dir in cache_dirs:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  📁 Created cache directory: {cache_dir}")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch',
        'sentence_transformers', 
        'faiss',
        'numpy',
        'openai',
        'transformers',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'faiss':
                # faiss-cpu installs as 'faiss'
                import faiss
            elif package == 'sentence_transformers':
                from sentence_transformers import SentenceTransformer
            elif package == 'torch':
                import torch
            elif package == 'numpy':
                import numpy
            elif package == 'openai':
                import openai
            elif package == 'transformers':
                import transformers
            
            print(f"  ✅ {package} is installed")
        except ImportError:
            print(f"  ❌ {package} is NOT installed")
            missing_packages.append(package)
    
    return missing_packages

def install_missing_dependencies(missing_packages):
    """Install missing dependencies."""
    if not missing_packages:
        return True
        
    print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
    
    # Map package names to pip install names
    pip_names = {
        'faiss': 'faiss-cpu',
        'sentence_transformers': 'sentence-transformers',
    }
    
    install_packages = []
    for pkg in missing_packages:
        install_packages.append(pip_names.get(pkg, pkg))
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + install_packages
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("✅ Successfully installed missing packages")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def test_pytorch_setup():
    """Test PyTorch setup to ensure it works correctly."""
    print("\n🧪 Testing PyTorch setup...")
    
    try:
        import torch
        
        # Test basic torch functionality
        tensor = torch.randn(2, 3)
        print(f"  ✅ Basic tensor creation: {tensor.shape}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  ✅ CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ℹ️  CUDA is not available (using CPU)")
        
        # Test device operations
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor_device = tensor.to(device)
        print(f"  ✅ Device operations work: {tensor_device.device}")
        
        return True
    except Exception as e:
        print(f"  ❌ PyTorch test failed: {e}")
        return False

def create_export_script():
    """Create a script to export environment variables for shell sessions."""
    script_content = """#!/bin/bash
# AgentOccam Environment Setup Script
# Source this file to set up environment variables for embedding model loading

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"

echo "✅ AgentOccam environment variables set"
echo "💡 To make these permanent, add them to your ~/.bashrc or ~/.zshrc"
"""
    
    export_script_path = Path(__file__).parent / 'export_embedding_env.sh'
    with open(export_script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    import stat
    export_script_path.chmod(export_script_path.stat().st_mode | stat.S_IEXEC)
    
    print(f"\n📝 Created environment export script: {export_script_path}")
    print("   Run: source export_embedding_env.sh")

def main():
    """Main setup function."""
    print("🚀 AgentOccam Embedding Environment Setup")
    print("=" * 50)
    
    # Set environment variables
    set_environment_variables()
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    # Install missing dependencies if any
    if missing_packages:
        install_success = install_missing_dependencies(missing_packages)
        if not install_success:
            print("\n❌ Setup failed due to dependency installation issues.")
            return 1
    
    # Test PyTorch setup
    pytorch_ok = test_pytorch_setup()
    if not pytorch_ok:
        print("\n⚠️  PyTorch setup has issues, but continuing...")
    
    # Create export script
    create_export_script()
    
    print("\n🎉 Environment setup complete!")
    print("\n💡 Next steps:")
    print("   1. If you see any dependency errors, run: pip install -r requirements.txt")
    print("   2. For shell sessions, run: source export_embedding_env.sh")
    print("   3. Test with: python test_embedding_fix.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

