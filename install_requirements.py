#!/usr/bin/env python3
"""
Smart requirements installer that handles PyMuPDF build issues gracefully.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚀 Smart Requirements Installer")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment!")
        print("   Consider creating one with: python -m venv .venv")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Installation cancelled.")
            return
    
    # Install basic requirements first
    basic_packages = [
        "streamlit==1.38.0",
        "pillow>=10.4.0", 
        "opencv-python-headless>=4.10.0.84",
        "numpy<2.0,>=1.26.0",
        "pandas>=2.2.2",
        "pydantic>=2.7.4",
        "python-dotenv>=1.0.1",
        "openai>=1.40.0"
    ]
    
    print("\n📦 Installing basic packages...")
    for package in basic_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"   ⚠️  Failed to install {package}, continuing...")
    
    # Try to install PaddleOCR (this will pull in PyMuPDF)
    print("\n🔧 Installing PaddleOCR stack...")
    if not run_command("pip install paddleocr==2.7.0.3 paddlepaddle==2.6.1", "Installing PaddleOCR"):
        print("   ⚠️  PaddleOCR installation failed - this may be due to PyMuPDF build issues")
        print("   💡 You can try installing Visual Studio Build Tools and retry")
    
    # Install remaining packages
    remaining_packages = [
        "rapidfuzz>=3.9.6",
        "jiwer>=3.0.4", 
        "pytesseract>=0.3.10",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "transformers>=4.44.0",
        "sentencepiece>=0.2.0",
        "timm>=1.0.7",
        "onnxruntime>=1.18.0",
        "scikit-image>=0.24.0",
        "pyyaml>=6.0.2",
        "pyyaml-include>=1.4.1",
        "pdf2image>=1.17.0",
        "PyPDF2>=3.0.1",
        "annotated-types>=0.6.0",
        "typing-inspection>=0.4.0",
        "anyio>=3.5.0",
        "astor>=0.8.1",
        "imgaug>=0.4.0"
    ]
    
    print("\n📦 Installing remaining packages...")
    for package in remaining_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"   ⚠️  Failed to install {package}, continuing...")
    
    print("\n🎉 Installation completed!")
    print("\n📋 Next steps:")
    print("   1. Set up your OpenAI API key in .env file")
    print("   2. Run: python app.py")
    print("   3. Or use: .\\scripts\\run_app.ps1")

if __name__ == "__main__":
    main()
