#!/usr/bin/env python3
"""
Skeleton Analyzer - Main Entry Point
Supports both GUI and headless modes
"""

import sys
import os
import argparse

# Add core and gui to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gui'))

def headless_mode():
    """Run in headless mode (Docker, servers, CI/CD)"""
    print("Skeleton Analyzer - Headless Mode")
    
    from core.config import validate_environment
    from core.evaluation import ModelEvaluator
    from core.preprocessing import ImageManager
    
    # Validate environment
    if not validate_environment():
        print("❌ Environment validation failed")
        return 1
    
    print("✅ Environment validated successfully")
    
    # Show available evaluations
    evaluations = ModelEvaluator.get_available_evaluations()
    print(f"📊 Available evaluations: {evaluations}")
    
    # Show available skeletal images
    human_images = ImageManager.get_skeletal_images('human')
    gorilla_images = ImageManager.get_skeletal_images('gorilla')
    print(f"Human skeletal images: {len(human_images)}")
    print(f"Gorilla skeletal images: {len(gorilla_images)}")
    
    return 0

def gui_mode():
    """Run in GUI mode (local development)"""
    try:
        from gui.app import run_gui
        print("🎨 Skeleton Analyzer - GUI Mode")
        run_gui()
        return 0
    except ImportError as e:
        print(f"❌ GUI mode not available: {e}")
        print("💡 Running in headless mode instead...")
        return headless_mode()

def main():
    parser = argparse.ArgumentParser(description='Skeleton Analyzer')
    parser.add_argument('--mode', choices=['gui', 'headless', 'auto'], 
                       default='auto', help='Run mode (default: auto)')
    
    args = parser.parse_args()
    
    if args.mode == 'headless' or (args.mode == 'auto' and not os.environ.get('DISPLAY')):
        return headless_mode()
    else:
        return gui_mode()

if __name__ == '__main__':
    sys.exit(main())