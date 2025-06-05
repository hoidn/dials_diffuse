#!/usr/bin/env python3
"""
Test script to verify all imports work correctly without DIALS.
"""

import sys
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    
    # Test plot utils
    try:
        from plot_utils import (
            plot_detector_image,
            plot_mask_overlay,
            plot_spot_overlay,
            ensure_output_dir
        )
        print("✅ plot_utils imports successful")
    except Exception as e:
        print(f"❌ plot_utils import failed: {e}")
    
    # Test diffusepipe components
    try:
        from diffusepipe.masking.pixel_mask_generator import PixelMaskGenerator
        from diffusepipe.masking.bragg_mask_generator import BraggMaskGenerator
        from diffusepipe.crystallography.still_processor import StillProcessorComponent
        print("✅ diffusepipe components imports successful")
    except Exception as e:
        print(f"❌ diffusepipe components import failed: {e}")
    
    # Test matplotlib backend
    try:
        import matplotlib
        print(f"✅ matplotlib backend: {matplotlib.get_backend()}")
    except Exception as e:
        print(f"❌ matplotlib import failed: {e}")
    
    # Test component instantiation
    try:
        pixel_gen = PixelMaskGenerator()
        bragg_gen = BraggMaskGenerator()
        print("✅ Component instantiation successful")
    except Exception as e:
        print(f"❌ Component instantiation failed: {e}")
    
    print("\nAll import tests completed!")

if __name__ == "__main__":
    test_imports()