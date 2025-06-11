#!/usr/bin/env python3
"""
Debug script to directly test Q-vector validation on processed DIALS data.
"""

import sys
import logging
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diffusepipe.crystallography.still_processing_and_validation import (
    StillProcessorAndValidatorComponent,
    create_default_config,
    create_default_extraction_config
)

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_q_validation():
    """Debug the Q-vector validation on a real processed image."""
    print("=" * 60)
    print("DEBUG Q-VECTOR VALIDATION")
    print("=" * 60)
    
    image_path = "747/lys_nitr_10_6_0491.cbf"
    
    # Create configurations
    dials_config = create_default_config(
        enable_partiality=True,
        enable_shoeboxes=True,
        known_unit_cell="27.424,32.134,34.513,88.66,108.46,111.88",
        known_space_group="P 1"
    )
    extraction_config = create_default_extraction_config()
    
    # Initialize processor
    processor = StillProcessorAndValidatorComponent()
    
    print(f"\n1. Processing image: {image_path}")
    outcome = processor.process_and_validate_still(
        image_path=image_path,
        config=dials_config,
        extraction_config=extraction_config,
        external_pdb_path="6o2h.pdb"
    )
    
    print(f"\n2. Processing result: {outcome.status}")
    
    if outcome.output_artifacts:
        experiment = outcome.output_artifacts.get("experiment")
        reflections = outcome.output_artifacts.get("reflections")
        
        if experiment and reflections:
            print(f"\n3. Direct Q-vector validation test:")
            
            # Create a validator and test directly
            validator = processor.validator
            
            # Test with debug logging
            tolerance = extraction_config.q_consistency_tolerance_angstrom_inv
            print(f"   Using tolerance: {tolerance} Å⁻¹")
            
            passed, stats = validator._check_q_consistency(
                experiment=experiment,
                reflections=reflections,
                tolerance=tolerance
            )
            
            print(f"\n4. Validation Results:")
            print(f"   Passed: {passed}")
            print(f"   Stats: {stats}")
            
            # Also inspect the reflection table directly
            print(f"\n5. Reflection Table Inspection:")
            try:
                print(f"   Type: {type(reflections)}")
                print(f"   Length: {len(reflections)}")
                if hasattr(reflections, 'keys'):
                    keys = list(reflections.keys())
                    print(f"   Available columns: {keys}")
                    
                    # Check for required columns
                    required = ['miller_index', 'panel']
                    for col in required:
                        if col in reflections:
                            sample = reflections[col][:3] if len(reflections) > 0 else []
                            print(f"   {col} (sample): {sample}")
                        else:
                            print(f"   {col}: MISSING")
                    
                    # Check position columns
                    pos_cols = ['xyzcal.mm', 'xyzobs.mm.value', 'xyzcal.px', 'xyzobs.px.value']
                    for col in pos_cols:
                        if col in reflections:
                            sample = reflections[col][:3] if len(reflections) > 0 else []
                            print(f"   {col} (sample): {sample}")
                
            except Exception as e:
                print(f"   Error inspecting reflections: {e}")
        else:
            print("\n3. No experiment/reflections available for direct testing")
    else:
        print("\n3. No output artifacts available")

if __name__ == "__main__":
    debug_q_validation()