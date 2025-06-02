#!/usr/bin/env python
# calculate_q_per_pixel.py
# Wrapper for pixq.py to maintain compatibility with run_dials_pipeline.sh

import os
import sys
import importlib.util

# Find pixq.py in the current directory or the parent directory
pixq_path = None
if os.path.exists("pixq.py"):
    pixq_path = "pixq.py"
elif os.path.exists("../pixq.py"):
    pixq_path = "../pixq.py"
else:
    print("Error: Could not find pixq.py in current or parent directory.")
    sys.exit(1)

# Load pixq.py as a module
print(f"Loading q-map generator from: {pixq_path}")
spec = importlib.util.spec_from_file_location("pixq", pixq_path)
pixq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pixq)

# Call the main function from pixq.py
print("Running q-map generator...")
pixq.main()

# Call the verification function if available
try:
    # Test for a specific pixel (these coordinates match the defaults in pixq.py)
    test_beam_model, test_detector_model = pixq.load_dials_models(pixq.EXPERIMENT_FILE)
    panel_to_test_idx = 0  # Test the first panel (index 0)
    
    if panel_to_test_idx < len(test_detector_model):
        panel_model_for_test = test_detector_model[panel_to_test_idx]
        q_verified = pixq.verify_single_pixel_q(
            test_beam_model, panel_model_for_test, panel_to_test_idx, 2342, 30
        )
        print("Verification complete.")
except (AttributeError, NameError) as e:
    print(f"Note: Could not run verification function: {e}")

print("q-map generation completed successfully.")