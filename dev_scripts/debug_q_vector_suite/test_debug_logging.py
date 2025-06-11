#!/usr/bin/env python3

import sys
sys.path.append('src')

import logging

# Set debug logging for the validation module
logging.getLogger("diffusepipe.crystallography.still_processing_and_validation").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Run one failing image with debug logging
from diffusepipe.crystallography.still_processing_and_validation import StillProcessorAndValidatorComponent
from diffusepipe.adapters.dials_sequence_process_adapter import DIALSSequenceProcessAdapter

print("="*60)
print("STEP 1: CONFIRMING PATCHED CODE PATH IS EXECUTED")
print("="*60)

processor = StillProcessorAndValidatorComponent()

print("\nRunning pipeline with debug logging...")
result = processor.process_and_validate_still(
    '747/lys_nitr_10_6_0491.cbf',
    {'pdb_file': '6o2h.pdb'},
    extraction_config=None
)

print(f"\nResult status: {result.status}")
print(f"Q-consistency passed: {result.validation_metrics.get('q_consistency_passed')}")
print(f"Mean delta q: {result.validation_metrics.get('mean_delta_q_mag')}")