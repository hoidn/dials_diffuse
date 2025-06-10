#!/usr/bin/env python3
"""Simple fix for Q-consistency validation - replace the problematic method."""

import logging
import numpy as np
from typing import Tuple, Dict

def simple_position_consistency_check(reflections, tolerance: float = 2.0) -> Tuple[bool, Dict[str, float]]:
    """
    Check consistency between calculated and observed reflection positions.
    
    This uses pixel position differences instead of complex Q-vector calculations.
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Performing reflection position consistency check")
        
        # Check what position columns are available
        has_observed = 'xyzobs.px.value' in reflections
        has_calculated = 'xyzcal.px' in reflections
        
        if not (has_observed and has_calculated):
            logger.warning("Missing required position columns for consistency check")
            return False, {'count': 0, 'mean': None, 'max': None, 'median': None}
        
        # Use a representative subset of reflections for testing
        n_total = len(reflections)
        n_test = min(n_total, 500)  # Test up to 500 reflections
        
        if n_total == 0:
            logger.warning("No reflections available for consistency check")
            return False, {'count': 0, 'mean': None, 'max': None, 'median': None}
        
        # Select random indices for testing
        indices = list(range(n_total))
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        
        position_differences = []
        
        for idx in test_indices:
            try:
                # Get observed and calculated positions (in pixels)
                obs_pos = reflections['xyzobs.px.value'][idx]
                calc_pos = reflections['xyzcal.px'][idx]
                
                # Calculate difference in XY position (ignore Z/frame difference)
                dx = obs_pos[0] - calc_pos[0]
                dy = obs_pos[1] - calc_pos[1]
                position_diff = np.sqrt(dx*dx + dy*dy)  # Euclidean distance in pixels
                
                position_differences.append(position_diff)
                
                # Debug: Log first few values
                if len(position_differences) <= 3:
                    logger.debug(f"Reflection {idx}: obs=({obs_pos[0]:.1f},{obs_pos[1]:.1f}), "
                               f"calc=({calc_pos[0]:.1f},{calc_pos[1]:.1f}), diff={position_diff:.3f} px")
                    
            except Exception as e:
                logger.debug(f"Failed to process reflection {idx}: {e}")
                continue
        
        if not position_differences:
            logger.error("No valid reflections for position consistency check")
            return False, {'count': 0, 'mean': None, 'max': None, 'median': None}
        
        # Calculate statistics (in pixels)
        diff_array = np.array(position_differences)
        stats = {
            'mean': float(np.mean(diff_array)),
            'max': float(np.max(diff_array)),
            'median': float(np.median(diff_array)),
            'count': len(position_differences)
        }
        
        # For pixel position differences, a reasonable tolerance is ~1-2 pixels
        pixel_tolerance = tolerance  # Default 2.0 pixels
        
        # Check against tolerance
        passed = stats['mean'] <= pixel_tolerance and stats['max'] <= (pixel_tolerance * 3)
        
        logger.info(f"Position consistency: mean = {stats['mean']:.3f} px, max = {stats['max']:.3f} px, "
                   f"tolerance = {pixel_tolerance:.1f} px, passed = {passed}")
        
        return passed, stats
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Position consistency check error: {e}")
        return False, {'count': 0, 'mean': None, 'max': None, 'median': None}

if __name__ == "__main__":
    print("Simple validation fix ready. Import and use simple_position_consistency_check()")