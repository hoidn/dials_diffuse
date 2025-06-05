#!/usr/bin/env python3
"""
Visual diagnostic script for pixel mask generation (Module 1.S.2).

This script generates static and dynamic pixel masks and creates visualizations
to verify that masking is working correctly for different detector regions and
anomalous pixels.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plot_utils import (
    plot_detector_image,
    plot_multi_panel_comparison,
    ensure_output_dir,
    close_all_figures,
    setup_logging_for_plots
)

# Import diffusepipe components
from diffusepipe.masking.pixel_mask_generator import (
    PixelMaskGenerator,
    StaticMaskParams,
    DynamicMaskParams,
    Circle,
    Rectangle,
    create_default_static_params,
    create_default_dynamic_params
)

logger = logging.getLogger(__name__)


def load_detector_from_expt(expt_path: str):
    """
    Load detector model from DIALS experiment file.
    
    Args:
        expt_path: Path to .expt file
        
    Returns:
        DIALS Detector object or None if failed
    """
    try:
        from dxtbx.model.experiment_list import ExperimentListFactory
        
        experiments = ExperimentListFactory.from_json_file(expt_path)
        detector = experiments[0].detector
        
        logger.info(f"Loaded detector with {len(detector)} panels")
        return detector
        
    except ImportError as e:
        logger.error(f"Failed to import DIALS: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load detector from {expt_path}: {e}")
        return None


def load_representative_images(image_paths: list):
    """
    Load representative images for dynamic mask generation.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        List of ImageSet objects
    """
    try:
        from dxtbx.imageset import ImageSetFactory
        
        image_sets = []
        for image_path in image_paths:
            if Path(image_path).exists():
                image_set = ImageSetFactory.new([image_path])[0]
                image_sets.append(image_set)
                logger.info(f"Loaded image: {image_path}")
            else:
                logger.warning(f"Image not found: {image_path}")
        
        return image_sets
        
    except ImportError as e:
        logger.error(f"Failed to import dxtbx: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        return []


def parse_mask_config(config_str: str) -> Dict[str, Any]:
    """
    Parse mask configuration from JSON string or file.
    
    Args:
        config_str: JSON string or path to JSON file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Try to parse as JSON string first
        if config_str.strip().startswith('{'):
            return json.loads(config_str)
        
        # Otherwise treat as file path
        config_path = Path(config_str)
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file not found: {config_str}")
            return {}
            
    except Exception as e:
        logger.error(f"Failed to parse mask config: {e}")
        return {}


def create_static_mask_params(config: Dict[str, Any]) -> StaticMaskParams:
    """
    Create StaticMaskParams from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        StaticMaskParams object
    """
    beamstop = None
    if 'beamstop' in config:
        bs_config = config['beamstop']
        if bs_config.get('type') == 'circle':
            beamstop = Circle(
                center_x=bs_config['center_x'],
                center_y=bs_config['center_y'],
                radius=bs_config['radius']
            )
        elif bs_config.get('type') == 'rectangle':
            beamstop = Rectangle(
                min_x=bs_config['min_x'],
                max_x=bs_config['max_x'],
                min_y=bs_config['min_y'],
                max_y=bs_config['max_y']
            )
    
    untrusted_rects = None
    if 'untrusted_rects' in config:
        untrusted_rects = []
        for rect_config in config['untrusted_rects']:
            rect = Rectangle(
                min_x=rect_config['min_x'],
                max_x=rect_config['max_x'],
                min_y=rect_config['min_y'],
                max_y=rect_config['max_y']
            )
            untrusted_rects.append(rect)
    
    untrusted_panels = config.get('untrusted_panels')
    
    return StaticMaskParams(
        beamstop=beamstop,
        untrusted_rects=untrusted_rects,
        untrusted_panels=untrusted_panels
    )


def create_dynamic_mask_params(config: Dict[str, Any]) -> DynamicMaskParams:
    """
    Create DynamicMaskParams from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DynamicMaskParams object
    """
    return DynamicMaskParams(
        hot_pixel_thresh=config.get('hot_pixel_thresh', 1e6),
        negative_pixel_tolerance=config.get('negative_pixel_tolerance', 0.0),
        max_fraction_bad_pixels=config.get('max_fraction_bad_pixels', 0.1)
    )


def create_pixel_mask_visualizations(
    detector,
    representative_images: list,
    static_params: StaticMaskParams,
    dynamic_params: DynamicMaskParams,
    output_dir: str
) -> bool:
    """
    Create visualizations for pixel mask generation.
    
    Args:
        detector: DIALS Detector object
        representative_images: List of ImageSet objects for dynamic analysis
        static_params: Static mask parameters
        dynamic_params: Dynamic mask parameters
        output_dir: Directory to save plots
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = ensure_output_dir(output_dir)
        generator = PixelMaskGenerator()
        
        # Generate masks
        logger.info("Generating static mask...")
        static_mask = generator.generate_static_mask(detector, static_params)
        
        logger.info("Generating dynamic mask...")
        dynamic_mask = generator.generate_dynamic_mask(
            detector, representative_images, dynamic_params
        )
        
        logger.info("Generating combined mask...")
        combined_mask = generator.generate_combined_pixel_mask(
            detector, static_params, representative_images, dynamic_params
        )
        
        # Create visualizations for each panel
        for panel_idx in range(len(detector)):
            logger.info(f"Creating visualizations for panel {panel_idx}...")
            
            # Get panel masks
            static_panel = static_mask[panel_idx]
            dynamic_panel = dynamic_mask[panel_idx]
            combined_panel = combined_mask[panel_idx]
            
            # Plot individual masks
            plot_detector_image(
                static_panel,
                title=f"Panel {panel_idx}: Static Mask",
                output_path=str(output_path / f"panel_{panel_idx}_static_mask.png"),
                cmap='RdYlBu'
            )
            
            plot_detector_image(
                dynamic_panel,
                title=f"Panel {panel_idx}: Dynamic Mask",
                output_path=str(output_path / f"panel_{panel_idx}_dynamic_mask.png"),
                cmap='RdYlBu'
            )
            
            plot_detector_image(
                combined_panel,
                title=f"Panel {panel_idx}: Combined Mask",
                output_path=str(output_path / f"panel_{panel_idx}_combined_mask.png"),
                cmap='RdYlBu'
            )
            
            # Plot comparison
            plot_multi_panel_comparison(
                [static_panel, dynamic_panel, combined_panel],
                [f"Panel {panel_idx}: Static", f"Panel {panel_idx}: Dynamic", f"Panel {panel_idx}: Combined"],
                output_path=str(output_path / f"panel_{panel_idx}_mask_comparison.png"),
                figsize=(18, 6),
                cmap='RdYlBu'
            )
        
        # Generate summary information
        info_file = output_path / "pixel_mask_info.txt"
        with open(info_file, 'w') as f:
            f.write("Pixel Mask Generation Visual Check Results\n")
            f.write("=" * 45 + "\n\n")
            
            f.write(f"Number of detector panels: {len(detector)}\n\n")
            
            # Statistics for each panel
            for panel_idx in range(len(detector)):
                panel_size = detector[panel_idx].get_image_size()
                total_pixels = panel_size[0] * panel_size[1]
                
                static_good = static_mask[panel_idx].count(True)
                dynamic_good = dynamic_mask[panel_idx].count(True)
                combined_good = combined_mask[panel_idx].count(True)
                
                f.write(f"Panel {panel_idx} ({panel_size[0]}x{panel_size[1]} pixels):\n")
                f.write(f"  Total pixels: {total_pixels}\n")
                f.write(f"  Static mask good pixels: {static_good} ({static_good/total_pixels*100:.1f}%)\n")
                f.write(f"  Dynamic mask good pixels: {dynamic_good} ({dynamic_good/total_pixels*100:.1f}%)\n")
                f.write(f"  Combined mask good pixels: {combined_good} ({combined_good/total_pixels*100:.1f}%)\n")
                f.write(f"  Static mask rejected: {total_pixels - static_good} pixels\n")
                f.write(f"  Dynamic mask rejected: {total_pixels - dynamic_good} pixels\n")
                f.write(f"  Total rejected: {total_pixels - combined_good} pixels\n\n")
            
            # Configuration summary
            f.write("Configuration used:\n")
            f.write(f"  Static mask parameters:\n")
            f.write(f"    Beamstop: {static_params.beamstop}\n")
            f.write(f"    Untrusted rectangles: {len(static_params.untrusted_rects) if static_params.untrusted_rects else 0}\n")
            f.write(f"    Untrusted panels: {static_params.untrusted_panels}\n")
            f.write(f"  Dynamic mask parameters:\n")
            f.write(f"    Hot pixel threshold: {dynamic_params.hot_pixel_thresh}\n")
            f.write(f"    Negative pixel tolerance: {dynamic_params.negative_pixel_tolerance}\n")
            f.write(f"    Max bad pixel fraction: {dynamic_params.max_fraction_bad_pixels}\n")
        
        logger.info(f"Pixel mask info saved to {info_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create pixel mask visualizations: {e}")
        return False


def main():
    """Main function for pixel mask visual checks."""
    parser = argparse.ArgumentParser(
        description="Visual diagnostic for pixel mask generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic pixel mask check with default parameters
  python check_pixel_masks.py --expt detector.expt --images image1.cbf image2.cbf --output-dir pixel_mask_check
  
  # With custom static mask configuration
  python check_pixel_masks.py --expt detector.expt --images image1.cbf \\
    --static-config '{"beamstop": {"type": "circle", "center_x": 1250, "center_y": 1250, "radius": 50}}' \\
    --output-dir pixel_mask_check
  
  # Using existing test data
  python check_pixel_masks.py \\
    --expt ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt \\
    --images ../../747/lys_nitr_10_6_0491.cbf \\
    --output-dir pixel_mask_visual_check
        """
    )
    
    parser.add_argument(
        '--expt',
        required=True,
        help='Path to DIALS experiment file (.expt) containing detector model'
    )
    
    parser.add_argument(
        '--images',
        nargs='+',
        required=True,
        help='Path(s) to representative image files for dynamic mask generation'
    )
    
    parser.add_argument(
        '--static-config',
        help='Static mask configuration (JSON string or file path)'
    )
    
    parser.add_argument(
        '--dynamic-config',
        help='Dynamic mask configuration (JSON string or file path)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='pixel_mask_visual_check',
        help='Output directory for plots (default: pixel_mask_visual_check)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging_for_plots()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting pixel mask visual check...")
    
    # Validate input files
    if not Path(args.expt).exists():
        logger.error(f"Experiment file not found: {args.expt}")
        return 1
    
    try:
        # Load detector
        logger.info(f"Loading detector from: {args.expt}")
        detector = load_detector_from_expt(args.expt)
        if detector is None:
            logger.error("Failed to load detector")
            return 1
        
        # Load representative images
        logger.info(f"Loading representative images: {args.images}")
        representative_images = load_representative_images(args.images)
        if not representative_images:
            logger.warning("No images loaded, dynamic mask will be empty")
        
        # Parse configurations
        static_config = {}
        dynamic_config = {}
        
        if args.static_config:
            static_config = parse_mask_config(args.static_config)
        
        if args.dynamic_config:
            dynamic_config = parse_mask_config(args.dynamic_config)
        
        # Create mask parameters
        static_params = create_static_mask_params(static_config) if static_config else create_default_static_params()
        dynamic_params = create_dynamic_mask_params(dynamic_config) if dynamic_config else create_default_dynamic_params()
        
        logger.info(f"Static mask config: {static_params}")
        logger.info(f"Dynamic mask config: {dynamic_params}")
        
        # Create visualizations
        logger.info(f"Creating visualizations in: {args.output_dir}")
        success = create_pixel_mask_visualizations(
            detector, representative_images, static_params, dynamic_params, args.output_dir
        )
        
        if success:
            logger.info(f"✅ Pixel mask visual check completed successfully!")
            logger.info(f"📁 Check output files in: {args.output_dir}")
            return 0
        else:
            logger.error("❌ Pixel mask visual check failed")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    finally:
        close_all_figures()


if __name__ == "__main__":
    sys.exit(main())