"""Q-vector calculator for diffuse scattering analysis."""

import logging
import os
import numpy as np

from diffusepipe.types.types_IDL import ComponentInputFiles, OperationOutcome

logger = logging.getLogger(__name__)


class QValueCalculator:
    """
    Calculator for generating per-pixel q-vector maps from DIALS experiment geometry.

    This class implements the q-vector calculation as specified in q_calculator_IDL.md,
    computing q-vectors for each detector pixel based on DIALS experiment geometry.
    """

    def __init__(self):
        """Initialize the QValueCalculator."""
        pass

    def calculate_q_map(
        self, inputs: ComponentInputFiles, output_prefix_for_q_maps: str
    ) -> OperationOutcome:
        """
        Generate q-vector maps for each detector pixel.

        Args:
            inputs: Input files containing DIALS experiment path
            output_prefix_for_q_maps: Prefix for output q-map files

        Returns:
            OperationOutcome with success/failure status and output artifact paths
        """
        try:
            # Validate inputs
            if not inputs.dials_expt_path:
                return OperationOutcome(
                    status="FAILURE",
                    error_code="InputFileError",
                    message="DIALS experiment path not provided in inputs",
                )

            if not os.path.exists(inputs.dials_expt_path):
                return OperationOutcome(
                    status="FAILURE",
                    error_code="InputFileError",
                    message=f"DIALS experiment file not found: {inputs.dials_expt_path}",
                )

            logger.info(f"Loading DIALS experiment from: {inputs.dials_expt_path}")

            # Load DIALS experiment
            experiment_list = self._load_dials_experiment(inputs.dials_expt_path)
            if len(experiment_list) == 0:
                return OperationOutcome(
                    status="FAILURE",
                    error_code="DIALSModelError",
                    message="No experiments found in DIALS experiment file",
                )

            # Use the first experiment
            experiment = experiment_list[0]
            beam_model = experiment.beam
            detector_model = experiment.detector

            logger.info(f"Calculating q-maps for {len(detector_model)} detector panels")

            # Calculate q-vectors for all panels
            output_paths = {}

            for panel_idx, panel in enumerate(detector_model):
                logger.info(f"Processing panel {panel_idx}")

                qx_map, qy_map, qz_map = self._calculate_panel_q_vectors(
                    beam_model, panel
                )

                # Generate output file paths
                if len(detector_model) > 1:
                    # Multi-panel detector - include panel index
                    qx_path = f"{output_prefix_for_q_maps}_panel{panel_idx}_qx.npy"
                    qy_path = f"{output_prefix_for_q_maps}_panel{panel_idx}_qy.npy"
                    qz_path = f"{output_prefix_for_q_maps}_panel{panel_idx}_qz.npy"
                else:
                    # Single panel detector
                    qx_path = f"{output_prefix_for_q_maps}_qx.npy"
                    qy_path = f"{output_prefix_for_q_maps}_qy.npy"
                    qz_path = f"{output_prefix_for_q_maps}_qz.npy"

                # Save q-vector maps
                np.save(qx_path, qx_map)
                np.save(qy_path, qy_map)
                np.save(qz_path, qz_map)

                # Store paths in output artifacts
                if len(detector_model) > 1:
                    output_paths[f"panel{panel_idx}_qx_map_path"] = qx_path
                    output_paths[f"panel{panel_idx}_qy_map_path"] = qy_path
                    output_paths[f"panel{panel_idx}_qz_map_path"] = qz_path
                else:
                    output_paths["qx_map_path"] = qx_path
                    output_paths["qy_map_path"] = qy_path
                    output_paths["qz_map_path"] = qz_path

                logger.info(
                    f"Saved q-maps for panel {panel_idx}: {qx_path}, {qy_path}, {qz_path}"
                )

            return OperationOutcome(
                status="SUCCESS",
                message=f"Successfully generated q-maps for {len(detector_model)} panels",
                output_artifacts=output_paths,
            )

        except ImportError as e:
            logger.error(f"Failed to import DIALS/DXTBX modules: {e}")
            return OperationOutcome(
                status="FAILURE",
                error_code="DIALSModelError",
                message=f"Failed to import DIALS/DXTBX modules: {e}",
            )
        except Exception as e:
            logger.error(f"Q-vector calculation failed: {e}")
            if "Input" in str(e) or "not found" in str(e).lower():
                error_code = "InputFileError"
            elif "write" in str(e).lower() or "permission" in str(e).lower():
                error_code = "OutputWriteError"
            elif "calculation" in str(e).lower():
                error_code = "CalculationError"
            else:
                error_code = "DIALSModelError"

            return OperationOutcome(
                status="FAILURE",
                error_code=error_code,
                message=f"Q-vector calculation failed: {e}",
            )

    def _load_dials_experiment(self, expt_path: str):
        """
        Load DIALS experiment from file.

        Args:
            expt_path: Path to DIALS experiment file

        Returns:
            DIALS ExperimentList object

        Raises:
            ImportError: If DIALS/DXTBX modules cannot be imported
            Exception: If experiment file cannot be loaded
        """
        try:
            from dxtbx.model import ExperimentList
        except ImportError:
            raise ImportError("Failed to import dxtbx.model.ExperimentList")

        try:
            experiment_list = ExperimentList.from_file(expt_path)
            return experiment_list
        except Exception as e:
            raise Exception(f"Failed to load DIALS experiment from {expt_path}: {e}")

    def _calculate_panel_q_vectors(self, beam_model, panel):
        """
        Calculate q-vectors for all pixels in a detector panel.

        Args:
            beam_model: DIALS Beam object
            panel: DIALS Panel object

        Returns:
            Tuple of (qx_map, qy_map, qz_map) as NumPy arrays
        """
        # Get panel dimensions
        image_size = panel.get_image_size()  # (fast_scan_size, slow_scan_size)

        # Initialize q-vector arrays
        # Array dimensions are (slow_scan, fast_scan) to match NumPy convention
        qx_map = np.zeros((image_size[1], image_size[0]), dtype=np.float64)
        qy_map = np.zeros((image_size[1], image_size[0]), dtype=np.float64)
        qz_map = np.zeros((image_size[1], image_size[0]), dtype=np.float64)

        # Get beam parameters
        wavelength = beam_model.get_wavelength()  # Angstroms
        k_magnitude = 2 * np.pi / wavelength  # |k| = 2π/λ

        # Get incident beam vector (k_in)
        s0 = beam_model.get_s0()  # This is k_in / |k_in|
        k_in = np.array([s0[0], s0[1], s0[2]]) * k_magnitude

        logger.debug(f"Panel image size: {image_size}")
        logger.debug(f"Wavelength: {wavelength} Å")
        logger.debug(f"k magnitude: {k_magnitude} Å⁻¹")

        # Use chunked vectorized calculation for memory efficiency
        qx_map, qy_map, qz_map = self._chunked_vectorized_q_calculation(
            beam_model, panel, image_size, k_magnitude, k_in
        )

        logger.info(
            f"Calculated q-vectors for {image_size[0]} x {image_size[1]} pixels"
        )
        logger.debug(
            f"Q-vector ranges: qx=[{qx_map.min():.4f}, {qx_map.max():.4f}], "
            f"qy=[{qy_map.min():.4f}, {qy_map.max():.4f}], "
            f"qz=[{qz_map.min():.4f}, {qz_map.max():.4f}] Å⁻¹"
        )

        return qx_map, qy_map, qz_map

    def _chunked_vectorized_q_calculation(
        self, beam_model, panel, image_size, k_magnitude, k_in
    ):
        """
        Calculate q-vectors using chunked vectorized approach for memory efficiency.

        This method processes pixels in manageable chunks to balance performance
        with memory usage when dealing with large detector panels. Chunking is
        mandatory here to manage memory for full detector processing.

        Args:
            beam_model: DIALS Beam object
            panel: DIALS Panel object
            image_size: Tuple of (fast_scan_size, slow_scan_size)
            k_magnitude: Magnitude of k-vector
            k_in: Incident beam k-vector

        Returns:
            Tuple of (qx_map, qy_map, qz_map) as NumPy arrays
        """
        width, height = image_size[0], image_size[1]
        total_pixels = width * height

        # Initialize result arrays
        qx_map = np.zeros((height, width), dtype=np.float64)
        qy_map = np.zeros((height, width), dtype=np.float64)
        qz_map = np.zeros((height, width), dtype=np.float64)

        # Process in chunks to manage memory usage (mandatory for large detectors)
        chunk_size = 1_000_000  # 1M pixels per chunk

        # Create coordinate grids for entire detector
        fast_grid, slow_grid = np.meshgrid(
            np.arange(width), np.arange(height), indexing="xy"
        )
        fast_coords_flat = fast_grid.flatten()
        slow_coords_flat = slow_grid.flatten()

        # Collect results from all chunks
        all_qx = []
        all_qy = []
        all_qz = []

        # Process chunks
        for chunk_start in range(0, total_pixels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pixels)

            # Get coordinates for this chunk
            chunk_fast = fast_coords_flat[chunk_start:chunk_end]
            chunk_slow = slow_coords_flat[chunk_start:chunk_end]

            # Get lab coordinates for this chunk
            # Note: panel.get_pixel_lab_coord is not vectorized, so we need a loop here
            chunk_lab_coords = np.array(
                [
                    panel.get_pixel_lab_coord((f, s))
                    for f, s in zip(chunk_fast, chunk_slow)
                ]
            )

            # Vectorized calculation for this chunk
            scatter_directions = (
                chunk_lab_coords
                / np.linalg.norm(chunk_lab_coords, axis=1)[:, np.newaxis]
            )
            k_out_vectors = scatter_directions * k_magnitude
            q_vectors_chunk = k_out_vectors - k_in

            # Store chunk results
            all_qx.append(q_vectors_chunk[:, 0])
            all_qy.append(q_vectors_chunk[:, 1])
            all_qz.append(q_vectors_chunk[:, 2])

        # Concatenate all chunks and reshape to final 2D maps
        qx_flat_final = np.concatenate(all_qx)
        qy_flat_final = np.concatenate(all_qy)
        qz_flat_final = np.concatenate(all_qz)

        qx_map = qx_flat_final.reshape(height, width)
        qy_map = qy_flat_final.reshape(height, width)
        qz_map = qz_flat_final.reshape(height, width)

        return qx_map, qy_map, qz_map
