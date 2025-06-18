"""
DiffuseDataMerger implementation for Phase 3 merging.

Applies refined scaling and merges observations into final voxel data.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

from diffusepipe.voxelization.global_voxel_grid import GlobalVoxelGrid
from diffusepipe.scaling.diffuse_scaling_model import DiffuseScalingModel

logger = logging.getLogger(__name__)


@dataclass
class VoxelDataRelative:
    """Relatively-scaled voxel data structure."""

    voxel_indices: np.ndarray
    H_center: np.ndarray
    K_center: np.ndarray
    L_center: np.ndarray
    q_center_x: np.ndarray
    q_center_y: np.ndarray
    q_center_z: np.ndarray
    q_magnitude_center: np.ndarray
    I_merged_relative: np.ndarray
    Sigma_merged_relative: np.ndarray
    num_observations: np.ndarray


class DiffuseDataMerger:
    """
    Applies refined scaling and merges diffuse observations into voxels.

    Handles scaling application, error propagation, and weighted merging
    for the final relatively-scaled 3D diffuse scattering map.
    """

    def __init__(self, global_voxel_grid: GlobalVoxelGrid):
        """
        Initialize merger with voxel grid.

        Args:
            global_voxel_grid: Grid definition for coordinate calculations
        """
        self.global_voxel_grid = global_voxel_grid
        logger.info("DiffuseDataMerger initialized")

    def vectorized_merge_scaled_data(
        self,
        binned_pixel_data: Dict,
        scaling_model: DiffuseScalingModel,
        merge_config: Dict,
    ) -> VoxelDataRelative:
        """
        VECTORIZED version: Apply scaling and merge observations using pandas and vectorized operations.
        
        Replaces slow loop-within-loop approach with efficient split-apply-combine strategy.
        
        Args:
            binned_pixel_data: Voxel-organized observations
            scaling_model: Refined scaling model from Module 3.S.3
            merge_config: Merging configuration

        Returns:
            VoxelDataRelative with merged, relatively-scaled data
        """
        logger.info(f"VECTORIZED merging data for {len(binned_pixel_data)} voxels")

        # Configuration
        outlier_rejection = merge_config.get("outlier_rejection", {"enabled": False})
        min_observations = merge_config.get("minimum_observations", 1)
        weight_method = merge_config.get("weight_method", "inverse_variance")

        # Step 1: Aggregate all observations into a single DataFrame
        logger.debug("Aggregating all observations into DataFrame...")
        df = self._create_observations_dataframe(binned_pixel_data)
        
        if len(df) == 0:
            logger.warning("No observations to merge")
            return self._create_empty_voxel_data()

        # Step 2: Vectorized scaling application
        logger.debug(f"Applying vectorized scaling to {len(df)} observations...")
        df = self._apply_vectorized_scaling(df, scaling_model)

        # Step 3: Filter voxels by minimum observations
        voxel_counts = df.groupby('voxel_idx').size()
        valid_voxels = voxel_counts[voxel_counts >= min_observations].index
        df_filtered = df[df['voxel_idx'].isin(valid_voxels)]
        
        if len(df_filtered) == 0:
            logger.warning("No voxels passed minimum observation criteria")
            return self._create_empty_voxel_data()

        # Step 4: Apply outlier rejection if enabled (vectorized per voxel)
        if outlier_rejection.get("enabled", False):
            df_filtered = self._vectorized_outlier_rejection(
                df_filtered, outlier_rejection.get("sigma_threshold", 3.0)
            )

        # Step 5: Grouped merging with vectorized operations
        logger.debug("Performing grouped merging with pandas...")
        merged_results = self._perform_grouped_merging(df_filtered, weight_method, min_observations)
        
        if len(merged_results) == 0:
            logger.warning("No voxels passed merging criteria")
            return self._create_empty_voxel_data()

        # Step 6: Calculate voxel coordinates and create final structure
        voxel_indices = merged_results.index.tolist()
        coordinates = self.calculate_voxel_coordinates(voxel_indices)

        # Create final data structure
        voxel_data = VoxelDataRelative(
            voxel_indices=np.array(voxel_indices),
            H_center=coordinates["H_center"],
            K_center=coordinates["K_center"],
            L_center=coordinates["L_center"],
            q_center_x=coordinates["q_center_x"],
            q_center_y=coordinates["q_center_y"],
            q_center_z=coordinates["q_center_z"],
            q_magnitude_center=coordinates["q_magnitude_center"],
            I_merged_relative=merged_results['merged_intensity'].values,
            Sigma_merged_relative=merged_results['merged_sigma'].values,
            num_observations=merged_results['n_observations'].values,
        )

        processed_voxels = len(voxel_data.voxel_indices)
        skipped_voxels = len(binned_pixel_data) - processed_voxels
        logger.info(f"VECTORIZED merged {processed_voxels} voxels, skipped {skipped_voxels}")
        return voxel_data

    def merge_scaled_data(
        self,
        binned_pixel_data: Dict,
        scaling_model: DiffuseScalingModel,
        merge_config: Dict,
    ) -> VoxelDataRelative:
        """
        Apply scaling and merge observations into final voxel data.
        
        By default, uses the optimized vectorized implementation for superior performance.
        Set merge_config["use_legacy"] = True to use the original loop-based approach.

        Args:
            binned_pixel_data: Voxel-organized observations
            scaling_model: Refined scaling model from Module 3.S.3
            merge_config: Merging configuration

        Returns:
            VoxelDataRelative with merged, relatively-scaled data
        """
        # Use vectorized implementation by default for performance
        use_legacy = merge_config.get("use_legacy", False)
        
        if use_legacy:
            logger.info("Using legacy loop-based merging implementation")
            return self._legacy_merge_scaled_data(binned_pixel_data, scaling_model, merge_config)
        else:
            return self.vectorized_merge_scaled_data(binned_pixel_data, scaling_model, merge_config)

    def _legacy_merge_scaled_data(
        self,
        binned_pixel_data: Dict,
        scaling_model: DiffuseScalingModel,
        merge_config: Dict,
    ) -> VoxelDataRelative:
        """
        Legacy implementation: Apply scaling and merge observations into final voxel data.
        
        This is the original loop-based implementation, kept for compatibility and testing.

        Args:
            binned_pixel_data: Voxel-organized observations
            scaling_model: Refined scaling model from Module 3.S.3
            merge_config: Merging configuration

        Returns:
            VoxelDataRelative with merged, relatively-scaled data
        """
        logger.info(f"Merging data for {len(binned_pixel_data)} voxels")

        # Configuration
        outlier_rejection = merge_config.get("outlier_rejection", {"enabled": False})
        min_observations = merge_config.get("minimum_observations", 1)
        weight_method = merge_config.get("weight_method", "inverse_variance")

        # Storage for merged results
        merged_voxel_indices = []
        merged_intensities = []
        merged_sigmas = []
        merged_obs_counts = []

        processed_voxels = 0
        skipped_voxels = 0

        for voxel_idx, voxel_data in binned_pixel_data.items():
            observations = voxel_data["observations"]

            if len(observations) < min_observations:
                skipped_voxels += 1
                continue

            # Apply scaling to all observations in this voxel
            scaled_observations = []

            for obs in observations:
                try:
                    scaled_intensity, scaled_sigma = self.apply_scaling_to_observation(
                        obs, scaling_model
                    )
                    scaled_observations.append((scaled_intensity, scaled_sigma))
                except Exception as e:
                    logger.warning(
                        f"Error scaling observation in voxel {voxel_idx}: {e}"
                    )
                    continue

            if len(scaled_observations) < min_observations:
                skipped_voxels += 1
                continue

            # Apply outlier rejection if enabled
            if outlier_rejection.get("enabled", False):
                scaled_observations = self._reject_outliers(
                    scaled_observations, outlier_rejection.get("sigma_threshold", 3.0)
                )

                if len(scaled_observations) < min_observations:
                    skipped_voxels += 1
                    continue

            # Perform weighted merge for this voxel
            merged_intensity, merged_sigma, n_obs = self.weighted_merge_voxel(
                scaled_observations, weight_method
            )

            # Store results
            merged_voxel_indices.append(voxel_idx)
            merged_intensities.append(merged_intensity)
            merged_sigmas.append(merged_sigma)
            merged_obs_counts.append(n_obs)

            processed_voxels += 1

        logger.info(f"Merged {processed_voxels} voxels, skipped {skipped_voxels}")

        if processed_voxels == 0:
            logger.warning("No voxels passed merging criteria")
            return self._create_empty_voxel_data()

        # Calculate voxel coordinates
        coordinates = self.calculate_voxel_coordinates(merged_voxel_indices)

        # Create final data structure
        voxel_data = VoxelDataRelative(
            voxel_indices=np.array(merged_voxel_indices),
            H_center=coordinates["H_center"],
            K_center=coordinates["K_center"],
            L_center=coordinates["L_center"],
            q_center_x=coordinates["q_center_x"],
            q_center_y=coordinates["q_center_y"],
            q_center_z=coordinates["q_center_z"],
            q_magnitude_center=coordinates["q_magnitude_center"],
            I_merged_relative=np.array(merged_intensities),
            Sigma_merged_relative=np.array(merged_sigmas),
            num_observations=np.array(merged_obs_counts),
        )

        logger.info(f"Final merged dataset: {len(voxel_data.voxel_indices)} voxels")
        return voxel_data

    def apply_scaling_to_observation(
        self, observation: Dict, scaling_model: DiffuseScalingModel
    ) -> Tuple[float, float]:
        """
        Apply scaling to a single observation.

        Args:
            observation: Observation data
            scaling_model: Refined scaling model

        Returns:
            Tuple of (scaled_intensity, scaled_sigma)
        """
        # Extract observation properties
        intensity = observation["intensity"]
        sigma = observation["sigma"]
        still_id = observation["still_id"]
        q_vector_lab = observation["q_vector_lab"]

        # Calculate q-magnitude
        q_magnitude = np.linalg.norm(q_vector_lab)

        # Get scaling parameters from model
        multiplicative_scale, additive_offset = (
            scaling_model.get_scales_for_observation(still_id, q_magnitude)
        )

        # Verify v1 model constraints (additive offset should be ~0)
        if abs(additive_offset) > 1e-9:
            logger.warning(
                f"v1 model violation: additive_offset={additive_offset:.2e} "
                f"(should be ~0)"
            )

        # Apply v1 scaling: I_final = (I_obs - C_i) / M_i
        # For v1: C_i ≈ 0, so I_final = I_obs / M_i
        scaled_intensity = (intensity - additive_offset) / multiplicative_scale

        # Propagate uncertainty: Sigma_final = Sigma_obs / |M_i|
        # Note: This is simplified for v1 where C_i has no uncertainty
        scaled_sigma = sigma / abs(multiplicative_scale)

        return scaled_intensity, scaled_sigma

    def weighted_merge_voxel(
        self,
        scaled_observations: List[Tuple[float, float]],
        weight_method: str = "inverse_variance",
    ) -> Tuple[float, float, int]:
        """
        Perform weighted merge of observations within a voxel.

        Args:
            scaled_observations: List of (intensity, sigma) tuples
            weight_method: Weighting method for merging

        Returns:
            Tuple of (merged_intensity, merged_sigma, n_observations)
        """
        if not scaled_observations:
            raise ValueError("No observations to merge")

        # Validate weight method first
        if weight_method not in ["inverse_variance", "uniform"]:
            raise ValueError(f"Unknown weight method: {weight_method}")

        n_obs = len(scaled_observations)

        if n_obs == 1:
            # Single observation - no merging needed
            intensity, sigma = scaled_observations[0]
            return intensity, sigma, 1

        intensities = np.array([obs[0] for obs in scaled_observations])
        sigmas = np.array([obs[1] for obs in scaled_observations])

        # Calculate weights
        if weight_method == "inverse_variance":
            # Weights = 1 / sigma^2
            variances = sigmas**2
            weights = 1.0 / (
                variances + 1e-10
            )  # Add small value to avoid division by zero
        elif weight_method == "uniform":
            # Equal weights
            weights = np.ones(n_obs)

        # Handle edge cases
        if np.sum(weights) <= 0:
            logger.warning("All weights are zero or negative, using uniform weights")
            weights = np.ones(n_obs)

        # Weighted average
        total_weight = np.sum(weights)
        weighted_intensities = intensities * weights
        merged_intensity = np.sum(weighted_intensities) / total_weight

        # Error propagation for weighted average
        if weight_method == "inverse_variance":
            # For inverse variance weighting: sigma_merged = 1 / sqrt(sum(weights))
            merged_sigma = 1.0 / np.sqrt(total_weight)
        else:
            # For uniform weighting: sigma_merged = sqrt(sum(sigma_i^2)) / N
            merged_sigma = np.sqrt(np.sum(sigmas**2)) / n_obs

        return merged_intensity, merged_sigma, n_obs

    def _reject_outliers(
        self, scaled_observations: List[Tuple[float, float]], sigma_threshold: float
    ) -> List[Tuple[float, float]]:
        """Reject outlier observations based on sigma threshold."""
        if len(scaled_observations) <= 2:
            return scaled_observations  # Can't reject from very small samples

        intensities = np.array([obs[0] for obs in scaled_observations])

        # Calculate robust statistics
        median_intensity = np.median(intensities)
        mad = np.median(
            np.abs(intensities - median_intensity)
        )  # Median absolute deviation
        robust_std = 1.4826 * mad  # Scale factor for normal distribution

        # Filter observations
        filtered_observations = []
        for obs in scaled_observations:
            intensity = obs[0]
            deviation = abs(intensity - median_intensity)

            if deviation <= sigma_threshold * robust_std:
                filtered_observations.append(obs)

        if len(filtered_observations) == 0:
            logger.warning("All observations rejected as outliers, keeping original")
            return scaled_observations

        n_rejected = len(scaled_observations) - len(filtered_observations)
        if n_rejected > 0:
            logger.debug(f"Rejected {n_rejected} outlier observations")

        return filtered_observations

    def calculate_voxel_coordinates(
        self, voxel_indices: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate coordinate arrays for voxel centers.

        Args:
            voxel_indices: List of voxel indices

        Returns:
            Dictionary of coordinate arrays
        """
        n_voxels = len(voxel_indices)

        # Initialize coordinate arrays
        H_centers = np.zeros(n_voxels)
        K_centers = np.zeros(n_voxels)
        L_centers = np.zeros(n_voxels)
        q_x_centers = np.zeros(n_voxels)
        q_y_centers = np.zeros(n_voxels)
        q_z_centers = np.zeros(n_voxels)
        q_magnitudes = np.zeros(n_voxels)

        for i, voxel_idx in enumerate(voxel_indices):
            # Get HKL center coordinates
            h, k, l = self.global_voxel_grid.voxel_idx_to_hkl_center(voxel_idx)
            H_centers[i] = h
            K_centers[i] = k
            L_centers[i] = l

            # Get lab-frame q-vector for center
            q_center = self.global_voxel_grid.get_q_vector_for_voxel_center(voxel_idx)
            q_x_centers[i] = q_center.elems[0]
            q_y_centers[i] = q_center.elems[1]
            q_z_centers[i] = q_center.elems[2]
            q_magnitudes[i] = q_center.length()

        return {
            "H_center": H_centers,
            "K_center": K_centers,
            "L_center": L_centers,
            "q_center_x": q_x_centers,
            "q_center_y": q_y_centers,
            "q_center_z": q_z_centers,
            "q_magnitude_center": q_magnitudes,
        }

    def get_merge_statistics(self, voxel_data_relative: VoxelDataRelative) -> Dict:
        """
        Calculate comprehensive statistics about merged data.

        Args:
            voxel_data_relative: Merged voxel data

        Returns:
            Dictionary of statistics
        """
        intensities = voxel_data_relative.I_merged_relative
        sigmas = voxel_data_relative.Sigma_merged_relative
        obs_counts = voxel_data_relative.num_observations
        q_magnitudes = voxel_data_relative.q_magnitude_center

        # Basic statistics
        total_voxels = len(intensities)
        total_observations = np.sum(obs_counts)

        # Intensity statistics
        intensity_stats = {
            "mean": float(np.mean(intensities)),
            "std": float(np.std(intensities)),
            "min": float(np.min(intensities)),
            "max": float(np.max(intensities)),
            "median": float(np.median(intensities)),
        }

        # Observation statistics
        obs_stats = {
            "mean_per_voxel": float(np.mean(obs_counts)),
            "total_observations": int(total_observations),
            "voxels_with_single_obs": int(np.sum(obs_counts == 1)),
            "max_observations_per_voxel": int(np.max(obs_counts)),
        }

        # Resolution coverage
        resolution_stats = {
            "q_min": float(np.min(q_magnitudes)),
            "q_max": float(np.max(q_magnitudes)),
            "mean_q": float(np.mean(q_magnitudes)),
        }

        # Data quality metrics
        sigma_over_intensity = sigmas / (intensities + 1e-10)  # Relative uncertainty
        high_intensity_threshold = intensity_stats["mean"] + 2 * intensity_stats["std"]
        good_precision_threshold = 0.1  # 10% relative uncertainty

        quality_stats = {
            "mean_sigma_over_intensity": float(np.mean(sigma_over_intensity)),
            "high_intensity_voxels": int(
                np.sum(intensities > high_intensity_threshold)
            ),
            "low_sigma_voxels": int(
                np.sum(sigma_over_intensity < good_precision_threshold)
            ),
        }

        return {
            "total_voxels": total_voxels,
            "intensity_statistics": intensity_stats,
            "observation_statistics": obs_stats,
            "resolution_coverage": resolution_stats,
            "data_quality": quality_stats,
        }

    def _create_empty_voxel_data(self) -> VoxelDataRelative:
        """Create empty VoxelDataRelative for edge cases."""
        return VoxelDataRelative(
            voxel_indices=np.array([]),
            H_center=np.array([]),
            K_center=np.array([]),
            L_center=np.array([]),
            q_center_x=np.array([]),
            q_center_y=np.array([]),
            q_center_z=np.array([]),
            q_magnitude_center=np.array([]),
            I_merged_relative=np.array([]),
            Sigma_merged_relative=np.array([]),
            num_observations=np.array([]),
        )

    def save_voxel_data(
        self, voxel_data: VoxelDataRelative, output_path: str, format: str = "npz"
    ):
        """
        Save merged voxel data to file.

        Args:
            voxel_data: Merged data to save
            output_path: Output file path
            format: Output format ("npz" or "hdf5")
        """
        if format == "npz":
            np.savez_compressed(
                output_path,
                voxel_indices=voxel_data.voxel_indices,
                H_center=voxel_data.H_center,
                K_center=voxel_data.K_center,
                L_center=voxel_data.L_center,
                q_center_x=voxel_data.q_center_x,
                q_center_y=voxel_data.q_center_y,
                q_center_z=voxel_data.q_center_z,
                q_magnitude_center=voxel_data.q_magnitude_center,
                I_merged_relative=voxel_data.I_merged_relative,
                Sigma_merged_relative=voxel_data.Sigma_merged_relative,
                num_observations=voxel_data.num_observations,
            )
            logger.info(f"Saved voxel data to {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _create_observations_dataframe(self, binned_pixel_data: Dict) -> pd.DataFrame:
        """
        Create pandas DataFrame from all observations for vectorized processing.
        
        Args:
            binned_pixel_data: Voxel-organized observations
            
        Returns:
            DataFrame with columns: voxel_idx, intensity, sigma, still_id, q_mag
        """
        observations_list = []
        
        for voxel_idx, voxel_data in binned_pixel_data.items():
            # Normalize data format for backward compatibility
            if "observations" in voxel_data:
                # Old format: list of observation dictionaries
                observations = voxel_data["observations"]
                for obs in observations:
                    q_mag = np.linalg.norm(obs["q_vector_lab"])
                    observations_list.append({
                        'voxel_idx': voxel_idx,
                        'intensity': obs["intensity"],
                        'sigma': obs["sigma"],
                        'still_id': obs["still_id"],
                        'q_mag': q_mag
                    })
            else:
                # New format: NumPy arrays
                n_obs = voxel_data["n_observations"]
                if n_obs == 0:
                    continue
                    
                intensities = voxel_data["intensities"]
                sigmas = voxel_data["sigmas"]
                still_ids = voxel_data["still_ids"]
                q_vectors = voxel_data["q_vectors_lab"]
                q_mags = np.linalg.norm(q_vectors, axis=1)
                
                for i in range(n_obs):
                    observations_list.append({
                        'voxel_idx': voxel_idx,
                        'intensity': intensities[i],
                        'sigma': sigmas[i],
                        'still_id': still_ids[i],
                        'q_mag': q_mags[i]
                    })
        
        return pd.DataFrame(observations_list)

    def _apply_vectorized_scaling(self, df: pd.DataFrame, scaling_model: DiffuseScalingModel) -> pd.DataFrame:
        """
        Apply scaling to all observations using vectorized operations.
        
        Args:
            df: DataFrame with observations
            scaling_model: Refined scaling model
            
        Returns:
            DataFrame with additional columns: scaled_intensity, scaled_sigma
        """
        if len(df) == 0:
            return df
            
        # Vectorized scaling calculation
        still_ids = df['still_id'].values
        q_mags = df['q_mag'].values
        
        # Use the vectorized scaling method from the scaling model
        scales, _ = scaling_model.vectorized_calculate_scales_and_derivatives(still_ids, q_mags)
        
        # Apply scaling: I_scaled = I_obs / M, Sigma_scaled = Sigma_obs / |M|
        df = df.copy()  # Avoid modifying original
        df['scaled_intensity'] = df['intensity'] / scales
        df['scaled_sigma'] = df['sigma'] / np.abs(scales)
        
        return df

    def _vectorized_outlier_rejection(self, df: pd.DataFrame, sigma_threshold: float) -> pd.DataFrame:
        """
        Apply outlier rejection to each voxel group using vectorized operations.
        
        Args:
            df: DataFrame with scaled observations
            sigma_threshold: Sigma threshold for outlier rejection
            
        Returns:
            DataFrame with outliers removed
        """
        def reject_outliers_group(group):
            if len(group) <= 2:
                return group  # Can't reject from very small samples
            
            intensities = group['scaled_intensity'].values
            
            # Calculate robust statistics
            median_intensity = np.median(intensities)
            mad = np.median(np.abs(intensities - median_intensity))
            robust_std = 1.4826 * mad  # Scale factor for normal distribution
            
            # Filter observations
            deviation = np.abs(intensities - median_intensity)
            valid_mask = deviation <= sigma_threshold * robust_std
            
            filtered_group = group[valid_mask]
            
            if len(filtered_group) == 0:
                return group  # Keep original if all rejected
                
            return filtered_group
        
        return df.groupby('voxel_idx', group_keys=False).apply(reject_outliers_group)

    def _perform_grouped_merging(self, df: pd.DataFrame, weight_method: str, min_observations: int) -> pd.DataFrame:
        """
        Perform weighted merging for each voxel using pandas groupby.
        
        Args:
            df: DataFrame with scaled observations
            weight_method: Weighting method for merging
            min_observations: Minimum observations required per voxel
            
        Returns:
            DataFrame with merged results indexed by voxel_idx
        """
        def _weighted_average_group(group):
            """Helper function for weighted averaging within each voxel group."""
            n_obs = len(group)
            
            if n_obs < min_observations:
                return pd.Series({
                    'merged_intensity': np.nan,
                    'merged_sigma': np.nan,
                    'n_observations': 0
                })
            
            if n_obs == 1:
                # Single observation - no merging needed
                return pd.Series({
                    'merged_intensity': group['scaled_intensity'].iloc[0],
                    'merged_sigma': group['scaled_sigma'].iloc[0],
                    'n_observations': 1
                })
            
            # Extract arrays
            intensities = group['scaled_intensity'].values
            sigmas = group['scaled_sigma'].values
            
            # Calculate weights
            if weight_method == "inverse_variance":
                # Weights = 1 / sigma^2
                variances = sigmas**2
                weights = 1.0 / (variances + 1e-10)  # Add small value to avoid division by zero
            elif weight_method == "uniform":
                # Equal weights
                weights = np.ones(n_obs)
            else:
                # Fallback to uniform
                weights = np.ones(n_obs)
            
            # Handle edge cases
            if np.sum(weights) <= 0:
                weights = np.ones(n_obs)
            
            # Weighted average
            total_weight = np.sum(weights)
            weighted_intensities = intensities * weights
            merged_intensity = np.sum(weighted_intensities) / total_weight
            
            # Error propagation for weighted average
            if weight_method == "inverse_variance":
                # For inverse variance weighting: sigma_merged = 1 / sqrt(sum(weights))
                merged_sigma = 1.0 / np.sqrt(total_weight)
            else:
                # For uniform weighting: sigma_merged = sqrt(sum(sigma_i^2)) / N
                merged_sigma = np.sqrt(np.sum(sigmas**2)) / n_obs
            
            return pd.Series({
                'merged_intensity': merged_intensity,
                'merged_sigma': merged_sigma,
                'n_observations': n_obs
            })
        
        # Apply weighted averaging to each voxel group
        merged_results = df.groupby('voxel_idx').apply(_weighted_average_group, include_groups=False)
        
        # Filter out voxels that didn't meet criteria (NaN results)
        merged_results = merged_results.dropna()
        
        return merged_results
