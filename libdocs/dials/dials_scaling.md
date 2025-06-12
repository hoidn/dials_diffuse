# DIALS Scaling Framework

This documentation covers the DIALS scaling framework for intensity correction, outlier rejection, and systematic error modeling in crystallographic data processing.

**Version Information:** Compatible with DIALS 3.x series. Some methods may differ in DIALS 2.x.

**Key Dependencies:**
- `dials.algorithms.scaling`: Main scaling framework
- `dials.array_family.flex`: Reflection tables and array operations
- `cctbx`: Unit cells, space groups, symmetry operations

## D.0. dials.algorithms.scaling Python Framework (Major)

**1. Purpose:**
Comprehensive framework for implementing custom scaling models, components, and refinement procedures for relative scaling of diffraction data in Module 3.S.3. Provides the infrastructure for building sophisticated scaling algorithms that can handle multi-crystal datasets and custom parameterizations.

**2. Primary Python Call(s):**
```python
from dials.algorithms.scaling.model import ScalingModelBase
from dials.algorithms.scaling.model.components.scale_components import ScaleComponentBase
from dials.algorithms.scaling.model.components.smooth_scale_components import (
    GaussianSmoother1D, GaussianSmoother2D, GaussianSmoother3D
)
from dials.algorithms.scaling.active_parameter_managers import (
    multi_active_parameter_manager,
    active_parameter_manager
)
from dials.algorithms.scaling.scaling_refiner import scaling_refinery
from dials.algorithms.scaling.target_function import ScalingTargetFunction

# Base classes for custom scaling models
class CustomScalingModel(ScalingModelBase):
    def __init__(self, configdict, is_scaled=False):
        super().__init__(configdict, is_scaled)
        # Initialize custom components
        
class CustomScaleComponent(ScaleComponentBase):
    def __init__(self, active_parameter_manager, params):
        super().__init__(active_parameter_manager, params)
        # Initialize component-specific parameters
```

**3. Key Framework Classes:**

**ScalingModelBase:**
- `__init__(self, configdict, is_scaled=False)`: Initialize scaling model
  - `configdict`: Dictionary with model configuration parameters
  - `is_scaled`: Boolean indicating if model has been applied to data
- `configure_components(self, reflection_table, experiment, params)`: Setup model components
- `_components`: Dictionary storing individual scale components
- `components`: Property returning list of active components
- `get_scales(self, reflection_table)` → flex.double: Calculate scale factors
- `get_inverse_scales(self, reflection_table)` → flex.double: Calculate inverse scales

**ScaleComponentBase:**
- `__init__(self, active_parameter_manager, params)`: Initialize component
- `calculate_scales_and_derivatives(self, reflection_table, block_id=None)`: Core scaling calculation
- `parameters`: Property accessing current parameter values
- `n_params`: Property returning number of parameters
- `update_reflection_data(self, reflection_table)`: Update component state

**GaussianSmoother Classes:**
- `GaussianSmoother1D(n_parameters, value_range)`: 1D smoothing component
- `GaussianSmoother2D(n_x_params, n_y_params, x_range, y_range)`: 2D smoothing
- `GaussianSmoother3D(n_x_params, n_y_params, n_z_params, x_range, y_range, z_range)`: 3D smoothing
- `value_error_for_location(locations)` → (values, errors): Evaluate smoother at coordinates

**4. Return Types:**
- `get_scales()`: flex.double array of scale factors
- `calculate_scales_and_derivatives()`: tuple of (scales, derivatives) flex arrays
- `value_error_for_location()`: tuple of (flex.double values, flex.double errors)

**5. Example Usage - Custom Scaling Model:**
```python
from dials.algorithms.scaling.model import ScalingModelBase
from dials.algorithms.scaling.model.components.scale_components import ScaleComponentBase
from dials.algorithms.scaling.active_parameter_managers import active_parameter_manager
from dials.array_family import flex
from scitbx import matrix

class SimpleMultiplicativeComponent(ScaleComponentBase):
    """Simple multiplicative scale factor component"""
    
    def __init__(self, active_parameter_manager, initial_scale=1.0):
        # Initialize with single parameter
        super().__init__(active_parameter_manager)
        self._initial_scale = initial_scale
        
        # Add parameter to manager
        self.active_parameter_manager.add_parameters(
            flex.double([initial_scale])  # Single scale parameter
        )
    
    @property
    def n_params(self):
        return 1
    
    @property  
    def parameters(self):
        return self.active_parameter_manager.get_parameters()[:self.n_params]
    
    def calculate_scales_and_derivatives(self, reflection_table, block_id=None):
        """
        Calculate scales and derivatives for all reflections
        
        Returns:
            tuple: (scales, derivatives) where derivatives[i,j] = ∂scale_i/∂param_j
        """
        n_refl = len(reflection_table)
        scale_param = self.parameters[0]
        
        # All reflections get the same scale factor
        scales = flex.double(n_refl, scale_param)
        
        # Derivatives: ∂scale_i/∂param_0 = 1 for all i
        derivatives = flex.double(flex.grid(n_refl, 1), 1.0)
        
        return scales, derivatives
    
    def update_reflection_data(self, reflection_table):
        """Update component with new reflection data if needed"""
        pass

class ResolutionDependentComponent(ScaleComponentBase):
    """Scale component that varies with resolution"""
    
    def __init__(self, active_parameter_manager, n_resolution_bins=10, d_min=1.0, d_max=50.0):
        super().__init__(active_parameter_manager)
        self.n_resolution_bins = n_resolution_bins
        self.d_min = d_min
        self.d_max = d_max
        
        # Initialize parameters (one per resolution bin)
        initial_params = flex.double([1.0] * n_resolution_bins)
        self.active_parameter_manager.add_parameters(initial_params)
        
        # Store bin edges
        import math
        log_d_min = math.log(d_min)
        log_d_max = math.log(d_max)
        self.log_d_edges = [log_d_min + i * (log_d_max - log_d_min) / n_resolution_bins 
                           for i in range(n_resolution_bins + 1)]
    
    @property
    def n_params(self):
        return self.n_resolution_bins
    
    @property
    def parameters(self):
        return self.active_parameter_manager.get_parameters()[:self.n_params]
    
    def calculate_scales_and_derivatives(self, reflection_table, block_id=None):
        """Calculate resolution-dependent scales"""
        import math
        
        # Get d-spacings
        if "d" not in reflection_table:
            raise ValueError("Reflection table must contain 'd' column for resolution-dependent scaling")
        
        d_spacings = reflection_table["d"]
        n_refl = len(d_spacings)
        
        scales = flex.double(n_refl)
        derivatives = flex.double(flex.grid(n_refl, self.n_params), 0.0)
        
        # Assign scales based on resolution bins
        for i, d_val in enumerate(d_spacings):
            if d_val <= 0:
                scales[i] = 1.0
                continue
                
            log_d = math.log(d_val)
            
            # Find appropriate bin
            bin_idx = 0
            for j in range(len(self.log_d_edges) - 1):
                if self.log_d_edges[j] <= log_d < self.log_d_edges[j + 1]:
                    bin_idx = j
                    break
            
            # Ensure bin_idx is valid
            bin_idx = max(0, min(bin_idx, self.n_params - 1))
            
            # Set scale and derivative
            scales[i] = self.parameters[bin_idx]
            derivatives[i, bin_idx] = 1.0
        
        return scales, derivatives

class CustomRelativeScalingModel(ScalingModelBase):
    """Custom scaling model for Module 3.S.3 relative scaling"""
    
    def __init__(self, configdict, is_scaled=False):
        super().__init__(configdict, is_scaled)
        self._components = {}
        
    def configure_components(self, reflection_table, experiment, params):
        """Configure scaling components based on data and parameters"""
        
        # Set up parameter manager
        self.active_parameter_manager = active_parameter_manager()
        
        # Add multiplicative component
        self._components["multiplicative"] = SimpleMultiplicativeComponent(
            self.active_parameter_manager, initial_scale=1.0
        )
        
        # Add resolution-dependent component if requested
        if hasattr(params, 'resolution_dependent') and params.resolution_dependent:
            self._components["resolution"] = ResolutionDependentComponent(
                self.active_parameter_manager, 
                n_resolution_bins=params.n_resolution_bins,
                d_min=params.d_min, 
                d_max=params.d_max
            )
        
        # Calculate d-spacings if needed
        if "resolution" in self._components and "d" not in reflection_table:
            # Add d-spacing calculation
            from cctbx import uctbx
            unit_cell = experiment.crystal.get_unit_cell()
            miller_indices = reflection_table["miller_index"]
            d_spacings = flex.double([unit_cell.d(hkl) for hkl in miller_indices])
            reflection_table["d"] = d_spacings
    
    @property
    def components(self):
        return list(self._components.values())
    
    def get_scales(self, reflection_table):
        """Calculate combined scale factors from all components"""
        if not self._components:
            return flex.double(len(reflection_table), 1.0)
        
        # Start with unit scales
        total_scales = flex.double(len(reflection_table), 1.0)
        
        # Multiply scales from all components
        for component in self.components:
            component_scales, _ = component.calculate_scales_and_derivatives(reflection_table)
            total_scales *= component_scales
        
        return total_scales
    
    def get_inverse_scales(self, reflection_table):
        """Calculate inverse scale factors"""
        scales = self.get_scales(reflection_table)
        return 1.0 / scales
```

**6. Advanced Framework Usage - Parameter Management and Refinement:**
```python
from dials.algorithms.scaling.active_parameter_managers import multi_active_parameter_manager
from dials.algorithms.scaling.scaling_refiner import scaling_refinery
from dials.algorithms.scaling.target_function import ScalingTargetFunction
from scitbx.lstbx import normal_eqns

class CustomScalingTargetFunction:
    """Custom target function for scaling refinement"""
    
    def __init__(self, scaling_models, reflection_tables):
        self.scaling_models = scaling_models
        self.reflection_tables = reflection_tables
    
    def compute_residuals_and_gradients(self):
        """
        Compute residuals and gradients for least squares refinement
        
        Returns:
            tuple: (residuals, jacobian) for least squares optimization
        """
        from dials.array_family import flex
        import math
        
        all_residuals = flex.double()
        all_gradients = []
        
        for model, refl_table in zip(self.scaling_models, self.reflection_tables):
            if len(refl_table) == 0:
                continue
                
            # Get observed intensities and calculate expected from model
            obs_intensities = refl_table["intensity.sum.value"]
            obs_variances = refl_table["intensity.sum.variance"]
            
            # Get scales from model
            scales = model.get_scales(refl_table)
            
            # Example residual: (I_obs - scale * I_ref) / sigma
            # For relative scaling, we might compare intensities between datasets
            
            # Simple example: residual based on deviation from mean intensity
            mean_intensity = flex.mean(obs_intensities)
            expected_intensities = scales * mean_intensity
            
            # Calculate residuals
            residuals = flex.double()
            gradients_matrix = []
            
            for i, (obs, exp, var) in enumerate(zip(obs_intensities, expected_intensities, obs_variances)):
                if var > 0:
                    sigma = math.sqrt(var)
                    residual = (obs - exp) / sigma
                    residuals.append(residual)
                    
                    # Calculate gradients with respect to model parameters
                    # ∂residual/∂param = -(∂exp/∂param) / sigma = -(∂(scale*I_ref)/∂param) / sigma
                    
                    # Get parameter gradients from components
                    component_gradients = flex.double()
                    for component in model.components:
                        _, derivatives = component.calculate_scales_and_derivatives(refl_table)
                        # ∂exp/∂param = I_ref * ∂scale/∂param
                        param_gradient = -mean_intensity * derivatives[i, :] / sigma
                        component_gradients.extend(param_gradient)
                    
                    gradients_matrix.append(component_gradients)
            
            all_residuals.extend(residuals)
            all_gradients.extend(gradients_matrix)
        
        # Convert gradients to proper format for least squares
        if all_gradients:
            n_residuals = len(all_residuals)
            n_params = len(all_gradients[0]) if all_gradients else 0
            jacobian = flex.double(flex.grid(n_residuals, n_params))
            
            for i, grad_row in enumerate(all_gradients):
                for j, grad_val in enumerate(grad_row):
                    jacobian[i, j] = grad_val
        else:
            jacobian = flex.double(flex.grid(0, 0))
        
        return all_residuals, jacobian

def run_scaling_refinement(scaling_models, reflection_tables, max_iterations=10):
    """
    Run scaling parameter refinement using DIALS scaling framework
    """
    from scitbx.lstbx import normal_eqns_solving
    
    # Create target function
    target_function = CustomScalingTargetFunction(scaling_models, reflection_tables)
    
    # Set up parameter manager for all models
    param_manager = multi_active_parameter_manager()
    for model in scaling_models:
        if hasattr(model, 'active_parameter_manager'):
            param_manager.add_parameter_manager(model.active_parameter_manager)
    
    # Refinement loop
    for iteration in range(max_iterations):
        print(f"Refinement iteration {iteration + 1}")
        
        # Calculate residuals and gradients
        residuals, jacobian = target_function.compute_residuals_and_gradients()
        
        if len(residuals) == 0:
            print("No data for refinement")
            break
        
        # Set up normal equations
        normal_eqns = normal_eqns_solving.levenberg_marquardt_iterations(
            residuals=residuals,
            jacobian=jacobian,
            step_threshold=1e-6,
            gradient_threshold=1e-6
        )
        
        # Solve for parameter shifts
        try:
            normal_eqns.build_and_solve()
            parameter_shifts = normal_eqns.solution()
            
            # Apply parameter shifts
            current_params = param_manager.get_parameters()
            new_params = current_params + parameter_shifts
            param_manager.set_parameters(new_params)
            
            # Check convergence
            shift_magnitude = flex.max(flex.abs(parameter_shifts))
            residual_rms = math.sqrt(flex.mean(residuals * residuals))
            
            print(f"  RMS residual: {residual_rms:.6f}")
            print(f"  Max parameter shift: {shift_magnitude:.6f}")
            
            if shift_magnitude < 1e-6:
                print("Refinement converged")
                break
                
        except Exception as e:
            print(f"Refinement failed at iteration {iteration + 1}: {e}")
            break
    
    return scaling_models

# Usage example
def create_and_refine_scaling_model(experiments, reflection_tables):
    """Complete example of creating and refining a custom scaling model"""
    
    # Create scaling models for each dataset
    scaling_models = []
    for i, (expt, refl_table) in enumerate(zip(experiments, reflection_tables)):
        
        # Configuration for this model
        configdict = {
            'model_type': 'custom_relative',
            'resolution_dependent': True,
            'n_resolution_bins': 10,
            'd_min': 1.5,
            'd_max': 50.0
        }
        
        # Create model
        model = CustomRelativeScalingModel(configdict, is_scaled=False)
        
        # Configure components based on data
        from types import SimpleNamespace
        params = SimpleNamespace()
        params.resolution_dependent = True
        params.n_resolution_bins = 10
        params.d_min = 1.5
        params.d_max = 50.0
        
        model.configure_components(refl_table, expt, params)
        scaling_models.append(model)
        
        print(f"Model {i}: {len(model.components)} components, "
              f"{sum(c.n_params for c in model.components)} parameters")
    
    # Run refinement
    refined_models = run_scaling_refinement(scaling_models, reflection_tables)
    
    # Apply scaling to data
    for model, refl_table in zip(refined_models, reflection_tables):
        scales = model.get_scales(refl_table)
        
        # Apply scales to intensities
        refl_table["intensity.scaled.value"] = refl_table["intensity.sum.value"] * scales
        refl_table["intensity.scaled.variance"] = refl_table["intensity.sum.variance"] * (scales * scales)
        
        print(f"Applied scaling: scale range {flex.min(scales):.3f} - {flex.max(scales):.3f}")
    
    return refined_models, reflection_tables
```

**7. Gaussian Smoothing Components Usage:**
```python
from dials.algorithms.scaling.model.components.smooth_scale_components import GaussianSmoother1D

class SmoothScaleComponent(ScaleComponentBase):
    """Component using Gaussian smoothing for parameter regularization"""
    
    def __init__(self, active_parameter_manager, coordinate_values, n_control_points=20):
        super().__init__(active_parameter_manager)
        
        # Set up coordinate range
        self.coord_min = flex.min(coordinate_values) 
        self.coord_max = flex.max(coordinate_values)
        self.coord_range = (self.coord_min, self.coord_max)
        
        # Create Gaussian smoother
        self.smoother = GaussianSmoother1D(
            n_parameters=n_control_points,
            value_range=self.coord_range
        )
        
        # Initialize control point values
        initial_values = flex.double([1.0] * n_control_points)
        self.active_parameter_manager.add_parameters(initial_values)
        
        # Store coordinate values for evaluation
        self.coordinate_values = coordinate_values
    
    @property
    def n_params(self):
        return self.smoother.n_parameters
    
    def calculate_scales_and_derivatives(self, reflection_table, block_id=None):
        """Calculate smoothed scales using Gaussian interpolation"""
        
        # Get current parameter values
        control_values = self.parameters
        
        # Set control point values in smoother
        self.smoother.set_parameters(control_values)
        
        # Evaluate smoother at coordinate positions
        scales, errors = self.smoother.value_error_for_location(self.coordinate_values)
        
        # Calculate derivatives: ∂scale_i/∂control_j
        derivatives = flex.double(flex.grid(len(scales), self.n_params), 0.0)
        
        # Use analytical derivatives from smoother weight matrices
        # DIALS GaussianSmoother classes provide analytical derivatives through weight matrices
        # This is more accurate and efficient than finite differences
        for i, coord in enumerate(self.coordinate_values):
            # Get analytical weights and values at coordinate location
            weight_result = self.smoother.value_weight(coord)
            weights = weight_result.get_weight()  # Weight matrix for this coordinate
            
            # Weights provide direct derivatives: ∂scale_i/∂control_j = weight[i,j]
            for j in range(self.n_params):
                derivatives[i, j] = weights[j]  # Direct analytical derivative
        
        return scales, derivatives
```

**8. Notes/Caveats:**
- **Framework Complexity:** The scaling framework is sophisticated; start with simple components before building complex models
- **Parameter Management:** Always use `active_parameter_manager` for proper parameter handling and refinement
- **Data Requirements:** Ensure reflection tables contain required columns (intensity, variance, Miller indices)
- **Memory Usage:** Large datasets with many parameters can consume significant memory during refinement
- **Convergence:** Monitor refinement convergence carefully; poor initial guesses can lead to non-convergence
- **Component Ordering:** The order of components in `_components` affects the final scaling calculation
- **Analytical Derivatives:** GaussianSmoother classes provide analytical derivatives through weight matrices from `value_weight()` methods, eliminating the need for finite difference calculations

**9. Integration with Diffuse Scattering Pipeline:**
This framework enables implementation of Module 3.S.3 relative scaling by providing:
- Custom scaling models tailored to diffuse scattering requirements  
- Flexible parameter management for multi-dataset scaling
- Robust refinement algorithms for optimizing scale factors
- Integration with DIALS reflection table infrastructure

**10. See Also:**
- **File I/O Operations**: [dials_file_io.md](dials_file_io.md)
- **Detector Models**: [dxtbx_models.md](dxtbx_models.md)
- **Crystallographic Calculations**: [crystallographic_calculations.md](crystallographic_calculations.md)
- **Array Operations**: [flex_arrays.md](flex_arrays.md)
- DIALS scaling documentation for additional built-in scaling models