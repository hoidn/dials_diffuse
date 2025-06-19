# Implementation Rules and Developer Guidelines

**1. Purpose**

This document outlines the standard conventions, patterns, and rules for implementing code (primarily Python, but principles can be adapted) within this project. Adhering to these guidelines ensures consistency, maintainability, testability, and portability across the codebase, especially when translating IDL specifications into concrete implementations.

**2. Core Principles**

*   **Consistency:** Code should look and feel consistent regardless of who wrote it. Follow established patterns and conventions.
*   **Clarity & Readability:** Prioritize clear, understandable code over overly clever or complex solutions. Code is read more often than it is written.
*   **Simplicity (KISS & YAGNI):** Implement the simplest solution that meets the current requirements. Avoid unnecessary complexity or features. (See `03_PROJECT_RULES.md` for details).
*   **Testability:** Design code with testing in mind. Use dependency injection and avoid tight coupling.
*   **Portability (Conceptual):** While implementing in a specific language (e.g., Python), aim for logic and structures that are conceptually portable to other common languages if future needs arise. Minimize reliance on highly language-specific idioms where simpler, universal constructs exist for core logic.
*   **Parse, Don't Validate:** Structure data transformations such that input is parsed into well-defined, type-safe structures upfront, minimizing the need for scattered validation checks later in the code. (See Section 5).

**3. Project Structure and Imports**

*   **Directory Structure:** Strictly follow the established directory structure outlined in `03_PROJECT_RULES.md`. Place new modules and files in their logical component directories.
*   **Import Conventions (Python Example):**
    *   Use **absolute imports** starting from the `src` directory (or your project's source root) for all internal project modules.
        ```python
        # Good (assuming 'src' is the root of your package)
        from src.components.my_component import MyClass
        from src.utils.errors import CustomError

        # Bad (Avoid relative imports that traverse too many levels or are ambiguous)
        # from ..utils.errors import CustomError
        ```
    *   Group imports in the standard order: standard library, third-party, project-specific.
    *   Place imports at the top of the module. Avoid imports inside functions/methods unless absolutely necessary for specific reasons (e.g., avoiding circular dependencies, optional heavy imports) and document the reason clearly.

**4. Safe Refactoring Practices**

### 4.1 The 5-10 Line Rule (MANDATORY)

**Core Principle:** Never attempt to refactor more than 5-10 lines at once.

**Proven Process:**
1. **Incremental Changes:** When refactoring large methods or classes (>100 lines), work in small increments of 5-10 lines at a time. Never attempt to replace 300+ lines in a single operation.
2. **Syntax Verification:** After each change, verify file syntax immediately: `python -m py_compile file.py`. Do not proceed to the next change until syntax is confirmed correct.
3. **Frequent Commits:** Commit working code after each successful incremental change. This provides recovery points if later changes introduce issues.

### 4.2 Extract First, Then Modify Pattern

**Rule:** When moving code to new files, create and test the new file completely before modifying the original. Never perform both operations simultaneously.

**Process:**
1. Create new file with extracted code
2. Test new file completely (imports, syntax, basic functionality)
3. Only then modify the original file to use the new code
4. Verify imports work before implementing method calls

### 4.3 Refactoring Safety Checks

**Before Each Change:**
*   Verify current syntax: `python -m py_compile file.py`
*   Ensure file is under version control with clean state
*   Test imports independently before implementing usage

**After Each Change:**
*   Immediate syntax verification
*   Commit if successful
*   If syntax fails, immediately revert and restart with smaller increment

### 4.4 Critical Warning Signs

**File Corruption Indicators:**
*   File "appears" to work but contains syntax errors in unused code paths
*   Duplicate method definitions with different signatures
*   Orphaned code blocks with incorrect indentation
*   Import statements without corresponding class usage

**These signs indicate immediate need to revert to last known good state.**

### 4.5 Recovery Strategy

**When Refactoring Corruption Occurs:**
1. **Immediate Revert:** Use version control to restore last known good state - do not attempt to fix corruption
2. **Restart Incrementally:** Re-implement refactoring with smaller steps (2-3 lines max)
3. **Use Proper Tools:** IDE/editor with syntax highlighting and error detection
4. **Avoid Complex String Replacements:** Manual verification is safer than complex multi-line string matching operations

**This process is based on real failure experience with 350+ line corruption incident. See `docs/LESSONS_LEARNED.md` for detailed case study.**

**5. Coding Style and Formatting**

*   **Language Standards:** Strictly adhere to the idiomatic style guidelines for your project's primary language (e.g., PEP 8 for Python). Use linters and formatters.
*   **Type Hinting (Python Example):**
    *   **Mandatory (if language supports):** All function and method signatures (parameters and return types) **must** include type hints.
    *   Use appropriate types for optional parameters/nullable types.
    *   Use specific types rather than generic "any" types whenever possible.
    *   For complex dictionary structures passed as parameters (especially from IDL "Expected Data Format"), define a `TypedDict`, data class, or Pydantic model for clarity and validation.
*   **Docstrings/Comments:**
    *   **Mandatory:** All modules, classes, functions, and methods must have clear documentation (e.g., docstrings in Python).
    *   Use a consistent style (e.g., Google Style for Python docstrings).
    *   Clearly document parameters, return values, and any exceptions/errors raised.
    *   Explain the *purpose* and *behavior*, not just *what* the code does.
*   **Vectorization with NumPy:**
    *   **Principle:** For operations involving numerical data, especially on arrays or sequences of numbers (e.g., pixel data, lists of coordinates, q-vectors), prioritize vectorized operations using NumPy over explicit Python loops where feasible and sensible.
    *   **Benefits:** NumPy vectorization typically leads to more concise, readable, and significantly more performant code for numerical tasks.
    *   **Practice:** Identify opportunities to replace loops that perform element-wise arithmetic or apply mathematical functions to sequences with equivalent NumPy array operations.
        ```python
        # Less Preferred (Python loop)
        # result = []
        # for x, y in zip(list_a, list_b):
        #     result.append(x * y + 5)

        # Preferred (NumPy vectorization)
        # import numpy as np
        # array_a = np.array(list_a)
        # array_b = np.array(list_b)
        # result_array = array_a * array_b + 5
        ```
    *   **Consideration:** While vectorization is generally preferred for performance, ensure that its use does not unduly obfuscate the logic for very complex, non-standard operations. Balance performance with clarity. DIALS `flex` arrays also offer vectorized operations and should be used when interacting directly with DIALS data structures.

**5.1 Performance Optimization Strategies**

**Mandatory Vectorization Policy:**
*   **Vectorization-First Development:** All array operations involving >100 elements MUST use vectorized approaches unless explicitly justified in code comments.
*   **Performance Review Requirement:** Any Python loop processing NumPy arrays requires justification and performance comparison with vectorized alternatives.
*   **Matrix Operation Standards:** Always prefer matrix multiplication over element-wise loops for linear algebra operations.
*   **External Library Integration:** Convert external library matrices (e.g., scitbx) to NumPy arrays for vectorization opportunities.

**Vectorization Success Patterns:**
*   **Vectorize by Default:** When in doubt, prefer vectorized implementations. Only use iterative approaches during initial prototyping or for very complex non-standard operations.
*   **Batch Processing:** Process coordinates, q-vectors, and corrections as arrays rather than individual elements. Group operations that can be parallelized.
*   **Memory Efficiency:** Minimize Python loop overhead through vectorized NumPy operations. Use array-based data structures over lists of individual objects.
*   **Algorithmic Validation:** Test vectorized implementations against known reference results or simplified test cases to ensure correctness.
*   **Split-Apply-Combine Strategy:** Use pandas DataFrame groupby operations for efficient voxel-wise processing instead of nested loops.
*   **Matrix Operations Priority:** Always prefer matrix multiplication (`@` operator) over element-wise loops for linear algebra operations.
*   **Proven Performance Gains:** Real-world experience shows 10x+ speedups are achievable with proper vectorization (see `docs/LESSONS_LEARNED.md` for case studies).

**Performance Measurement Framework:**
*   **Mandatory Performance Tests:** All computational methods processing arrays must include performance benchmarks in their test suite.
*   **Structured Testing:** Document speedups with before/after timing measurements using consistent test conditions.
*   **Realistic Data Sizes:** Test with actual detector geometries and data volumes representative of production use.
*   **Performance Characterization:** Document specific improvements with quantified metrics (e.g., "2.4x speedup: 4.0s → 1.7s").
*   **Scalability Validation:** Ensure optimizations work across different data scales and don't degrade with larger inputs.

**Example Vectorization Pattern:**
```python
# ANTI-PATTERN: Python loop processing arrays (requires justification)
result = []
for x, y in zip(list_a, list_b):
    result.append(complex_calculation(x, y))

# CORRECT: Vectorized operation (10x+ faster for large arrays)
array_a = np.array(list_a)
array_b = np.array(list_b)
result_array = vectorized_complex_calculation(array_a, array_b)

# CRITICAL: Matrix operations with external libraries
# ANTI-PATTERN: Loop-based transformation
hkl_fractional = []
for q_vec in combined_q_vectors:
    q_matrix = matrix.col(q_vec)
    hkl_frac = A_inv * q_matrix
    hkl_fractional.append(hkl_frac.elems)
hkl_array = np.array(hkl_fractional)

# CORRECT: Vectorized matrix multiplication
A_inv_np = np.array(A_inv.elems).reshape(3, 3)
hkl_array = (A_inv_np @ combined_q_vectors.T).T

# MANDATORY: Prove equivalence
assert np.allclose(result, result_array, rtol=1e-15)
```

**Optimization Implementation Guidelines:**
*   **Profile First:** Identify actual bottlenecks through profiling before optimizing
*   **Vectorize First:** Implement vectorized solutions directly rather than iterative approaches
*   **Validate Results:** Ensure vectorized code produces correct results using reference data or simplified test cases
*   **Document Benefits:** Record specific performance improvements achieved
*   **Test Edge Cases:** Verify optimizations work correctly with boundary conditions
*   **Code Review Checklist:** All array processing code must be reviewed for vectorization opportunities

**5.2 Numerical Stability Requirements (MANDATORY)**

### 5.2.1 Division Safety Rules

**Core Principle:** Never divide by computed values without stabilization.

**Mandatory Patterns:**
```python
# REQUIRED: Epsilon stabilization for variance calculations
safe_variance = variance + 1e-10
weights = 1.0 / safe_variance

# REQUIRED: Safe division with threshold replacement
safe_scales = np.where(np.abs(scales) > 1e-9, scales, 1e-9)
scaled_values = observations / safe_scales

# REQUIRED: JSON serialization type conversion
key = int(still_id)  # Cast NumPy int64 to Python int
result_dict[key] = values
```

### 5.2.2 Numerical Validation Requirements

**Mandatory Checks:**
*   **Division Operations:** All division operations must include epsilon stabilization (1e-10) or use safe division patterns
*   **Variance Calculations:** All variance and weight calculations must be protected against zero denominators
*   **Scale Factor Stability:** All multiplicative scale factors must be checked for numerical range and stability
*   **JSON Compatibility:** All NumPy integer types used as dictionary keys must be cast to Python int for serialization
*   **Range Validation:** Critical outputs must include numerical range checking (`np.isnan`, `np.isinf`, extreme values)

### 5.2.3 Code Review Checklist for Numerical Code

**Required Reviews:**
1. ✅ All division operations include stabilization or safe division patterns
2. ✅ Variance and weight calculations protected against zero denominators  
3. ✅ Scale factors checked for numerical range and stability
4. ✅ JSON serialization compatibility verified for all dictionary keys
5. ✅ Unit tests include edge cases (zero, near-zero, extreme values)
6. ✅ Validation includes numerical range checking for critical outputs

**Warning Signs to Flag:**
*   Division operations on computed values without stabilization
*   Variance calculations that could produce exact zeros
*   NumPy integer types used as dictionary keys for JSON output
*   Missing numerical validation in iterative refinement algorithms
*   Test suites that only use "nice" input values

### 5.2.4 Production Safeguards

**Validation Helper Pattern:**
```python
def validate_numerical_health(values, context="computation"):
    """Validate that computed values are numerically stable."""
    if np.any(np.isnan(values)):
        raise ValueError(f"NaN values detected in {context}")
    if np.any(np.isinf(values)):
        raise ValueError(f"Infinite values detected in {context}")
    if np.any(np.abs(values) > 1e10):
        logger.warning(f"Very large values detected in {context}: max={np.max(np.abs(values))}")
```

**Integration Requirements:**
*   Run E2E pipelines with diverse input data to catch numerical edge cases
*   Include validation scripts that check for numerical health of outputs
*   Monitor convergence behavior in iterative algorithms
*   Test serialization/deserialization round-trips for all data structures

**This section is based on real production failures in scaling refinement. See `docs/LESSONS_LEARNED.md` for detailed case studies.**

**5.3 Scientific Accuracy Implementation Guidelines**

**NIST Data Integration Standards:**
*   **Reference Data Sources:** Use tabulated NIST X-ray mass attenuation coefficients over rough approximations for critical calculations.
*   **Proper Atmospheric Composition:** Standard dry air by mass (N: 78.084%, O: 20.946%, Ar: 0.934%, C: 0.036%) with correct molar masses.
*   **Thermodynamic Accuracy:** Implement ideal gas law with configurable temperature and pressure parameters for environmental calculations.
*   **Energy Range Coverage:** Support wide energy ranges (e.g., 1-100 keV) with appropriate interpolation methods for accuracy.

**Validation Requirements:**
*   **Reference Value Testing:** Validate calculated values against NIST reference data within 1% tolerance where possible.
*   **Cross-Validation:** Compare against known theoretical values from authoritative scientific literature.
*   **Unit Conversion Precision:** Ensure proper dimensional consistency throughout calculations (e.g., mm to m conversions).
*   **Configurable Parameters:** Support variable environmental conditions (temperature, pressure, humidity) rather than hard-coded values.

**Scientific Enhancement Process:**
1. **Identify Approximations:** Locate rough approximations or "magic numbers" in existing code that could benefit from scientific accuracy.
2. **Research Authoritative Sources:** Find official data from NIST, IUCR, or other recognized scientific institutions.
3. **Implement Accurate Calculations:** Replace approximations with scientifically validated formulas and tabulated data.
4. **Create Validation Tests:** Develop comprehensive tests against reference data with appropriate tolerances.
5. **Document Scientific Basis:** Include references to data sources, formulas used, and any assumptions made.

**Example: Air Attenuation Correction Enhancement:**
```python
# Before: Rough approximation
air_correction = wavelength**3 * 0.001  # Magic number

# After: NIST-based calculation
def calculate_air_attenuation(wavelength_angstrom, path_length_mm, 
                            temperature_k=293.15, pressure_atm=1.0):
    """Calculate air attenuation using NIST mass attenuation data"""
    energy_kev = 12.398 / wavelength_angstrom
    mu_air = get_nist_mass_attenuation_coefficient(energy_kev)
    air_density = calculate_air_density(temperature_k, pressure_atm)
    path_length_m = path_length_mm / 1000.0
    return np.exp(-mu_air * air_density * path_length_m)
```

*   **Naming:** Follow language-standard naming conventions (e.g., snake_case for Python variables/functions, CamelCase for Python classes). Use descriptive names.

**5.3 Data Structure Conventions for Inter-Component Communication (NEW SECTION)**

To maintain performance across the entire pipeline, it is not enough to vectorize calculations *within* a component. The data structures used to pass information *between* components must also be designed for efficiency.

**The "Struct-of-Arrays" Principle:**

*   **Rule:** When passing a large collection of records (e.g., >1000 items) between components, the data structure **MUST** use a "Struct-of-Arrays" (SoA) format. It **MUST NOT** use an "Array-of-Structs" (AoS) format.

    *   **ANTI-PATTERN (Array-of-Structs - SLOW, HIGH MEMORY):** A Python list where each element is an object or a dictionary.
        ```python
        # DO NOT DO THIS FOR LARGE DATASETS
        inefficient_data = [
            {"intensity": 100.0, "sigma": 10.0, "still_id": 1},
            {"intensity": 120.0, "sigma": 12.0, "still_id": 2},
            # ... millions more dicts ...
        ]
        ```

    *   **CORRECT PATTERN (Struct-of-Arrays - FAST, LOW MEMORY):** A single dictionary where each key holds a NumPy array containing all values for that field.
        ```python
        # DO THIS
        efficient_data = {
            "intensities": np.array([100.0, 120.0, ...]),
            "sigmas": np.array([10.0, 12.0, ...]),
            "still_ids": np.array([1, 2, ...]),
        }
        ```

*   **Rationale:** The "Array-of-Structs" pattern introduces massive overhead from creating millions of individual Python objects. It breaks vectorization, forcing the consuming component to loop through the list to re-assemble arrays. The "Struct-of-Arrays" pattern maintains data in contiguous, cache-friendly NumPy arrays, minimizing memory usage and allowing the receiving component to perform vectorized operations immediately.

*   **Case Study (VoxelAccumulator Failure):** The performance hang in `get_all_binned_data_for_scaling` was caused by a direct violation of this principle. The method was converting its efficient internal HDF5 arrays into a `list[dict]` structure, which caused the system to hang when trying to allocate memory for millions of Python objects.

**IDL Specification Guideline:**

*   When defining an `Expected Data Format` in an `*_IDL.md` file for a method that returns a large collection, the format **MUST** be specified in the "Struct-of-Arrays" format.

    *   **ANTI-PATTERN in IDL:**
        `// Returns: list<ObservationTuple>`

    *   **CORRECT PATTERN in IDL:**
        ```
        // Expected Data Format:
        // Returns: map<string, numpy.ndarray>
        // {
        //     "intensities": numpy.ndarray,
        //     "sigmas": numpy.ndarray,
        //     "still_ids": numpy.ndarray,
        //     ...
        // }
        ```

**Code Review Mandate:**

*   Any Pull Request that includes a method or function that returns a collection of data must be reviewed for adherence to the "Struct-of-Arrays" principle. A method returning a `list` of objects or `dict`s where the list could become large (>1000 items) must be rejected unless a strong justification is provided.

**5. Data Handling: Parse, Don't Validate (Leveraging Models like Pydantic)**

*   **Principle:** Instead of passing raw dictionaries or loosely typed data and validating fields throughout the code, parse external/untrusted data (e.g., API responses, configuration files, parameters described via IDL "Expected Data Format") into **well-defined data models** (e.g., Pydantic models in Python, or data classes/structs) at the boundaries of your system or component.
*   **Benefits:**
    *   **Upfront Validation:** Data validity is checked once during parsing/instantiation.
    *   **Type Safety:** Subsequent code operates on validated, type-hinted objects.
    *   **Reduced Boilerplate:** Eliminates repetitive validation checks.
    *   **Clear Data Contracts:** Models serve as clear definitions of expected data structures.
*   **Implementation (Python/Pydantic Example):**
    *   Define Pydantic `BaseModel` subclasses (or equivalent in your language) for structured data.
    *   Use these models in function signatures where appropriate.
    *   Parse incoming data using model validation methods (e.g., `YourModel.model_validate(data)`).
    *   Handle validation errors (e.g., `pydantic.ValidationError`) during parsing to manage invalid input gracefully.

    ```python
    from pydantic import BaseModel, ValidationError
    from typing import Optional

    # IDL: Expected Data Format: { "name": "string", "retries": "int" }
    class TaskParams(BaseModel):
        name: str
        retries: Optional[int] = 3 # Example with default

    def process_task(raw_params: dict):
        try:
            params = TaskParams.model_validate(raw_params)
            print(f"Processing task: {params.name} with {params.retries} retries")
            # ... logic using validated params ...
        except ValidationError as e:
            print(f"Invalid task parameters: <!-- Warning: File 'e' not found -->")
            # Handle error
    ```
*   **Understanding External Library Data Structures:** When your component consumes objects directly instantiated or returned by an external library, consult that library's documentation to understand their precise structure, attributes, and access methods. Do not assume a generic structure. This understanding should inform both implementation and test case design.
    *   **Action:** During preparation, if your component relies on specific object types from an external library, review the documentation for those object types.

**5.x Dependency Injection & Initialization**

*   **Constructor/Setter Injection:** Components MUST receive their runtime dependencies (other components, resources specified in IDL `@depends_on`) via their constructor or dedicated setter methods. Avoid complex internal logic within components to locate or instantiate their own major dependencies.
*   **Initialization Order:** In orchestrating components (like a main `Application` class), instantiate dependencies in the correct order *before* injecting them into dependent components that require them during their own initialization.
*   **Circular Dependencies:** Be vigilant for circular import dependencies. Minimize top-level imports in modules involved in complex interactions; prefer imports inside methods/functions where feasible. Use string type hints or forward references if supported by the language to break cycles needed only for type checking. If cycles are identified, prioritize refactoring.
*   **Test Instantiation:** Include tests verifying that components can be instantiated correctly with their required (real or mocked) dependencies.

**6. External Service Interaction (e.g., LLMs, APIs, Databases)**

*   **Standard:** Use dedicated "Manager" or "Bridge" classes to encapsulate interactions with significant external services or APIs.
*   **Implementation Pattern:**
    *   A Manager/Bridge class is responsible for handling connection details, API call formatting, authentication, and basic request/response processing for a specific external service.
    *   Other components in the system use this Manager/Bridge class rather than interacting directly with the external service's raw API or client library.
    *   The Manager/Bridge class itself might use a specific client library for the external service (e.g., `requests`, `boto3`, `pydantic-ai`).
*   **Structured Output/Input:** If the external service supports or requires structured data (e.g., JSON schemas for input/output), leverage this. If using Pydantic, you can define models for these schemas and use them in your Manager/Bridge.
    *   **Schema-to-Model Resolution (If applicable):** If task/service definitions include references to data schemas (e.g., by name or path), implement a helper function (e.g., `resolve_model_class`) to dynamically load the corresponding Pydantic model or data class.
*   **Reference:** Familiarize yourself with the chosen client libraries. Document key usage patterns or link to official documentation in `LIBRARY_INTEGRATION/`.
*   **Verify Library Usage:** **Crucially, when integrating *any* significant third-party library, carefully verify API usage** (function signatures, required arguments, expected data formats, object constructor parameters) against the library's official documentation for the specific version being used.
*   **Test Wrapper Interactions:** Manager/Bridge classes should have targeted integration tests verifying their interaction with the (mocked) external service endpoint.

**7. Testing Conventions**

*   **Framework:** Use a standard testing framework (e.g., `pytest` for Python).
*   **Prioritize Integration and Functional Tests:**
    *   **Core Principle:** Integration tests with real components should be the foundation of your testing strategy. Test components together as they would operate in production to verify their interactions fulfill IDL contracts.
    *   **Real-World Scenarios:** Design tests around realistic workflows that exercise multiple components working together.
    *   **No Mock Chains:** Never use chains of mocks where one mock returns another mock. This creates tests that pass but don't validate actual behavior.

*   **Mandatory Performance Testing:**
    *   **Computational Method Coverage:** All methods processing arrays or performing matrix operations MUST include performance tests.
    *   **Vectorization Verification:** Include tests comparing vectorized implementations against loop-based reference implementations for correctness.
    *   **Performance Benchmarks:** Record baseline performance metrics that will catch significant regressions.
    *   **Implementation Comparison:** When multiple implementation approaches exist, include tests comparing their performance.
    *   **Example Performance Test Pattern:**
    ```python
    def test_hkl_transformation_performance(self):
        """Test that vectorized transformation is significantly faster than loop-based."""
        import time
        # Create large test dataset
        n_vectors = 100000
        test_q_vectors = np.random.uniform(-2.0, 2.0, (n_vectors, 3))
        
        # Time loop-based approach (on subset)
        start = time.perf_counter()
        loop_result = loop_based_transformation(test_q_vectors[:1000])
        loop_time = time.perf_counter() - start
        
        # Time vectorized approach (full dataset)
        start = time.perf_counter()
        vectorized_result = vectorized_transformation(test_q_vectors)
        vectorized_time = time.perf_counter() - start
        
        # Assert significant speedup
        assert (loop_time * 100) / vectorized_time > 5  # At least 5x faster
        
        # Verify correctness
        subset_vectorized = vectorized_transformation(test_q_vectors[:1000])
        assert np.allclose(loop_result, subset_vectorized, rtol=1e-10)
    ```

*   **Avoiding Mocks - Preferred Alternatives:**
    *   **Use Real Components:** Whenever possible, instantiate and use actual component implementations in tests rather than mocks.
    *   **Test Databases:** Use ephemeral test databases (e.g., SQLite in-memory, containerized PostgreSQL) rather than mocking database interactions.
    *   **File System:** Use temporary directories and files rather than mocking file system operations.
    *   **Test Fixtures:** Create comprehensive fixtures that provide real test data and properly configured components.
    *   **Test Doubles:** When necessary, prefer simple stubs or fakes that implement the same interface as the real component but with simplified behavior, rather than mocks with complex expectations.

*   **Limited Mocking - Only When Necessary:**
    *   **External API Boundaries:** Mock third-party APIs with usage limits, authentication requirements, or that require complex infrastructure that can't be containerized.
    *   **Non-Deterministic Components:** Mock components whose behavior cannot be controlled deterministically in a test environment (e.g., random number generators, time-dependent operations).
    *   **Expensive Resources:** Mock resources that are prohibitively expensive to create for each test run and cannot be reasonably containerized.

*   **When Mocking Is Unavoidable:**
    *   **Patch Where It's Looked Up:** When using `patch` (e.g., `unittest.mock.patch`), the `target` string must be the path to the object *where it is looked up/imported*, not necessarily where it was defined.
    *   **Prefer Dependency Injection:** For classes using Dependency Injection, pass test doubles into the constructor during test setup rather than patching.
    *   **Ensure Type Fidelity:** When mocking external libraries, ensure mock return values match the **expected type** returned by the real library. Use real library types for mock data if possible.
    *   **Verify Critical Dependencies:** Ensure critical runtime dependencies are installed and importable in the test environment.
    *   **Test Boundaries Separately:** Test argument preparation logic for external API calls separately from the external call itself.
*   **Test Doubles:** Use appropriate test doubles (Stubs, Mocks, Fakes).
*   **Arrange-Act-Assert:** Structure tests clearly.
*   **Fixtures:** Use testing framework fixtures for setup.
*   **Markers:** Use markers to categorize tests (e.g., `@pytest.mark.integration`).
*   **Testing Error Conditions:**
    *   Verify overall failure status (e.g., `result_status == "FAILED"`).
    *   Prefer asserting error type/reason codes over exact message strings.
    *   Check key details in structured error objects.
    *   Use message substring checks sparingly.
    *   Test exception raising using appropriate framework mechanisms (e.g., `pytest.raises`).
    *   When asserting complex return structures (e.g., dictionaries from Pydantic models), be mindful of serialization effects and assert against the actual returned structure.
*   **Unit Test Complex Logic:** Complex internal algorithms or utility functions should have dedicated unit tests.
*   **Debugging Mock Failures & Test Failures:** Systematically inspect mock attributes, call logs, and actual vs. expected values when tests fail.
*   **Test Setup for Error Conditions:** Ensure tests for error handling satisfy preconditions up to the point where the error is expected.
*   **Testing Configurable Behavior and Constants:** Write assertions that test behavioral outcomes rather than being rigidly tied to exact constant values. Review tests when constants change.

**7.1 C++ Backend Object Compatibility Testing (MANDATORY)**

### 7.1.1 DIALS C++ Integration Requirements

When testing components that interact with DIALS or other libraries with C++ backends, standard mocking approaches often fail due to type compatibility issues.

**Real Class Mocking (REQUIRED):**
Create actual Python classes that mimic C++ object interfaces instead of using MagicMock for constructors that will be passed to C++ code.

**Type Compatibility Checks:**
*   Ensure mock objects work with `isinstance()` checks required by C++ backends
*   Implement proper magic methods as required by actual usage patterns
*   Test that mocks survive C++ type conversion requirements during library calls

**Example C++ Compatible Mock Class:**
```python
class MockExperimentList:
    """Mock class compatible with DIALS C++ ExperimentList constructor"""
    def __init__(self, experiments=None):
        self.experiments = experiments or []
    
    def __len__(self):
        return len(self.experiments)
    
    def __getitem__(self, index):
        return self.experiments[index]
    
    def __iter__(self):
        return iter(self.experiments)
    
    # Required for isinstance() compatibility with C++ backends
```

### 7.1.2 Mock Strategy Evolution

**Proven Progression Pattern:**
1. **Mock → MagicMock:** For objects requiring magic methods (`__getitem__`, `__and__`, `__or__`, `__iter__`)
2. **MagicMock → Real Components:** For authentic integration testing where possible
3. **Real DIALS flex Arrays:** Replace complex mock hierarchies with actual DIALS data structures

**Patching Strategy:**
*   **Method Patching:** Use `patch.object(adapter, '_method_name')` for internal method mocking
*   **Avoid Module-Level Patching:** Don't patch module-level imports; patch actual method calls
*   **flex Module Mocking:** Create comprehensive mocks with proper `bool`, `grid`, and `int` setup

### 7.1.3 Common C++ Integration Failures

**Known Failure Patterns:**
*   `ExperimentList([MagicMock])` fails due to C++ backend requiring real Experiment objects
*   Mock objects lacking proper structure for `isinstance()` checks
*   Incomplete mock setup for vectorized operations in `dials.array_family.flex` modules
*   Wrong patch targets (patching imports instead of method calls)

**Proven Fix Patterns:**
*   Use real DIALS `flex` arrays instead of complex mock hierarchies when possible
*   Create comprehensive `flex` module mocks with proper magic method support
*   Test import error scenarios using `builtins.__import__` patching with proper `sys.modules` cleanup
*   Enhance reflections mocks with proper `__contains__` and `__getitem__` setup

### 7.1.4 Realistic Testing Bounds

**Update Assertions to Match Detector Physics:**
*   Solid angle corrections should be < 3e6, not arbitrary small values
*   Use realistic tolerances based on actual detector geometries
*   Test with actual data ranges representative of production use

**Example Bound Updates:**
```python
# Before: Unrealistic assertion
assert correction_factor < 1e6  # Too restrictive

# After: Physics-based assertion  
assert correction_factor < 3e6  # Matches actual detector geometry
```

**This section is based on systematic test failure remediation achieving 64% reduction in failures. See `docs/LESSONS_LEARNED.md` for detailed case studies.**

**7.2 Test Suite Remediation Methodology**

When facing systematic test failures, use this proven remediation strategy:

**Systematic Failure Analysis:**
*   **Failure Categorization:** Group test failures by root cause (API changes, mock strategy issues, assertion problems, import errors).
*   **Mock Evolution Priority:** Transition systematically: Mock → MagicMock → Real Components as appropriate for each test case.
*   **Realistic Bounds:** Update assertions to match actual system behavior and detector physics (e.g., solid angle corrections < 3e6, not arbitrary small values).
*   **Error Handling Enhancement:** Improve bounds checking and edge case handling throughout test implementations.

**Proven Remediation Process:**
1. **Categorize Failures:** Group similar failures by failure type (import errors, mock issues, assertion problems, API incompatibilities).
2. **Fix by Category:** Apply targeted fixes to each category rather than attempting wholesale changes.
3. **Real Component Integration:** Replace complex mocking with actual components where feasible for more authentic testing.
4. **Iterative Validation:** Test each fix independently before proceeding to ensure no regressions are introduced.
5. **Regression Prevention:** Ensure fixes don't break existing passing tests - run full test suite after each category fix.

**Test Authenticity Guidelines:**
*   **Real Components Over Mocks:** Use actual DIALS `flex` arrays and objects instead of complex mock hierarchies where possible.
*   **API Compatibility:** Fix DIALS import issues and method signatures systematically when library versions change.
*   **Magic Method Support:** Use `MagicMock` for objects requiring `__getitem__`, `__and__`, `__or__`, `__iter__` operations.
*   **Bounds Validation:** Use realistic tolerances based on actual detector geometries and physical correction factors.

**Example Successful Remediation Pattern:**
```python
# Before: Complex mock failing with AttributeError
mock_detector = Mock()
mock_detector.__getitem__ = Mock(side_effect=AttributeError)

# After: MagicMock with proper magic method support
mock_detector = MagicMock()
mock_detector.__getitem__.return_value = mock_panel
mock_detector.__iter__.return_value = iter([mock_panel])

# Best: Real component when feasible
real_detector_data = flex.bool(flex.grid(height, width), True)
```

**Success Metrics:**
*   **Quantified Improvement:** Track test pass rates (e.g., "64% reduction in failures: 22 → 8")
*   **Stability Validation:** Ensure remediated tests pass consistently across multiple runs
*   **Regression Monitoring:** Verify that fixes don't introduce new failure modes
*   **Integration Success:** Confirm that fixed tests validate actual component behavior, not just mock interactions

**8. Error Handling**

*   Use custom exception classes where appropriate for application-specific errors.
*   Catch specific exceptions rather than generic ones.
*   Provide informative error messages.
*   Format errors into a standard result structure (e.g., a `TaskResult`-like object with `status: "FAILED"`, details in `content`/`notes`) at appropriate boundaries.
*   Adhere to the project's [Error Handling Philosophy] (e.g., in `ARCHITECTURE/overview.md` or similar) regarding returning structured errors vs. raising exceptions.
*   **Consistent Error Formatting in Orchestrators:** Components orchestrating calls to other components MUST implement consistent error handling, using helpers to standardize FAILED result creation and populate structured error details.
*   **Defensive Handling of Returned Data Structures:** Use defensive checks (`isinstance()`, `dict.get()`) when processing complex data structures returned from other components or external sources.

**9. IDL to Code Implementation**

*   **Contractual Obligation:** The IDL file is the source of truth. The implementation **must** match interfaces, method signatures (including type hints), preconditions, postconditions, and described behavior precisely.
*   **Naming:** Code names should correspond directly to IDL names.
*   **Parameters & Return Types:** Must match the IDL. Use data models for complex "Expected Data Format" parameters/returns.
*   **Error Raising:** Implement error conditions described in the IDL.
*   **Dependencies:** Implement dependencies (e.g., from `@depends_on`) using constructor injection.

**10. Logging Conventions**

*   **Early Configuration:** Configure logging as the **very first step** in application entry-point scripts **before importing any application modules**.
    ```python
    # Example (Entry Point Script - Python)
    import logging
    # --- Logging Setup FIRST ---
    LOG_LEVEL = logging.DEBUG # Or get from args/env
    logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # --- Application Imports (After logging) ---
    # from src.main import Application
    ```
*   **Module Loggers (Python Example):** Use `logger = logging.getLogger(__name__)` at the top of each module.
*   **Setting Specific Levels (Python Example):** If needed, explicitly set levels for specific module loggers *after* `basicConfig` in your entry point: `logging.getLogger("src.noisy_module").setLevel(logging.WARNING)`.

**11. Utility Scripts**

*   **Path Setup:** Scripts outside main source/test directories MUST include robust path setup to reliably import project modules. Assume scripts might be run from different working directories; ideally, design them to be run from the project root.
*   **Execution Location:** Document the intended execution location.
*   **Environment Consistency:** Ensure scripts use the project's standard environment.

**12. Guidelines for DSLs or Complex Parsers/Evaluators (If Applicable)**

*   **Principle of Explicit Intent:** Ensure DSL syntax is unambiguous, especially for distinguishing executable code from literal data. Use quoting or specific data constructors for literal data.
*   **Separate Evaluation from Application:** Design core evaluation to determine an expression's *value*. Isolate logic that *applies* a function/operator to *already evaluated* arguments.
*   **Implement Robust and Explicit Dispatch Logic:** Clearly define dispatching rules for different language constructs and handle unrecognized/invalid constructs with specific errors.
*   **Validate Inputs at Key Boundaries (Defensive Programming):** Perform basic validation on inputs passed between internal evaluator functions.
*   **Ensure Contextual Error Reporting:** Error messages should pinpoint the semantic source of the error and include relevant context.
*   **Host Language Orchestration Pattern (Recommended Usage):**
    *   **Guideline:** Leverage the host language (e.g., Python) for complex data preparation before invoking a DSL evaluator.
    *   **Pattern:**
        1.  Prepare data in the host language.
        2.  Create DSL environment.
        3.  Bind prepared data to DSL variables.
        4.  Write focused DSL script referencing these variables.
        5.  Call DSL evaluator.
        6.  Process result in the host language.
    *   **Rationale:** Leverages host language strengths, keeps DSL focused on orchestration, simplifies DSL evaluator.

**13. Data Merging Conventions**

*   **Document Precedence:** Clearly document merging logic and precedence rules.
*   **Establish Conventions:** For common scenarios (e.g., merging status notes from called components with orchestrator notes), define a project convention (e.g., component's notes take precedence).
*   **Implement Correctly (Python Example for Dicts):**
    ```python
    # Correct precedence: component_data overwrites orchestrator_defaults
    final_data = orchestrator_defaults.copy()
    final_data.update(component_data)
    ```

**14. Documenting Component Interactions**

*   **Preferred Method:** Use a dedicated "Component Interactions" section within the component's IDL file (see `01_IDL_GUIDELINES.md`).
*   **Content:** Mermaid sequence diagrams and textual explanations for key scenarios.
*   **Maintenance:** Update this section when interaction patterns change significantly.

**15. DIALS Integration Best Practices**

### 15.1 Mandatory Data Type Detection

**Core Requirement:** Always check CBF headers for `Angle_increment` before processing. This value determines the processing pathway and prevents processing failures.

**Processing Mode Selection:**
*   **True Stills Data (Angle_increment = 0.0°):** Use the `DIALSStillsProcessAdapter`, which wraps the `dials.stills_process` Python API. Ensure PHIL parameters are appropriate for stills.
*   **Sequence Data (Angle_increment > 0.0°):** Use the `DIALSSequenceProcessAdapter`, which executes a sequential DIALS CLI workflow (`dials.import` → `dials.find_spots` → `dials.index` → `dials.integrate`). Use the critical PHIL parameters specified in `plan.md` (Section 0.6) for this route.

**Critical PHIL Parameters for Sequence Processing:**
```python
# REQUIRED parameters for oscillation data (not defaults):
phil_overrides = [
    "spotfinder.filter.min_spot_size=3",          # Not default 2
    "spotfinder.threshold.algorithm=dispersion",   # Not default  
    "indexing.method=fft3d",                       # Not fft1d
    "geometry.convert_sequences_to_stills=false"   # Preserve oscillation
]
```

### 15.2 Adapter Implementation Requirements

**Automatic Routing:** The `StillsPipelineOrchestrator` (or equivalent) must implement data type detection and routing logic automatically.

**Configuration Override:** Allow forcing a processing mode via configuration (`DIALSStillsProcessConfig.force_processing_mode`) to override auto-detection when necessary for debugging.

**Adapter Output Consistency:** Both adapters (`DIALSStillsProcessAdapter` and `DIALSSequenceProcessAdapter`) must return DIALS `Experiment` and `reflection_table` objects with identical structure for downstream compatibility.

### 15.3 Validation and Debugging

**Primary Validation Method:** Q-vector consistency checking (`q_model` vs. `q_observed`) with tolerance typically `|Δq|` < 0.01 Å⁻¹. Pixel-based validation is a secondary diagnostic tool.

**Debugging Strategy:**
1. **Confirm Processing Route:** First verify the correct processing route was chosen based on CBF header analysis
2. **Compare DIALS Logs:** Compare logs from failing adapter with manually executed, working DIALS workflow for that data type
3. **PHIL Parameter Verification:** Ensure critical parameters are correctly applied, especially for sequence processing
4. **CLI Fallback:** Use CLI-based adapters as fallback for unstable Python APIs

**API Compatibility Monitoring:**
*   Test DIALS imports independently before integration 
*   Implement version compatibility checks where feasible
*   Watch for common breaking changes: PHIL scope imports, method signatures, import paths

**This section incorporates lessons from systematic DIALS integration failures. See `docs/LESSONS_LEARNED.md` for detailed troubleshooting case studies.**

### 15.4 Testing Sequential and Scan-Varying Data Processing

**Mandatory Test Requirements:** Tests **must** verify that observations from different frames, corresponding to the *same detector pixel*, are mapped to *different HKL coordinates* when using scan-varying crystal models, consistent with crystal rotation.

**Required Test Scenarios:**

1. **Frame-Specific Transformation Validation:**
   ```python
   def test_frame_specific_hkl_mapping(self):
       """Verify same pixel maps to different HKL for different frames"""
       # Test same detector pixel across multiple frames
       pixel_coord = (100, 200)  # Fixed detector pixel
       frame_indices = [0, 10, 20]  # Different rotation angles
       
       hkl_results = []
       for frame_idx in frame_indices:
           q_lab = calculate_q_for_pixel(pixel_coord, frame_idx)
           hkl = transform_q_to_hkl(q_lab, frame_idx)
           hkl_results.append(hkl)
       
       # Assert different HKL coordinates for same pixel
       assert not np.allclose(hkl_results[0], hkl_results[1])
       assert not np.allclose(hkl_results[1], hkl_results[2])
   ```

2. **Reciprocal Space Consistency Testing:**
   ```python  
   def test_reciprocal_space_point_consistency(self):
       """Verify same reciprocal space point maps to same HKL from different frames"""
       # Calculate frames where a specific HKL point is accessible
       target_hkl = [1.0, 2.0, 3.0]
       
       accessible_frames = find_frames_for_hkl(target_hkl)
       hkl_results = []
       
       for frame_idx in accessible_frames:
           # Calculate detector position for this HKL at this frame
           pixel_coord = calculate_pixel_for_hkl(target_hkl, frame_idx)
           q_lab = calculate_q_for_pixel(pixel_coord, frame_idx) 
           hkl_observed = transform_q_to_hkl(q_lab, frame_idx)
           hkl_results.append(hkl_observed)
       
       # Assert same HKL coordinates for same reciprocal space point
       for hkl_result in hkl_results:
           assert np.allclose(hkl_result, target_hkl, rtol=1e-6)
   ```

3. **Data Flow Integration Testing:**
   ```python
   def test_frame_index_propagation(self):
       """Verify frame indices propagate through extraction to voxelization"""
       # Test with sequence data containing multiple frames
       sequence_cbf_file = "test_data/sequence_0.1deg_increment.cbf"
       
       # Extract diffuse data
       extraction_result = data_extractor.extract_from_still(
           inputs={"cbf_image_path": sequence_cbf_file},
           config=extraction_config
       )
       
       # Verify frame_indices array is present and correct
       npz_data = np.load(extraction_result.npz_path)
       assert "frame_indices" in npz_data
       assert len(npz_data["frame_indices"]) == len(npz_data["intensities"])
       
       # Test voxelization uses frame indices correctly
       voxel_accumulator.add_observations(
           still_id=1,
           q_vectors_lab=npz_data["q_vectors"],
           intensities=npz_data["intensities"],
           sigmas=npz_data["sigmas"],
           frame_indices=npz_data["frame_indices"]
       )
       
       # Verify observations are binned with correct transformations
       # (Implementation-specific verification)
   ```

**Validation Requirements:**
- Tests must distinguish between still image processing (static orientation) and sequence processing (scan-varying orientation)
- Integration tests must verify entire data flow from extraction through voxelization
- Performance tests should confirm frame-specific transformations don't create unacceptable bottlenecks

**Error Condition Testing:**
- Test missing frame_indices for sequence data
- Test frame index out of bounds conditions  
- Test mixed still/sequence data handling

**This section ensures correct handling of the pixel-to-voxel mapping that was identified as a critical error in sequential data processing. See `docs/LESSONS_LEARNED.md` for the complete case study.**

**16. Service/Plugin Registration and Naming (If Applicable)**

*   **Naming Constraints:** If registering callables (tools, plugins, services) that will be exposed to external systems (e.g., LLMs, other APIs), ensure their names conform to any constraints imposed by those external systems (e.g., regex for valid characters, length limits).
*   **Lookup and Invocation:** The key used for registration is typically the identifier used for lookup and invocation.
*   **Recommendation:** Prefer names valid for both internal use and external exposure to avoid complex mapping layers.

**17. Standard Development Process**

To ensure that the implementation rules outlined in this document are consistently applied, all significant development and refactoring tasks must follow a structured planning and execution process.

The standard tool for managing this process is the **Implementation Checklist**. Before beginning implementation, a checklist must be created that details the plan, from understanding the IDL contract to final verification.

All developers and agents must refer to the official guide for creating these checklists:

**`agent/writing_checklists.md`**

This guide provides templates and detailed instructions for creating task-specific checklists that ensure:
- IDL contract compliance
- Proper testing strategy
- Incremental implementation with verification points
- Documentation updates
- Code quality standards adherence
