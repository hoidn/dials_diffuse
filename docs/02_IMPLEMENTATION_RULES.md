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

**4. Coding Style and Formatting**

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
*   **Naming:** Follow language-standard naming conventions (e.g., snake_case for Python variables/functions, CamelCase for Python classes). Use descriptive names.

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
            print(f"Invalid task parameters: {e}")
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
*   **Emphasis on Integration/Functional Tests:** Prioritize integration tests that verify the collaboration between real component instances according to their IDL contracts.
*   **Mocking and Patching Strategy (Python/unittest.mock Example):**
    *   **Guideline 1: Test Dependency Injection by Passing Mocks:** For classes using Dependency Injection, pass mock *objects* into the constructor during test setup. Use `with patch.object(mock_instance, 'method_name', ...)` for per-test configuration of these injected mocks.
    *   **Guideline 2: Patch Where It's Looked Up:** When using `patch` (e.g., `unittest.mock.patch`), the `target` string must be the path to the object *where it is looked up/imported*, not necessarily where it was defined.
    *   **Guideline 3: Prefer Specific Patching:** Apply patches only where needed (e.g., `@patch` on test functions, `with patch(...)` context managers) rather than broad, automatic patching.
    *   **Guideline 4: Minimize Mocking (Strategic Use):** Avoid excessive mocking. Mock primarily at external boundaries (external APIs, filesystem if necessary) or for components that are slow, non-deterministic, or have significant side effects not relevant to the test.
    *   **Guideline 4a: Ensure Mock Type Fidelity for External Libraries:** When mocking methods from external libraries, ensure mock return values match the **expected type** returned by the real library, especially if your code uses `isinstance()`. Use real library types for mock data if possible.
    *   **Guideline 6: Verify Mock Calls Correctly:** Assert mock calls on the correct mock object (the one created by the patcher or injected).
    *   **Guideline 7.X: Mock Configuration for Multiple/Complex Interactions:** Configure mocks appropriately *before each specific interaction* within a test.
    *   **Guideline 7a: Maintain Test Environment Dependency Parity:** Critical runtime dependencies MUST be installed and importable in the test environment. Avoid dummy class fallbacks in test setup.
    *   **Guideline 7b: Testing Wrapper Interactions / Library Boundaries:** Test argument preparation logic for external library calls separately from the external call itself (which can be mocked).
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

**15. Service/Plugin Registration and Naming (If Applicable)**

*   **Naming Constraints:** If registering callables (tools, plugins, services) that will be exposed to external systems (e.g., LLMs, other APIs), ensure their names conform to any constraints imposed by those external systems (e.g., regex for valid characters, length limits).
*   **Lookup and Invocation:** The key used for registration is typically the identifier used for lookup and invocation.
*   **Recommendation:** Prefer names valid for both internal use and external exposure to avoid complex mapping layers.
