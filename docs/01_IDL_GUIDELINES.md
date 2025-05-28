# Interface Definition Language (IDL) Guidelines

**i. Overview & Purpose**

These guidelines define a system for creating and understanding Interface Definition Language (IDL) specifications and their corresponding code implementations. The process is designed to be **bidirectional**:

1.  **IDL-to-Code:** Use a defined IDL as a comprehensive specification to generate compliant, high-quality code.
2.  **Code-to-IDL:** Reduce existing code to its essential IDL specification, abstracting away implementation details and presentation logic to reveal the core behavioral specification and functional contract.

The goal is a clear, language- and UI-agnostic **specification (SPEC)** that defines the *what* (the functional contract and essential behavior) and separates it from the *how* (the specific implementation details within the code).

**ii. IDL Creation & Structure Guidelines**

*(These apply both when writing IDLs from scratch and when reducing code to IDL)*

1.  **Object Oriented (or Component-Based):** Structure the IDL using modules and interfaces (or equivalent concepts like components/services) to represent logical groupings and entities.
2.  **Dependency Declaration:**
    *   **Purpose:** To explicitly declare dependencies required to fulfill the interface's contract. This clarifies coupling at the design level.
    *   **Syntax (Example using comments):** Use comment lines placed immediately before the `interface` definition (or equivalent).
        *   For dependencies on other IDL-defined interfaces/modules:
            `# @depends_on([idl_module_or_interface_name], ...)`
        *   For dependencies on abstract types of external resources or systems:
            `# @depends_on_resource(type="Database", purpose="Storing user profiles")`
            `# @depends_on_resource(type="MessageQueue", purpose="Processing background jobs")`
            `# @depends_on_resource(type="FileSystem", purpose="Reading/writing files")`
    *   **Target:**
        *   `@depends_on`: Names must refer to other `module` or `interface` definitions within the IDL system.
        *   `@depends_on_resource`: `type` is an abstract category (e.g., "Database", "FileSystem", "ExternalAPI"), `purpose` is a brief description.
    *   **Implication:** These declarations signal requirements for the implementation. `@depends_on` implies needing access to implementations of other IDL contracts. `@depends_on_resource` implies needing access to a specific *kind* of external system or resource, configured appropriately.

### Implementation Conformance

The IDL file serves as a **strict contract**. Implementations (e.g., in Python) MUST adhere precisely to all aspects defined in the IDL, including:

*   **Interfaces and Signatures:** Class/component names, method names, parameter names, parameter order, type hints (mapping IDL types to language equivalents), and return types must match exactly.
*   **Behavior:** The implemented logic must fulfill the functional description provided in the `Behavior:` block.
*   **Preconditions/Postconditions:** The implementation must respect documented `Preconditions:` and guarantee `Postconditions:` upon successful execution.
*   **Data Structures:** Implementations must correctly handle data structures as defined (e.g., via `Expected Data Format`, `struct` definitions, or linked type definitions), including required keys or fields. Internal validation logic within components must align with these IDL specifications.
*   **Error Conditions:** Documented error conditions (e.g., via `@raises_error`) must be handled, either by raising the specified (or a corresponding mapped) exception or by returning a failure status as defined by the project's error handling philosophy (see `02_IMPLEMENTATION_RULES.md`).

3.  **Design Patterns:** Utilize (when creating IDLs) or identify (when reducing code) interfaces supporting established design patterns like Factory, Builder, Strategy where they clarify the design or improve flexibility. The IDL defines the *contract* and potentially the *behavioral role* of these patterns.
4.  **Complex Parameters (e.g., JSON, Structured Objects):**
    *   **Preference for Structure:** If the IDL syntax supports defining custom structures/records/data classes (see `struct` below), prefer them for complex data transfer to maximize type safety at the interface level.
    *   **Serialized String Fallback (e.g., JSON):** Where structured types are not feasible in the IDL language or cross-language string simplicity is paramount, a single serialized string parameter (e.g., JSON string) can be used.
    *   **Mandatory Documentation:** *Always* document the exact expected data format within the IDL comment block (e.g., `// Expected Data Format (JSON): { "key1": "type1", "key2": "type2" }`).
5.  **Defining Custom Data Structures (`struct` or equivalent):**

*   **Purpose:** To explicitly define the structure of complex data types passed between components, improving clarity and mapping directly to implementation structures (like Pydantic models in Python, or classes/records in other languages). Replaces reliance on generic `dict`/`map` types with comments for complex data.
*   **Syntax (Example):** Use a `struct` block (or similar keyword based on IDL language), typically within a `module` or global types definition.

    ```
    // Example IDL-like syntax for a struct
    struct StructName {
        field_name1: type; // e.g., string, int, boolean, list<AnotherStruct>, map<string, int>
        field_name2: optional type; // Use 'optional' for fields that may be absent/null
        field_name3: union<type1, type2>; // Use 'union' if a field can be one of multiple types
        // ... other fields
        // Use consistent naming (e.g., snake_case or camelCase) for field names.
    }
    ```

*   **Placement Strategy:**
    *   **`ARCHITECTURE/types.md` (or equivalent):** Define globally shared structures used across many modules/components (e.g., `OperationOutcome`, `StandardRequest`) in this central file.
    *   **Module-Specific IDL File:** Define structures primarily used by interfaces within a single module/component directly within that module's IDL file.

*   **Referencing:** Use the defined `StructName` as a type name in method parameters, return types, or within other `struct` definitions.

*   **Example:**

    ```
    // In ARCHITECTURE/types.md (Conceptual Example)
    struct OperationOutcome {
        status: string; // e.g., "SUCCESS", "FAILURE", "PENDING"
        message: optional string;
        data: optional Any; // Or specific union type if known
    }

    // In src/data_processor/data_processor_IDL.md
    module src.data_processor {

        // Locally defined struct
        struct ProcessedItem {
             item_id: string;
             processed_value: float;
             notes: optional string;
        }

        // Struct referencing another struct
        struct BatchProcessingResult {
             batch_id: string;
             items_processed: list<ProcessedItem>;
             overall_status: string; // Could reference OperationOutcome.status conceptually
             error_details: optional string;
        }

        // Interface using the defined structs
        interface DataProcessor {
             BatchProcessingResult process_data_batch(list<string> raw_data_items);

             // Method using a globally defined struct
             OperationOutcome check_system_health();
        }
    }
    ```
*   **Dependency Note (Optional Convention):** If an interface relies heavily on a type defined in the global types file, you *may* add a comment like `# @depends_on_type(ARCHITECTURE.types.OperationOutcome)` near the interface definition for extra clarity.

6.  **Clarity and Behavioral Specification:** The IDL must clearly express the *purpose* and *complete expected behavior* of each interface and method. This is achieved through:
    *   Well-chosen names for interfaces, methods, and parameters.
    *   **Comprehensive documentation comments:** These are critical for the SPEC. They must detail:
        *   **Preconditions:** Necessary conditions before calling.
        *   **Postconditions:** Expected outcomes, return value guarantees, and significant state changes after successful execution.
        *   **Essential Algorithms/Logic:** A conceptual description of any core algorithms or business logic the method implements, sufficient to understand its behavior without seeing the code.
        *   **Conceptual State:** Description of key internal state variables managed by the interface (if any) and how methods affect them.
        *   **Error Conditions:** Use an error annotation (see below) for specific, named error conditions that are part of the contract.
        *   Data format documentation if applicable.
7.  **Completeness (Behavioral Specification):** The IDL must represent the *complete behavioral specification* of the component's public interface and its essential, observable behavior.
8.  **Error Condition Annotation:** Use a dedicated annotation to formally define specific, named error conditions that are part of the interface's contract.
    *   **Syntax (Example using comments):** Use comment lines within the method's documentation block.
        `// @raises_error(condition="UniqueErrorCode", description="Explanation of when this error occurs.")`
    *   **Purpose:** To standardize the reporting of specific failure modes beyond simple success/failure, making the contract more precise. `condition` should be a stable identifier.
    *   **Note:** While this documents potential exceptions/errors, the project's error handling philosophy (see `02_IMPLEMENTATION_RULES.md`) should guide whether errors are raised as exceptions or returned as structured error responses.

**iii. Code-to-IDL Reduction Guidelines**

*(These specific rules apply only when generating an IDL from existing code)*

1.  **Goal: Extract the Contract:** The primary objective is to distill the code down to its public interface and functional guarantees, omitting all non-essential implementation details.
2.  **Mapping:**
    *   Public classes/modules generally map to IDL `module` or `interface`.
    *   Public methods/functions map to IDL method definitions.
    *   Method signatures (name, parameter types, return type) must be accurately reflected.
3.  **Dependency Identification:** Identify dependencies on other components *that are also being defined via IDL*. Represent these using the **dependency declaration** syntax (e.g., `# @depends_on(...)`) described in section ii.2. Exclude dependencies on third-party libraries or internal implementation details not represented by an interface in the IDL.
4.  **Documentation Extraction:**
    *   Infer **preconditions** from input validation, assertions, and documentation comments in the code.
    *   Infer **postconditions** from return value guarantees, state changes described in documentation, or observable outcomes. Document these clearly.
    *   Identify and document **invariants** – properties of the object's conceptual state that hold true between public method calls.
    *   Extract descriptions of **essential algorithms or core logic** from code/comments and summarize them conceptually in the method documentation.
    *   Identify **specific error conditions** raised or returned that represent contractual failure modes. Document these using the error annotation.
    *   If complex objects/dictionaries are passed, represent them using the **complex parameters** guideline (ii.4) and document the format.
5.  **Exclusion Criteria:** The following elements **must be excluded** from the generated IDL *structure* (interfaces, methods) as they are implementation details, presentation logic, or non-functional aspects. However, *essential behavioral aspects* derived from these (like core logic or error conditions) should be *described* in the IDL documentation comments or captured via annotations.
    *   **Presentation Logic:** Any code related to GUIs, TUIs, web page rendering, console output formatting.
    *   **Internal Implementation Code:** Private methods, helper functions, specific algorithms, internal data structures. (Note: The *behavior* or *purpose* of essential algorithms or the *conceptual* nature of key internal state *should* be documented).
    *   **Type Enforcement/Validation Code:** The *internal logic* for validation. The *requirement* for valid input is a precondition. Specific validation failure *error conditions* might be documented if contractual.
    *   **Non-Functional Code:** Logging statements, metrics collection, performance monitoring, debugging utilities, internal comments explaining *how* the code works (vs. *what* it guarantees).
    *   **Language/Platform Specifics:** Boilerplate code, language-specific idioms, environment configuration loading, build system artifacts, specific library dependencies (unless they form part of public signatures or are abstracted via `@depends_on_resource`).
    *   **Internal Error Handling Mechanisms:** Specific exception types thrown/caught internally. Contractual, observable error conditions should be documented. Generic internal failures are not part of the IDL spec.
    *   Dependencies on concrete libraries or modules *not* represented by an IDL interface or abstracted via `@depends_on_resource`.
    *   **Internal Helper Components:** Classes, functions, or modules created solely as internal implementation details to support a public interface, and not intended for direct use by other independent components. The delegation of work to such helpers *should*, however, be documented in the `Behavior` section of the public interface's method(s) that use them.

**iv. IDL Template (Conceptual)**

```
// == BEGIN IDL TEMPLATE (Conceptual) ==
module [your_system_name].[component_name] {

    // Optional: Define shared data structures if IDL language supports
    // struct SharedData { ... }

    // Example interface demonstrating dependency declarations and annotations
    # @depends_on([another_interface_name], [shared_module_name]) // Dependency on other IDL contracts
    # @depends_on_resource(type="KeyValueStore", purpose="Caching intermediate results") // Abstract resource dependency
    interface [EntityName] {

        // Action/method definition
        // Preconditions:
        // - Define necessary conditions before calling.
        // - [parameter_name] must be positive.
        // - (If using complex param) Expected Data Format: { "key1": "type1", ... }
        // Postconditions:
        // - Define expected outcomes and state changes after successful execution.
        // - Returns the calculated result based on input.
        // - Internal cache (conceptual state) may be updated.
        // Behavior:
        // - Describe essential algorithm/logic here, e.g., "Calculates result using the [algorithm_name] algorithm."
        // - "If the result is found in the KeyValueStore cache, it's returned directly."
        // @raises_error(condition="InvalidInput", description="Raised if [parameter_name] is non-positive.")
        // @raises_error(condition="ResourceUnavailable", description="Raised if the KeyValueStore cannot be accessed.")
        [return_type] [method_name]([parameter_type] [parameter_name]);

        // Additional methods...

        // Invariants: (Optional: define properties that always hold true for this entity's conceptual state)
        // - Describe state invariants here, e.g., "Internal cache size never exceeds max limit."
    }

    // Another entity or component that [EntityName] might depend on
    interface [another_interface_name] {
        // ... methods ...
    }

    // Could also be a module containing utility interfaces/functions
    module [shared_module_name] {
        // ... interfaces or potentially functions if IDL language supports them ...
    }
}
// == END IDL TEMPLATE (Conceptual) ==
```

**v. Code Creation Rules**

*(These apply when implementing code based on an IDL)*

1.  **Strict Typing:** Always use strict typing in your implementation language. Avoid ambiguous or "any" types where possible.
2.  **Primitive Types Focus (Balanced):** Prefer built-in primitive and standard collection types where they suffice. Use well-defined data classes/structs (like Pydantic models) for related data elements passed together. Avoid "primitive obsession." Match IDL types precisely.
3.  **Portability Mandate:** Write code intended for potential porting to other languages (e.g., Java, Go, JavaScript). Use language-agnostic logic and avoid platform-specific dependencies or language features without clear equivalents where feasible.
4.  **Minimize Side Effects:** Strive for pure functions for data processing. Clearly document all necessary side effects (state mutation, I/O, external calls) associated with methods defined in the IDL, typically in the implementation's documentation, aligning with the IDL's postconditions.
5.  **Testability & Dependency Injection:** Design for testability. Use dependency injection; avoid tight coupling. Ensure methods corresponding to IDL definitions are unit-testable. Pay attention to the dependency declarations in the IDL to identify required dependencies that should likely be injected.
6.  **Documentation:** Thoroughly document implementation details, especially nuances not obvious from the IDL or code signature. Link back to the IDL contract being fulfilled.
7.  **Contractual Obligation:** The IDL is a strict contract. Implement *all* specified interfaces, methods, and constraints *precisely* as defined. Do not add public methods or change signatures defined in the IDL without updating the IDL.

**vi. Example (Generic Social Media Post)**

```
module [social_platform].[post_service] {

    # @depends_on([user_service.UserValidator]) // Depends on a user validation contract
    # @depends_on_resource(type="DataStore", purpose="Persisting post data")
    # @depends_on_resource(type="NotificationService", purpose="Sending notifications")
    interface PostManager {
        // Preconditions:
        // - User referenced by `user_id` in `post_data_json` exists (verified via UserValidator).
        // - Post content is non-null and within allowable size limits.
        // Expected Data Format (JSON for post_data_json): { "user_id": "string", "content": "string" }
        // Postconditions:
        // - A new post is created and persisted in the DataStore.
        // - If content contains mentions (e.g., @username), a notification task may be queued via NotificationService.
        // Behavior:
        // - Validates input JSON structure.
        // - Checks user existence via UserValidator.
        // - Checks content length.
        // - Persists post data to DataStore.
        // - Parses content for mentions and potentially triggers notifications.
        // @raises_error(condition="InvalidPostFormat", description="JSON format is incorrect.")
        // @raises_error(condition="UserNotFound", description="User specified in user_id does not exist.")
        // @raises_error(condition="ContentTooLong", description="Post content exceeds the size limit.")
        // @raises_error(condition="StorageFailure", description="Failed to persist post to DataStore.")
        void create_post(string post_data_json);

        // Preconditions:
        // - Post referenced by `post_id` exists (verified via DataStore lookup).
        // Postconditions:
        // - Returns the details of the post (content, author, stats) as a JSON string.
        // Behavior:
        // - Retrieves post data from DataStore.
        // - Formats data into the specified JSON structure.
        // @raises_error(condition="PostNotFound", description="Post specified by post_id does not exist.")
        // @raises_error(condition="StorageFailure", description="Failed to retrieve post details from DataStore.")
        string get_post_details(string post_id);
    }
}
```

**vii. Documenting Component Interactions (Optional but Recommended)**

To enhance understanding of how components collaborate, especially for complex workflows or critical interaction patterns, IDL files can optionally include a dedicated section for interaction documentation. This section typically uses sequence diagrams (e.g., using Mermaid syntax) and textual explanations.

**7.1. Standard Location and Markers**

Interaction documentation should be placed within the main IDL definition, but *outside* any specific `module` or `interface` blocks, usually towards the end of the file or in a clearly marked section.

Example Markers:
```
// == BEGIN COMPONENT INTERACTIONS ==
// ... diagrams and explanations ...
// == END COMPONENT INTERACTIONS ==
```

**7.2. Content**

This section can contain:

*   **Scenario Titles:** Use headings (e.g., `// === Workflow: Descriptive Title ===`) to delineate different interaction scenarios.
*   **Sequence Diagrams:** Use a diagramming syntax like Mermaid, enclosed in appropriate comment or code fences:
  ```
  // ```mermaid
  // sequenceDiagram
  //    participant A
  //    participant B
  //    A->>B: Request
  //    B-->>A: Response
  // ```
  ```
*   **Textual Explanations:** Provide a brief explanation of the diagram, the context of the interaction, and key steps or data flows.

**7.3. Example**

```
// == BEGIN COMPONENT INTERACTIONS ==
// This section illustrates how the [OrchestratorComponent] interacts with its dependencies.

// === Workflow: Processing a Complex Request ===
// This diagram shows the call flow when `process_complex_request` is invoked.

// ```mermaid
// sequenceDiagram
//    Client->>+[OrchestratorComponent]: process_complex_request("input_data")
//    [OrchestratorComponent]->>+[ValidatorComponent]: validate("input_data")
//    [ValidatorComponent]-->>-[OrchestratorComponent]: validation_result
//    alt validation_result is OK
//        [OrchestratorComponent]->>+[ProcessorComponent]: execute_processing("input_data")
//        [ProcessorComponent]-->>-[OrchestratorComponent]: processing_output
//        [OrchestratorComponent]->>+[LoggerComponent]: log_success("input_data", processing_output)
//        [LoggerComponent]-->>-[OrchestratorComponent]: ack
//    else validation_result is ERROR
//        [OrchestratorComponent]->>+[LoggerComponent]: log_failure("input_data", validation_result)
//        [LoggerComponent]-->>-[OrchestratorComponent]: ack
//    end
//    [OrchestratorComponent]-->>-Client: final_status
// ```
//
// **Explanation:**
// 1. The `Client` calls `process_complex_request` on the `[OrchestratorComponent]`.
// 2. The `[OrchestratorComponent]` first validates the input using the `[ValidatorComponent]`.
// 3. If validation succeeds, it proceeds to the `[ProcessorComponent]` and logs success.
// 4. If validation fails, it logs the failure.
// 5. A final status is returned to the `Client`.

// == END COMPONENT INTERACTIONS ==
```

**7.4. Alternative: Linked Interaction Documents**

For components with exceptionally numerous or detailed interaction scenarios, a separate Markdown file may be used. In such cases, the IDL file should include a clear reference:

```
// For detailed interaction diagrams and explanations, see:
// ./[component_name]_interactions.md
// (Example: ./orchestrator_interactions.md)
```
Or, using a custom tag:
```
// @see_interactions_doc(./[component_name]_interactions.md)
```
The linked file would then contain the diagrams and explanations. However, embedding directly within the IDL is preferred for co-location of contract and behavioral examples when feasible.

**7.5. Purpose and Maintenance**

This section is intended to:
* Clarify the behavioral contract of the component in relation to its dependencies.
* Aid developers in understanding how to use the component and how it fits into larger workflows.
* Serve as living documentation that should be updated when significant interaction patterns change.

While optional, adding this section is highly recommended for components that play a central role in orchestrating other services or have complex internal call flows involving multiple dependencies.
