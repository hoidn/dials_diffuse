**IDL Implementation Readiness Checklist**

**Project:** `[Your Project Name]`
**IDL File(s) Reviewed:** `_________________________` (e.g., `src/component_x/module_y_IDL.md`)

**Instructions:** For each IDL file defining interfaces, modules, or types intended for implementation, review against the following criteria. Mark each item as Yes (✅), No (❌), or N/A. Provide comments for any "No" answers, detailing the required changes or clarifications. An IDL is generally considered "implementation ready" when all applicable items are marked "Yes".

**I. Feature / User Story Coverage**

*   *(List the key features/user stories this IDL is intended to support. Examples below - **replace/add specific stories relevant to the IDL being reviewed**)*

| Feature/Story                                       | Relevant Methods/Types in IDL | Sufficiently Specified? | Comments / Missing Details                                                                 |
| :-------------------------------------------------- | :--------------------------- | :---------------------- | :----------------------------------------------------------------------------------------- |
| **Ex: User Registration via API Endpoint**          | `UserManager.register_user`  | ☐ ✅ ☐ ❌ ☐ N/A         | _e.g., Need clarity on password hashing requirements if not handled by a dependency._        |
| **Ex: Process Batch Data from Message Queue**       | `BatchProcessor.process_next_batch`, `DataItem` struct | ☐ ✅ ☐ ❌ ☐ N/A         | _e.g., Error handling for individual item failures within a batch needs more detail._      |
| **Ex: Retrieve Configuration for Component [X]**    | `ConfigService.get_config`   | ☐ ✅ ☐ ❌ ☐ N/A         | _e.g., How are dynamic/overridden configurations handled by this interface?_               |
| **Ex: Execute [CoreAlgorithm] with given inputs**   | `CoreModule.execute_algorithm` | ☐ ✅ ☐ ❌ ☐ N/A         | _e.g., Preconditions for input data ranges are not fully specified._                       |
| *(Add other relevant features/stories for this IDL)* |                              |                         |                                                                                            |

**II. Interface Definition & Clarity (e.g., `interface ...` or `class ...` in IDL)**

| #   | Criteria                                                                                                                                                                                             | Status          | Comments / Required Changes |
| :-- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :-------------------------- |
| 2.1 | **Clear Name:** Is the interface name clear, concise, and accurately reflect its purpose (e.g., `UserManager`, `ConfigService`)?                                                                       | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 2.2 | **Purpose Defined:** Is the overall purpose/responsibility of the interface clearly stated in the module or interface docstring/comment block?                                                         | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 2.3 | **Dependencies Declared:** Are all significant dependencies (other modules/interfaces, external resources like Filesystem, Shell, Database, ExternalAPI) explicitly listed (e.g., using `@depends_on`)? | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 2.4 | **Inheritance/Extension Clear:** If the interface conceptually extends another, is this clearly stated in comments or via IDL syntax?                                                                  | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 2.5 | **Interaction Documentation:** Are complex or key interactions involving this interface/module clearly documented, preferably with sequence diagrams (e.g., using Mermaid in the IDL file or linked doc)? | ☐ ✅ ☐ ❌ ☐ N/A |                             |

**III. Method / Function Signatures (e.g., `returnType methodName(...)` in IDL)**

| #   | Criteria                                                                                                                                                                                                                            | Status          | Comments / Required Changes |
| :-- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :-------------------------- |
| 3.1 | **Clear Method Names:** Are method names clear, using standard conventions (e.g., snake_case or camelCase per project style), and accurately describing the action performed?                                                        | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 3.2 | **Accurate Return Types:** Is the return type specified using appropriate primitives (`string`, `int`, `boolean`, `list<T>`, `map<K,V>`) or defined types/structs? Is `optional` used correctly? Is `union` used correctly?           | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 3.3 | **Accurate Parameter Types:** Are all parameters clearly named and typed using primitives, defined types/structs, `optional`, or `union`?                                                                                             | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 3.4 | **Complex Parameter Formats:** For complex dictionary/object parameters, is the expected structure clearly documented (e.g., via `Expected Data Format: { ... }` comment or by referencing a `struct`)?                               | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 3.5 | **Preconditions Documented:** Are the necessary conditions required *before* calling the method clearly listed under "Preconditions"?                                                                                                 | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 3.6 | **Postconditions Documented:** Are the expected outcomes, state changes, or guarantees *after* successful method execution clearly listed under "Postconditions"?                                                                    | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 3.7 | **Behavior Described:** Is the core logic and sequence of actions performed by the method clearly described under "Behavior"? Does it mention interactions with dependencies?                                                        | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 3.8 | **Error Conditions Declared:** Are potential error conditions and how they are signaled (e.g., raised exception type, specific return value/status code) documented (e.g., using `@raises_error` or within Behavior/Postconditions)? | ☐ ✅ ☐ ❌ ☐ N/A |                             |

**IV. Type Definitions / Custom Structures (If applicable, e.g., `struct ...` in IDL or `ARCHITECTURE/types.md`)**

| #   | Criteria                                                                                                                                                           | Status          | Comments / Required Changes |
| :-- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :-------------------------- |
| 4.1 | **Clear Type Names:** Are custom type names (e.g., `UserProfile`, `BatchResult`, `ApiRequestOptions`) clear and descriptive?                                         | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 4.2 | **Structure Defined:** Is the structure of custom types (e.g., fields within a struct/object, elements of a list/tuple) clearly defined with types for each element? | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 4.3 | **Consistency:** Are types used consistently across different IDL files where they are referenced?                                                                    | ☐ ✅ ☐ ❌ ☐ N/A |                             |

**V. Architectural Alignment**

| #   | Criteria                                                                                                                                                                                                                                                          | Status          | Comments / Required Changes |
| :-- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :-------------------------- |
| 5.1 | **Component Responsibilities:** Does the interface adhere to established component responsibilities (e.g., `[DataServiceComponent]` handles data persistence, `[BusinessLogicComponent]` handles core logic, `[ExternalAPIClientComponent]` isolates external calls)? | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 5.2 | **[Project-Specific Concern 1, e.g., Legacy System Decoupling]:** Does the IDL avoid defining or referencing [specific legacy patterns/components]?                                                                                                                 | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 5.3 | **[Project-Specific Concern 2, e.g., DSL/Orchestration Role]:** Does the IDL correctly reflect that [specific type of workflow composition] is handled exclusively by [the designated DSL/Orchestrator Component]?                                                    | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 5.4 | **[Project-Specific Concern 3, e.g., Core Executor Isolation]:** Does the `[CoreExecutorComponent]` IDL behavior forbid implicit environment access and mandate use of only passed parameters?                                                                        | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 5.5 | **Data Handling:** Does the design encourage parsing into specific types (aligning with "Parse, Don't Validate") rather than passing raw dicts/lists widely? (Assessed via parameter types and data format comments).                                                | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 5.6 | **IDL Versioning:** Is the interface version clearly marked (e.g., `[Interface:ComponentName:1.0]`) and consistent with related documentation?                                                                                                                      | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 5.7 | **External API Interaction (If applicable):** Does the interface correctly use any designated Manager/Bridge classes for external API interactions where applicable?                                                                                                | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 5.8 | **[Project-Specific Concern 4, e.g., Configuration Handling]:** Does the component correctly receive, pass, or act upon `[ConfigObjectType]` if it's involved in a configurable workflow?                                                                           | ☐ ✅ ☐ ❌ ☐ N/A |                             |

**VI. Overall Readiness**

| #   | Criteria                                                                                                                                         | Status          | Comments / Required Changes |
| :-- | :----------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :-------------------------- |
| 6.1 | **Completeness:** Does the IDL provide enough detail to implement the described functionality *for the targeted features/stories* without ambiguity? | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 6.2 | **Consistency:** Is the terminology, naming, and style consistent within the file and with other project IDLs?                                    | ☐ ✅ ☐ ❌ ☐ N/A |                             |
| 6.3 | **Actionability:** Can a developer reasonably translate this IDL specification into code following the project's implementation rules?              | ☐ ✅ ☐ ❌ ☐ N/A |                             |

**Summary:**

*   **Feature Coverage Assessment:** `___________________________________________________________`
    *(Summarize if the IDLs sufficiently cover the necessary interfaces/types/behaviors for the target features)*
*   **Overall Readiness:** ☐ Ready for Implementation / ☐ Needs Revision
*   **Key Issues / Blockers (if any):**
    *   `___________________________________________________________`
    *   `___________________________________________________________`
*   **Next Steps:** `___________________________________________________________`
