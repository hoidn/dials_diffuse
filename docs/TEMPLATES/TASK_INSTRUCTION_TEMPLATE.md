# Task Implementation Instructions

*   **Task Name/ID:** `[Link to Issue Tracker Ticket or Task Name]`
*   **Assigned To:** `[Developer Name]`
*   **Assigned By:** `[Tech Lead/Assigner Name]`
*   **Date Assigned:** `[Date]`
*   **Relevant ADRs/Docs:**
    *   `[Link to relevant ADR(s), e.g., ../ARCHITECTURE/adr/ADR_001.md]`
    *   `[Link to relevant pattern document(s), e.g., ../ARCHITECTURE/patterns/pattern_x.md]`
    *   `[Link to relevant library integration guide(s), e.g., ../LIBRARY_INTEGRATION/library_y_guide.md]`
    *   `[Link to project's 01_IDL_GUIDELINES.md, 02_IMPLEMENTATION_RULES.md]`

---

**1. Task Goal:**

*   `[Clearly describe the overall objective of this task. What feature should be implemented or what bug should be fixed? What is the desired outcome? Keep it concise.]`

**2. Context & Requirements:**

*   **Primary IDL File(s) & Contract:**
    *   The main specification for the component(s) you will be working on is defined in:
        *   `[Link to primary IDL file, e.g., ../../src/component_x/module_y_IDL.md]`
        *   `[Link to another relevant IDL file, if any]`
    *   **Key Focus Areas in IDL(s):** Pay close attention to the following methods/behaviors/error conditions defined in the IDL specification(s):
        *   `[Method/Behavior 1 from IDL - e.g., ComponentX.process_data]`
        *   `[Method/Behavior 2 from IDL - e.g., Handling of 'Expected Data Format' for parameter Z]`
        *   `[Error condition from IDL - e.g., @raises_error(condition="DataNotFound", ...)]`
        *   `[...]`
*   **Dependencies & Interactions:**
    **Action Required:** Before implementing the interactions below, please review the linked `IDL / Docs` for each dependency. Pay close attention to the documentation for any external libraries, APIs, or tools to fully understand their expected usage, parameters, and return values.
    *   This task involves interacting with the following components/libraries:
        *   **`[Dependency 1 Name, e.g., DataStoreService]`:**
            *   IDL/Docs: `[Link to Dependency 1 IDL or relevant docs, e.g., ../../src/services/data_store_IDL.md]`
            *   Key Interaction: `[Explain how your code should use this dependency. E.g., "You will need to call data_store_service.fetch_record(record_id). Expect a RecordObject or null back. Handle potential connection errors."] `
        *   **`[Dependency 2 Name, e.g., External [SomeAPI] Service]`:**
            *   IDL/Docs: `[Link to relevant docs, e.g., ../LIBRARY_INTEGRATION/[SomeAPI]_client_guide.md]`
            *   Key Interaction: `[E.g., "Use the [SomeAPI]Client instance (passed as api_client) to call api_client.submit_job(job_params). Ensure you construct the job_params dictionary correctly based on the API's requirements. Handle potential API rate limits or authentication errors."] `
        *   `[...]`
    *   **Data Structures:** You will primarily work with these data structures defined in `[e.g., ../ARCHITECTURE/types.md or component-specific IDL files]`:
        *   `[List relevant data models/structs, e.g., UserProfile, OrderDetails, OperationOutcome]`

**3. Provided Stubs (If Applicable):**

*   The following skeleton files and test stubs have been created for you:
    *   `[path/to/implementation_file.ext]` (Contains class/function stubs)
    *   `[path/to/test_file.ext]` (Contains empty test function stubs)
    *   `[...]`
*   Please implement the logic directly within these files. Ensure you fetch the latest changes if working on a shared branch.

**4. Implementation Plan:**

*   Implement the logic for `[ClassName.method_name or function_name]` in `[path/to/implementation_file.ext]`.
*   Follow these specific steps:
    1.  `[Detailed step 1: e.g., "Retrieve the 'config_options' dictionary from the 'settings' parameter."]`
    2.  `[Detailed step 2: e.g., "Validate that 'config_options' contains the required 'timeout' key using the 'get_required_config' helper."]`
    3.  `[Detailed step 3: e.g., "If validation passes, call the 'dependency_x.process_item(item_data, timeout_value)' method. Wrap this call in a try...except block to catch 'DependencyXTimeoutError'."]`
        *   `[Sub-step/detail: e.g., "If 'DependencyXTimeoutError' occurs, log a warning and return a FAILED OperationOutcome with reason 'DEPENDENCY_TIMEOUT'."]`
    4.  `[Detailed step 4: e.g., "If the call to 'dependency_x.process_item' is successful, transform its result into the 'ProcessedData' model."]`
    5.  `[...]`
*   **Data Models:** `[Specify any data models the developer needs to use or be aware of, e.g., "Ensure the final return value is an OperationOutcome object/dictionary."]`
*   **Key Considerations:** `[Highlight any tricky parts, edge cases to consider, or specific project patterns to follow, e.g., "Remember to use the shared 'error_reporting_util' for consistent error logging.", "Ensure all external calls are idempotent if possible."]`
    *   **Host Language/DSL Interaction (If Applicable):** If this task involves invoking a Domain-Specific Language (DSL) evaluator with complex inputs, ensure data preparation logic resides in the host language code. Pass prepared data via the DSL environment bindings. Refer to `02_IMPLEMENTATION_RULES.md` for guidelines.

**5. Testing Plan:**

*   **Testing Strategy Overview:** `[Tech Lead provides brief context, e.g., "Focus on integration testing the main success path and key error conditions. Mock the ExternalAPIService."]`
*   **Test Files:** Implement tests in `[path/to/test_file.ext]`.
*   **Detailed Test Cases:** Implement the following test functions (stubs may be provided):
    *   **`test_function_name_scenario_1`**:
        *   **Purpose:** `[e.g., "Verify successful processing when all inputs are valid and dependencies behave as expected."]`
        *   **Setup/Fixtures:** `[e.g., "Use 'mock_data_store' fixture. Configure 'mock_external_api.submit.return_value = successful_api_response'."]`
        *   **Assertions:** `[e.g., "mock_data_store.save.assert_called_once_with(expected_record)`. `assert result.status == 'SUCCESS'". "assert result.data['processed_id'] is not None".]`
    *   **`test_function_name_scenario_2_error_case`**:
        *   **Purpose:** `[e.g., "Verify correct error handling when 'DependencyX' raises 'ItemNotFoundError'."]`
        *   **Setup/Fixtures:** `[e.g., "Configure 'mock_dependency_x.process_item.side_effect = ItemNotFoundError('item not found')'."]`
        *   **Assertions:** `[e.g., "assert result.status == 'FAILURE'". "assert result.error_code == 'ITEM_NOT_FOUND_IN_DEPENDENCY'". "mock_dependency_x.process_item.assert_called_once()".]`
    *   `[...]` *(Add entries for all required test cases)*

**6. Running Tests & Debugging:**

*   **Running Tests:**
    *   To run all tests for this component: `[e.g., pytest tests/component_x/test_module_y.py]`
    *   To run a specific test: `[e.g., pytest tests/component_x/test_module_y.py -k test_function_name_scenario_1]`
*   **Debugging Tips:**
    *   `[Tip 1: e.g., "Use logging.debug() extensively within your implemented logic to trace variable values."]`
    *   `[Tip 2: e.g., "Set breakpoints using breakpoint() before critical calls to inspect inputs."]`
    *   `[Tip 3: e.g., "If tests fail on assertions about mock calls, print the mock_dependency.method_calls attribute to see exactly how the mock was called."]`
    *   `[Tip 4: e.g., "Common Issue: Ensure data models are being instantiated correctly if parsing external data."]`
    *   **Asking for Help:** If you are stuck for more than `[e.g., 30-60 minutes]` after trying to debug, please reach out. Explain what you are trying to achieve, what you have tried, and what error you are seeing.

**7. Definition of Done:**

*   [ ] All implementation steps in Section 4 are completed.
*   [ ] All detailed test cases in Section 5 are implemented and pass.
*   [ ] Code passes linting (`[e.g., make lint]`).
*   [ ] Code passes formatting (`[e.g., make format]`).
*   [ ] The full project test suite passes (`[e.g., make test]`).
*   [ ] You have performed a self-review of your code for clarity and correctness against the IDL specification and project rules.
*   [ ] Your working memory log (`TEMPLATES/WORKING_MEMORY_LOG_TEMPLATE.md` or equivalent) has been updated with your work.
*   [ ] Code is committed with a clear commit message.
*   [ ] Pull Request (if applicable) is created and linked to the issue ticket.
*   [ ] Relevant IDL specification(s) have been reviewed and updated if the implementation required contract changes (discuss with Tech Lead).

**8. Notes/Questions (For Developer to fill in):**

*   `[Space for the developer to jot down questions, observations, or issues encountered during implementation.]`
