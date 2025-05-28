# Project Rules and Conventions

**1. Purpose**

This document outlines general project-level rules, conventions, and processes that complement the coding and IDL guidelines. These rules help maintain a consistent and organized development environment.

**2. Directory Structure (Recommended)**

A consistent directory structure is crucial for navigability and maintainability. We recommend the following structure as a baseline, which can be adapted as needed:

```
[YourProjectName]/
├── .git/                     # Version control
├── .gitignore
├── pyproject.toml            # Or requirements.txt, package.json, etc. (Project dependencies & config)
├── Makefile                  # Optional: for common development tasks (test, lint, format, build)
├── README.md                 # Top-level project overview, setup, and usage
├── src/                      # Main source code for the application/library
│   ├── [your_project_package_name]/ # Primary package
│   │   ├── __init__.py
│   │   ├── main.py             # Main application entry point (if applicable)
│   │   ├── components/         # Logical grouping of components/modules
│   │   │   ├── component_a/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── module_a1.py
│   │   │   │   └── module_a1_IDL.md # IDL specification for module_a1
│   │   │   └── component_b/
│   │   │       └── ...
│   │   ├── core/               # Core functionalities, shared utilities, base classes
│   │   │   ├── __init__.py
│   │   │   ├── utils.py
│   │   │   └── errors.py
│   │   └── models/             # Data models (e.g., Pydantic models, data classes)
│   │       ├── __init__.py
│   │       └── general_models.py
│   └── scripts/                # Utility or operational scripts (not part of main app)
│       └── some_script.py
├── tests/                    # All tests
│   ├── __init__.py
│   ├── conftest.py             # Shared pytest fixtures (if using pytest)
│   ├── components/             # Tests mirroring src/components structure
│   │   └── component_a/
│   │       ├── __init__.py
│   │       └── test_module_a1.py
│   └── core/
│       └── test_utils.py
└── docs/                     # All project documentation (as outlined in this template)
    ├── 00_START_HERE.md
    ├── 01_IDL_GUIDELINES.md
    ├── 02_IMPLEMENTATION_RULES.md
    ├── 03_PROJECT_RULES.md     # This file
    ├── ...
    └── ARCHITECTURE/
        ├── overview.md
        ├── types.md
        └── adr/
            ├── README.md
            └── ADR_TEMPLATE.md
```

*   **IDL Files:** Place IDL files (e.g., `*_IDL.md`) alongside the code module they define (e.g., `src/components/component_a/module_a1.py` and `src/components/component_a/module_a1_IDL.md`).
*   **Source Root:** The `src/` directory is typically the root for application code imports.
*   **Test Structure:** The `tests/` directory should mirror the structure of `src/` for easy navigation.

**3. Module/File Length Guideline**

*   **Principle:** Strive to keep modules and files concise and focused on a single responsibility (Single Responsibility Principle).
*   **Guideline:** As a general rule of thumb, aim to keep individual source code files (e.g., Python `.py` files) under **`[e.g., 300-500]` lines of code (LoC)**, excluding comments and blank lines.
*   **Rationale:** Shorter files are generally easier to understand, test, and maintain.
*   **Action:** If a file significantly exceeds this guideline, consider it a signal to refactor. Look for opportunities to extract classes, functions, or sub-modules. Refer to `04_REFACTORING_GUIDE.md`.
*   **Exception:** This is a guideline, not an absolute rule. Some files, by their nature (e.g., extensive data definitions, large auto-generated files), might be longer. Justify exceptions if they occur.

**4. Version Control (e.g., Git) Workflow**

*   **Branching Strategy:**
    *   `main` (or `master`): Stable, production-ready code. Protected branch. Merges typically happen via Pull Requests.
    *   `develop`: Integration branch for ongoing development. Features are merged here before going to `main`.
    *   `feature/[feature-name]` or `feat/[issue-id]-[short-desc]`: For new features. Branched from `develop`.
    *   `bugfix/[bug-name]` or `fix/[issue-id]-[short-desc]`: For bug fixes. Branched from `develop` (or `main` for hotfixes).
    *   `hotfix/[issue-id]`: For critical production fixes. Branched from `main`, merged back to `main` and `develop`.
*   **Commit Messages:**
    *   Follow conventional commit message format (e.g., `<type>(<scope>): <subject>`).
        *   Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`.
        *   Scope: Optional, e.g., `(component_a)`, `(auth)`.
    *   Subject line: Concise (e.g., <50 chars), imperative mood (e.g., "Add user login" not "Added user login").
    *   Body: Explain *what* and *why* vs. *how*. Reference issue numbers.
*   **Pull Requests (PRs) / Merge Requests (MRs):**
    *   All changes to `develop` and `main` should go through PRs/MRs.
    *   PRs should be focused and address a single feature or bug fix.
    *   Include a clear description of changes and link to relevant issues.
    *   Require at least one approval (or as per team policy).
    *   Ensure all automated checks (CI tests, linting) pass before merging.
    *   Prefer squash merging or rebase merging for a cleaner `develop`/`main` history, if agreed by the team.
*   **Rebasing:** Prefer rebasing feature branches onto `develop` before creating a PR to maintain a linear history and simplify merges. `git pull --rebase` for updating local branches.

**5. Architecture Decision Records (ADRs)**

*   **Purpose:** To document significant architectural decisions, their context, consequences, and alternatives considered.
*   **When to Create an ADR:**
    *   Introducing a new technology, library, or framework.
    *   Making a significant change to system structure or component responsibilities.
    *   Choosing a specific pattern or approach over others for a critical part of the system.
    *   Deprecating a major feature or component.
*   **Format:** Use the template provided in `docs/ARCHITECTURE/adr/ADR_TEMPLATE.md`.
*   **Storage:** Store ADRs in `docs/ARCHITECTURE/adr/`.
*   **Review:** ADRs should be reviewed by the team before being marked as "Accepted."

**6. Documentation Conventions**

*   **Living Document:** Documentation should be treated as a living part of the project and kept up-to-date with code changes.
*   **Audience:** Write for other developers (current and future) who will work on the project.
*   **Location:** All primary project documentation resides in the `docs/` directory.
*   **Updates:** Follow the process outlined in `05_DOCUMENTATION_GUIDE.md` for reviewing and updating documentation after significant changes.
*   **Clarity and Conciseness:** Be clear, concise, and unambiguous. Use diagrams where helpful.

**7. Dependency Management**

*   **File:** Use a standard dependency management file for your language/ecosystem (e.g., `pyproject.toml` with Poetry or PDM, `requirements.txt` for Python; `package.json` for Node.js).
*   **Pinning:** Pin versions of direct dependencies to ensure reproducible builds and avoid unexpected breaking changes from transitive dependencies. Use version ranges (e.g., `^1.2.3`, `~1.2.3`) thoughtfully.
*   **Review:** Regularly review and update dependencies to address security vulnerabilities and leverage new features/fixes.
*   **Minimize Dependencies:** Only add dependencies that provide significant value and are well-maintained.

**8. Code Reviews**

*   **Purpose:** Improve code quality, share knowledge, ensure adherence to standards, and catch bugs early.
*   **Focus Areas:**
    *   Correctness: Does the code do what it's supposed to do?
    *   Clarity & Readability.
    *   Test Coverage & Quality.
    *   Adherence to IDL contract and project rules.
    *   Design: Simplicity, maintainability, potential issues.
    *   Security considerations.
*   **Constructive Feedback:** Provide specific, actionable, and respectful feedback. Explain the "why" behind suggestions.
*   **Timeliness:** Aim to review PRs in a timely manner.

**9. Issue Tracking**

*   Use an issue tracker (e.g., GitHub Issues, Jira) for managing tasks, bugs, and feature requests.
*   Clearly describe issues, including steps to reproduce for bugs.
*   Link commits and PRs to relevant issues.

**10. Continuous Integration/Continuous Deployment (CI/CD) (Recommended)**

*   Set up CI pipelines to automatically:
    *   Run linters and formatters.
    *   Run all tests.
    *   Build the application/library.
    *   (Optional) Deploy to staging/production environments.
*   Ensure CI checks must pass before merging PRs.
