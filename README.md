# Generic IDL-Driven Python Project Template

## Overview

This repository provides a generic template and a set of best-practice guidelines for developing Python projects using an **Interface Definition Language (IDL) driven approach**. The goal is to foster clear specifications, consistent implementation patterns, robust testing, and maintainable codebases.

This template is designed to be adaptable for various Python projects, particularly those that benefit from:
*   Clear contracts between components.
*   Separation of concerns between interface specification and implementation.
*   A structured development workflow.
*   Emphasis on testability and maintainability.

## Core Philosophy

The development process promoted by this template is centered around **IDL specifications** as the primary specification for components.

*   **The IDL is the Contract:** Each IDL file defines the strict contract for a component, specifying its public interface, behavior, dependencies, and error conditions.
*   **Specification, Not Implementation:** IDLs focus on *what* a component does, separating this from *how* it's implemented.
*   **Source of Truth:** IDLs are the authoritative source for component requirements.

Refer to `docs/01_IDL_GUIDELINES.md` for detailed information on creating and interpreting IDLs.

## What's Included?

This template provides:

*   **A Recommended Directory Structure:** For organizing your source code, tests, and documentation (see `docs/03_PROJECT_RULES.md`).
*   **Core Documentation Templates:**
    *   `docs/00_START_HERE.md`: Developer onboarding guide.
    *   `docs/01_IDL_GUIDELINES.md`: Rules for writing IDL specifications.
    *   `docs/02_IMPLEMENTATION_RULES.md`: Coding standards and implementation patterns.
    *   `docs/03_PROJECT_RULES.md`: General project conventions (directory structure, version control).
    *   `docs/04_REFACTORING_GUIDE.md`: Guidelines for refactoring code.
    *   `docs/05_DOCUMENTATION_GUIDE.md`: Process for keeping documentation up-to-date.
*   **Workflow & Utility Templates:**
    *   `docs/TEMPLATES/TASK_INSTRUCTION_TEMPLATE.md`: For creating detailed task instructions.
    *   `docs/TEMPLATES/IDL_IMPLEMENTATION_CHECKLIST.md`: For ensuring IDL readiness.
    *   `docs/TEMPLATES/WORKING_MEMORY_LOG_TEMPLATE.md`: For developers to track their work.
    *   `docs/WORKFLOWS/TASK_PREPARATION_WORKFLOW.md`: For tech leads preparing tasks.
*   **Architecture Documentation Structure:**
    *   `docs/ARCHITECTURE/types.md`: Template for defining shared project data types.
    *   `docs/ARCHITECTURE/adr/`: Directory for Architecture Decision Records, including a `README.md` and `ADR_TEMPLATE.md`.
*   **Library Integration Guidance:**
    *   `docs/LIBRARY_INTEGRATION/README.md`: Explains how to document key third-party library usage.

## How to Use This Template

1.  **Clone or Copy:** Clone this repository or copy its contents to start your new project.
2.  **Customize Placeholders:**
    *   Search for `[YourProjectName]` and replace it with your actual project name throughout the documentation.
    *   Review and customize other placeholders like `[Specify target Python version]`, `[Formatter, e.g., Black]`, `[Testing Framework, e.g., pytest]`, `[YourSystemComponentType]`, etc., in the `docs/` files to match your project's specifics.
    *   Update the example directory structure in `docs/03_PROJECT_RULES.md` if needed.
3.  **Adapt Architectural Concerns:** The "Architectural Alignment" section in `docs/TEMPLATES/IDL_IMPLEMENTATION_CHECKLIST.md` and examples in `docs/ARCHITECTURE/types.md` are highly generic. Adapt these to reflect your project's specific architectural principles and common data patterns.
4.  **Review and Adopt Guidelines:** Read through all the `.md` files in the `docs/` directory. Understand the proposed workflows, rules, and conventions. Adapt them as necessary to fit your team's needs, but try to maintain the core principles of IDL-driven development.
5.  **Set Up Your Project:**
    *   Initialize your version control system (e.g., `git init`).
    *   Set up your Python environment (e.g., virtual environment, `pyproject.toml` or `requirements.txt`).
    *   Configure linters, formatters, and testing tools.
    *   Consider adding a `Makefile` or `justfile` for common development tasks.
6.  **Start Defining Interfaces:** Begin by creating IDL files for your core components as per `docs/01_IDL_GUIDELINES.md`.
7.  **Follow the Workflow:** Use the development workflow outlined in `docs/00_START_HERE.md` and the task preparation workflow in `docs/WORKFLOWS/TASK_PREPARATION_WORKFLOW.md`.

## Key Principles to Emphasize

*   **Interface First:** Define clear contracts (IDLs) before extensive implementation.
*   **Test the Contract:** Ensure your tests verify adherence to the IDL.
*   **Consistency:** Follow the established rules and conventions for code, documentation, and processes.
*   **Documentation as Code:** Treat documentation as an integral part of the development process, keeping it co-located and up-to-date.

## Contributing to This Template (Meta)

If you find ways to improve this generic template itself, please consider contributing back or forking it for your own organizational needs. The goal is to provide a solid foundation that can be broadly useful.

---

Happy Coding!
