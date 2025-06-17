the following project files are 'critical' and should always be included as context, regardless of the specific task. They provide the fundamental "rules of the road," project goals, and architectural principles.

These files can be categorized into three groups: **Core Plans**, **Project Rules & Conventions**, and **Essential Guides**.

### 1. Core Plans & High-Level Strategy

These files define the "what" and "why" of the project. An agent cannot understand the purpose of its task without them.

*   `plan.md`: **The Master Plan.** This is the highest-authority technical specification for the entire processing pipeline. It is the single most critical document for understanding any task's place in the larger system.
*   `plan_adaptation.md`: **Critical Modifications to the Master Plan.** This document details the crucial pivot to support both stills and sequence data, a fundamental architectural decision that impacts how the pipeline is orchestrated. It's an essential addendum to `plan.md`.
*   `CLAUDE.md`: **Project-Specific AI Guidance & High-Level Log.** This file appears to be a working log of high-level decisions, goals, and technical achievements. For an AI agent, this provides invaluable context on recent progress, proven strategies, and the project's "state of mind."
*   `README.md`: **The Project Entry Point.** Provides the highest-level overview of the project's purpose and features.

### 2. Project Rules & Conventions

These files define the "how" of the project. They ensure that any work done by the agent is consistent with the established development process, coding standards, and architectural patterns.

*   `docs/00_START_HERE.md`: **Developer Orientation.** The primary onboarding document that explains the project's core philosophy, including the critical role of IDL files as specifications.
*   `docs/01_IDL_GUIDELINES.md`: **The "Language" of the Project.** This defines the syntax and semantics of the Interface Definition Language (IDL) used to specify every component. Without this, an agent cannot correctly interpret or create component contracts.
*   `docs/02_IMPLEMENTATION_RULES.md`: **Coding and Testing Standards.** This dictates how IDLs are translated into code, how testing should be approached (emphasizing integration over unit tests), and other critical implementation patterns.
*   `docs/03_PROJECT_RULES.md`: **Project Organization.** Defines the directory structure, version control workflow, and conventions for file organization. Essential for any task that involves creating or moving files.
*   `src/diffusepipe/types/types_IDL.md` (and `docs/ARCHITECTURE/types.md`): **Shared Data Contracts.** Defines the core data structures (`OperationOutcome`, configuration objects, etc.) that are used across the entire pipeline. Understanding these is fundamental to any component interaction.

### 3. Essential Guides & Distilled Knowledge

These documents contain "hard-won" knowledge and troubleshooting information that can prevent common errors and significantly accelerate development.

*   `docs/LESSONS_LEARNED.md`: **The "Wisdom" of the Project.** This file consolidates critical insights from past development challenges, such as the "Test Failure Crisis" and "Safe Refactoring" incident. It's a high-leverage document that helps avoid repeating past mistakes.
*   `docs/06_DIALS_DEBUGGING_GUIDE.md`: **The DIALS Bible.** DIALS is the most critical and complex external dependency. This guide is essential for debugging any task involving DIALS processing, which is a core part of the pipeline.
*   `docs/VISUAL_DIAGNOSTICS_GUIDE.md`: **Validation and Verification Guide.** This documents the tools used to visually verify the correctness of the pipeline's outputs. It's crucial for any task involving pipeline development or debugging.
