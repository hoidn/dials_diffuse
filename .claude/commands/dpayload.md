prepare a self contained 'debug payload' file containing:

- description of the issue
- all context information / files needed to understand the issue and relevant parts of the codebase
- all files that might be the source of the bug and / or relevant to debugging 
- if appropriate, a description of which debugging approaches we already tried, and why it didn't work

Instead of including the literal contents of each file that you decided to include, do the following for each file:

if only some portions of the file are relevant to the task, literally quote those sections and replace the removed 
sections by placeholders saying that they've been removed. 

If the whole file is relevant, don't literally include it. instead specify it with 
jinja style template syntax:

{rel/path/to/file.py}
{another/file.md}

Don't include data files with binary sections (such as cbf).


Remember to include api documentation context (lives under libdocs/dials/), but ONLY in the format of excerpted relevant sections, not jinja entries (this is to reduce size, since some documentation files are large). Use subagents to parse such big files.

Include these files:
./phase1_demo_output/validation_failure_report.txt
./phase1_demo_output/validation_report.txt
