prepare a self contained 'debug payload' file containing:

- description of the issue
- all context information / files needed to understand the issue and relevant parts of the codebase
- all files that might be the source of the bug and / or relevant to debugging 

Instead of including the literal contents of each file that you decided to include, specify them with 
jinja style template syntax:

{rel/path/to/file.py}
{another/file.md}

