from .settings import NOCOT

edit_action_start = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. The shape is being edited to satisfy the user's edit request.
In this step, you must decide how the $part part of the shape must be edited. A symbolic solver has discovered an edit which minimally breaks one or more of the part relations. You must decide whether this edit should be accepted (as the relations could be updated later) or if the symbolic solver should search more complex edits.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Part/Relation in Focus

$part_in_detail

$unedited_parts

## Minimally relation Breaking Edit

$minimally_relation_breaking_edit

## Output Specification

Specify a string variable `action` in python which specifies whether or not the minimally relation breaking edit should be accepted or not.

```python
action = "search_more_complex" # or "accept"
```

### Steps for inferring the action

Follow these instructions to infer the most appropriate action:

1. **Follow the Edit Request**: For the $part part, if the type of the solved edit matches the type of edit requested by the user, then accept the edit.

2. **Consider Accepting the Edit**: Carefully consider the relations that are being broken. Can they be easily updated after the edit is performed? If so, the action should be "accept".

3. **Consider Search**: Consider what will happen if we search more complex edits. Here are the edit types in increasing order of complexity: `translate`, `rotate`, `scale`, `shear`. Would a higher complexity edit be better for this part? If so the action should be "search_more_complex".
"""

variable_list = [
    "API",
    "shape_specification",
    "edit_request",
    "shape_class"
]
edit_type_guideline_list = [
    "**Analytical Approach**: Carefully follow the steps above explaining the logic behind your choices. Perform this task step by step.",
    "**Code Format**: Specify the `action` in a single Python code snippet, strictly adhering to the specified format.",
    "**Simplicity**: Generally, simpler edits are preferrable over more complex edits.",
]
edit_action_end = """Now, please specify the `action` variable by following the instructions above."""

def base_instructions(nocot=NOCOT):
    if nocot:
        selected_guidelines = edit_type_guideline_list[1:]
    else:
        selected_guidelines = edit_type_guideline_list
    guideline_str = "### Guidelines\n\n"
    for ind, guideline in enumerate(selected_guidelines):
        guideline_str += f"{ind + 1}. {guideline}\n\n"
    instructions = "\n\n".join([edit_action_start, guideline_str, edit_action_end])
    return instructions

variables = [
    "shape_class",
    "shape_specification",
    "edit_request",
    "part_in_detail",
    "unedited_parts",
    "part",
    "minimally_relation_breaking_edit",

]