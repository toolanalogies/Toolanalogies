from .settings import NOCOT
edit_spec_start = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. The shape is being edited to satisfy the user's edit request. In this step, you must identify the type edit which must be performed on the $part of the shape. This type hint is then used by a solver to generate a valid edit.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Part/Relation in Focus

$part_in_detail

$unedited_parts

## Output Specification

Specify a string variable `edit_type` in python which specifies the type of edit that must be performed on the part.

```python
edit_type = "type_hint" # must be one of the type_hint_options
```

The type hints must be one of the following:

```python
type_hint_options = $remaining_type_hints

```

`change_count` and `change_delta` are to be used only for parts with translation or rotation symmetry. When editing the object, `change_count` specifies that the number of instances in a symmetry group must be changed, and `change_delta` specifies that the distance between the instances should be changed.

### Steps for inferring the Edit Type

Follow these instructions to infer the most appropriate edit type for the $part part:

1. **Follow the Edit Request**: If the edit request explicitly specifies a desired edit type, then strictly adhere to it.

2. **Analyze the Part**: When the edit request does not specify an edit type, analyze the part to determine the most appropriate edit type. There are two primary considerations: a) As the object is being edited to fulfill the user's request, how should the part be modified? b) given the relations that the part has with other parts, what edit type will keep the relations intact?

3. **Simplicity**: Select the simplest edit type that can satisfy the user's request. Here are the edit types arranged in increasing complexity left to right: `translate`, `rotate`, `scale`, `shear`, `change_count`, `change_delta`."""

edit_spec_with_failure = """#Overview"""

variable_list = [
    "API",
    "shape_specification",
    "edit_request",
    "shape_class"
]
edit_type_guideline_list = [
    "**Analytical Approach**: Carefully follow the steps above explaining the logic behind your choices. Perform this task step by step.",
    "**Code Format**: Specify the `edit_type` in a single Python code snippet, strictly adhering to the specified format.",
    "**Valid Type**: The edit type must be one of the type_hint_options given above.",
    "**Always Specify**: You must specify a edit type. If none of the types seem appropriate, select the one which is least inappropriate.",
]
edit_spec_end = """Now, please specify the `edit_type` by following the instructions above.\n"""

def base_instructions(nocot=NOCOT):
    if nocot:
        selected_guidelines = edit_type_guideline_list[1:]
    else:
        selected_guidelines = edit_type_guideline_list
    guideline_str = "### Guidelines\n\n"
    for ind, guideline in enumerate(selected_guidelines):
        guideline_str += f"{ind + 1}. {guideline}\n\n"
    instructions = "\n\n".join([edit_spec_start, guideline_str, edit_spec_end])
    return instructions

def with_failure_instructions(nocot=NOCOT):
    if nocot:
        selected_guidelines = edit_type_guideline_list[1:]
    else:
        selected_guidelines = edit_type_guideline_list
    guideline_str = "### Guidelines\n\n"
    for ind, guideline in enumerate(selected_guidelines):
        guideline_str += f"{ind + 1}. {guideline}\n\n"
    instructions = "\n\n".join([edit_spec_start, guideline_str, edit_spec_end])
    return instructions

variables = [
    "shape_class",
    "shape_specification",
    "edit_request",
    "part_in_detail",
    "unedited_parts",
    "remaining_type_hints"
    "part"

]