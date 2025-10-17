from .settings import NOCOT
edit_spec_start = """#Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. The shape is being edited to satisfy the user's edit request. In this step, you must specity the edit which must be performed on the $part part of the shape. The specified edit is them adjusted by a symbolic solver to satisfy the constraints on the program.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Part/Relation in Focus

$part_in_detail

$unedited_parts

## Output Specification

Specify the `edit` variable in python using the API specified below. The `edit` should be specified in a code snippet in the following format:

```python
edit = ... # must be created via the API
```

The `primary_edit` variable must be created using the following API:

$API

### Specifying the Edit

Follow these instructions to infer the most appropriate edit for the $part part:

1. **Follow the Edit Request**: If the edit request explicitly specifies a desired edit type for $part, then strictly adhere to it. 

2. **Identify Edit classes** Otherwise, first identify edit class(es) from the API which are appropriate for the edit request. Carefully consider edits on the parts with relation to the $part. Often similar edits are suitable.

2. **Identify Scope of Edit**: Next, Identify if the part should be edited entirely or only a specific face or edge of the part should be edited.

4. If there are multiple edit classes which are feasible, first consider the effect of editing $part with each candidate class. Then, based on this analysis, select the edit class which is the most appropriate.

5. Now, for the selected edit class, consider each of its attributes one-by-one, and specify what they should be, and why. Then, use functions such as `center`, `face_center`, `edge_center`, `direction`, `RightUnitVector` etc. to define these parameters. Note that when necessary, other edited parts can also be used to specify these parameters."""

edit_spec_with_failure = """#Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. The shape is being edited to satisfy the user's edit request. In this step, you must specity the edit which must be performed on the $part part of the shape. The specified edit is them adjusted by a symbolic solver to satisfy the constraints on the program.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Part/Relation in Focus

$part_in_detail

$unedited_parts

## Output Specification

Specify the `edit` variable in python using the API specified below. The `edit` should be specified in a code snippet in the following format:

```python
edit = ... # must be created via the API
```

The `primary_edit` variable must be created using the following API:

$API

## Failed Edit(s)

The follow edit was tried before and it failed:

$failed_edit

### Specifying the Edit

Follow these instructions to infer the most appropriate edit for the $part part:

1. **Follow the Edit Request**: If the edit request explicitly specifies a desired edit type for $part, then strictly adhere to it. 

2. **Identify Edit classes** Otherwise, first identify edit class(es) from the API which are appropriate for the edit request. Carefully consider edits on the parts with relation to the $part. Often similar edits are suitable.

2. **Identify Scope of Edit**: Next, Identify if the part should be edited entirely or only a specific face or edge of the part should be edited.

4. If there are multiple edit classes which are feasible, first consider the effect of editing $part with each candidate class. Then, based on this analysis, select the edit class which is the most appropriate.

5. Now, for the selected edit class, consider each of its attributes one-by-one, and specify what they should be, and why. Then, use functions such as `center`, `face_center`, `edge_center`, `direction`, `RightUnitVector` etc. to define these parameters. Note that when necessary, other edited parts can also be used to specify these parameters."""

edit_spec_guideline_list = [
    "**Analytical Approach**: Carefully follow the steps above explaining the logic behind your choices. Perform this task step by step.",
    "**Code Format**: Specify the `edit` variable in a single Python code snippet, strictly adhering to the specified format.",
    """**Function Use**: Employ functions like `center` and `face_center` to define the primary edit precisely. Do not use sympy, numpy etc. to create these variables. The amount of the edit will be adjusted by a solver and must not be specified.""",
    """**Part Selection**: Use the `shape.get(label)` function to access the part with label `label`, where the label is the path from the root part to the part to be accessed (e.g., `left_front_leg = shape.get("legs/leg_front_left")`). Do not use shape.get("$shape_class"). Use one of the parts from the shape specification.""",
    """**Cuboidal Parts**: Remember that all the parts are simple cuboids.""",
    """**Scaling / Rotating Parts** When scaling or rotating parts, always carefully consider the value of the `origin` parameter. When considering rotation, consider if it results in a unstable shape. If so, consider using face translation instead (unless the edit request explicitly specifies rotation).""",
    """**Natural Language Directions**: Remember that all the directional variables "front", "back", "left", "right", "up", "down" are defined with respect to the global coordinate system. Therefore, the bottom face of the seat approximately faces the 'down' direction and the top face of the seat faces the 'up' direction approximately. Futhermore, remember that 'width' corresponds to the 'left-right' direction, 'height' corresponds to the 'up-down' direction, and 'depth' corresponds to the 'front-back' direction."""
]
edit_spec_failure_guideline_list = [
    """**Increase Complexity**: Remember to specify an edit more complex than the failed edit.""",

]
edit_spec_end = """Now, please specify the `edit` by following the instructions above."""

def base_instructions(nocot=NOCOT):
    if nocot:
        selected_guidelines = edit_spec_guideline_list[1:]
    else:
        selected_guidelines = edit_spec_guideline_list
    guideline_str = "### Guidelines\n\n"
    for ind, guideline in enumerate(selected_guidelines):
        guideline_str += f"{ind + 1}. {guideline}\n\n"
    instructions = "\n\n".join([edit_spec_start, guideline_str, edit_spec_end])
    return instructions

def with_failure_instructions(nocot=NOCOT):
    if nocot:
        selected_guidelines = edit_spec_guideline_list[1:] + edit_spec_failure_guideline_list
    else:
        selected_guidelines = edit_spec_guideline_list + edit_spec_failure_guideline_list
    guideline_str = "### Guidelines\n\n"
    for ind, guideline in enumerate(selected_guidelines):
        guideline_str += f"{ind + 1}. {guideline}\n\n"
    instructions = "\n\n".join([edit_spec_with_failure, guideline_str, edit_spec_end])
    return instructions

variables = [
    "shape_class",
    "shape_specification",
    "edit_request",
    "part_in_detail",
    "unedited_parts",
    "remaining_type_hints",
    "part",
    "API"
    "failed_edit"

]