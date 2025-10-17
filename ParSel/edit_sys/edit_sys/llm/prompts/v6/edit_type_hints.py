instructions = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. You must identify the type of part-level edits which are explicitly requested in the edit request. These type hints are then used by a solver to generate a valid edit.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Output Specification

Specify a dictionary `edit_type_hints` in python which specifies a mapping from a part's full name to the type restriction explicitly mentioned in the edit request.


```python
edit_type_hints = {
    "part_full_name" : "type_hint" # must be one of the type_hint_options
    ...
}
```

 The type hints must be one of the following:

```python
type_hint_options = ['translate', 'rotate', 'scale', 'shear', 'change_count', 'change_delta']

```

`change_count` and `change_delta` are to be used only for parts with translation or rotation symmetry. When editing the object, `change_count` specifies that the number of instances in a symmetry group must be changed, and `change_delta` specifies that the distance between the instances should be changed. 


### Steps for Identifying the type hints

1. **Identify Editable Parts**: First, determine which parts need to be edited based on the edit request. These parts are candidates for the edit type hints.

2. **Assess Explicit Restrictions**: Next, determine if any of these parts have a specific type of edit requested. If so, add the part to the `edit_type_hints` dictionary with the appropriate type hint.

### Guidelines

1. **Analytical Approach**: Follow the above instructions step-by-step and explaining your solution along the way.

2. **Code Format**: Specify the `edit_type_hints` dictionary in a single Python code snippet, strictly adhering to the instructions above. 

3. **Be Conservative**: Add parts to the dict only if the edit request explicitly states some hints. Do add hints for parts which are not specified in the edit request. These parts may need to be adjusted freely to fulfill the edit request. If there are not edit type hints, let `edit_type_hints` be an empty dictionary.

4. **Carefully Evaluate Hints**: Please pay attention to the type of hint specified in the edit request while adding parts to the dictionary. 

5. **Part Full Names**: A part's full name is its path from the root of the shape. For e.g., if the leg_front_left is under "legs", then full name is "legs/leg_front_left".

Now, please specify the `edit_type_hints` by following the instructions above."""

variable_list = [
    "API",
    "shape_specification",
    "edit_request",
    "shape_class"
]
