instructions = """# Overview

You are a 3D shape editing expert. You are given a 3D shape composed of parts with relations between them, and a user's edit request in natural language. You must identify the most important or primary part-level edit that should be performed to fulfill the edit request. 

## Edit Specification

Specify the `primary_edit` variable in python using the API specified below. The `primary_edit` should be specified in a code snippet in the following format:

```python
primary_edit = ... # must be created via the API
```

The `primary_edit` variable must be created using the following API:

$API

## Shape Specification

Your input is a $shape_class. The 3D shape consists of:

* `parts`: A list of semantically meaningful parts, such as 'front_right_legs', 'back', 'seat', etc.

```python
$shape_specification
```

## User's Edit Request

$edit_request

### Steps for Identifying the Primary Edit

1. Based on the edit request, identify the primary part to edit.
2. For the identified part, identify edit classes from the API which may fulfill the edit request. Consider the effect of editing the identified part with each candidate class. Based on this analysis, select the edit class that is the most appropriate.
3. Now, for the selected edit class, consider each of its attributes one-by-one, and articulate what they should be, and why. Then, use functions such as `get_center`, `get_face_center`, `RightUnitVector` etc. to define these parameters.

### Guidelines

1. **Analytical Approach**: Provide a step-by-step rationale for the primary edit, explaining the logic behind your choices.
2. **Code Format**: Specify the primary edit in a executable Python code snippet, strictly adhering to the specified API format. Remember to specify only a single primary edit with the `primary_edit` variable.
3. **Function Use**: Employ functions like `get_center` and `get_face_center` to define the primary edit more precisely.
4. **Specifying `amount`**: Specify the `amount` parameter of the edit only if its required. When specified, express `amount` as a sympy expression using "X" as the only free variable. When scaling objects, use negative amount to indicate shrinking/shortening/thinning.
5. **Translation vs Scaling**: Translation edit cannot be used to scale objects. To scale a part you must use one of the scaling edits.
6. **Scaling / Rotating Parts** When scaling or rotating parts, always carefully consider the value of the `origin` parameter. For example, while scaling legs, scaling with the origin set to the center of bottom face is more intuitive.
7. **Global Directions**: Remember that all the directional variables "front", "back", "left", "right", "up", "down" are defined with respect to the global coordinate system. Therefore, the bottom face of the seat faces the 'down' direction, and the top face of the seat faces the 'up' direction.

Now, please specify a single primary edit by following the instructions above."""

variable_list = [
    "API",
    "shape_specification",
    "edit_request",
    "shape_class"
]