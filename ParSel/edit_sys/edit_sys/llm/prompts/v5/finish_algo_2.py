instructions = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of parts with relations between them, and a user's edit request in natural language. You are given an incomplete list of part-level edits. Your goal is to define a one more new edit to help fulfill the user's edit request.

## Shape Specification

Your input is a 3D $shape_class. It consists:

1. A list of semantically meaningful parts, such as 'front_right_legs', 'back', 'seat', etc. Each part is an independent 3D Mesh.
2. A set of relations identified between the shape parts, such as symmetry groups and contact pairs **before** the edit takes place. Note that, some of these relations may be rejected (i.e. they are not enforced) in later steps if they conflict with the user's edit request.

$shape_specification

## User's Edit Request

$edit_request

## Edit API

The edits are defined using the following API:

$API

## Incomplete List of Edits

The following part-level edits have been defined already:

```python
$all_edits
```

$remaining_parts

## Output specification

Specify the `new_edit` variable in python using the API defined above. The `new_edit` should be specified in a code snippet in the following format:

```python
new_edit = ... # must be created via the API
```

### Steps for Determining New Edit

1. Carefully analyze the unedited parts. Did the edit request explicitly require any of them to be edited? Parts which should have been edited but are not edited are candidate parts for the new edit. Select one of these parts to edit.

2. For the identified part, identify edit classes from the API which may fulfill the edit request. Consider the effect of editing the identified part with each candidate class. Based on this analysis, select the edit class that is the most appropriate.

3. Now, for the selected edit class, consider each of its attributes one-by-one, and articulate what they should be, and why. Then, use functions such as `get_center`, `get_face_center`, `RightUnitVector` etc. to define these parameters.

### Guidelines

1. **Analytical Approach**: Carefully follow the steps above and explain the logic behind your choices.

2. **Code Format**: Specify the primary edit in a single Python code snippet, strictly adhering to the specified API format. Remember to specify only a single primary edit with the `primary_edit` variable.

3. **Function Use**: Employ functions like `get_center` and `get_face_center` to define the primary edit more precisely.

4. **Specifying `amount`**: Specify the `amount` parameter of the edit only if its required. When specified, express `amount` as a sympy expression using "X" as the only free variable. When scaling objects, use negative amount to indicate shrinking/shortening/thinning.

5. **Translation vs Scaling**: Translation edit cannot be used to scale objects. To scale a part you must use one of the scaling edits.

6. **Scaling / Rotating Parts** When scaling or rotating parts, always carefully consider the value of the `origin` parameter. For example, while scaling legs, scaling with the origin set to the center of bottom face is more intuitive.

7. Remember that all the directional variables "front", "back", "left", "right", "up", "down" are defined with respect to the global coordinate system. Therefore, the bottom face of the seat faces the 'down' direction, and the top face of the seat faces the 'up' direction.

Now, please specify a single new edit by following the instructions above.
"""