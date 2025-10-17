instructions = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. You must return a list of part-level edit operations that should be performed to fulfill the edit request.

## Shape Specification

Your input is a $shape_class. The part-hierarchy along with the specification of the part-cuboid of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Output Specification

Specify the a list of edits in a variable named `program` in python using the API specified below. The `program` variable should be specified in a code snippet in the following format:

```python
edit_1 = ... # must be created via the API
edit_2 = ... # must be created via the API
program = [edit_1, edit_2, ...]
```

The edits variable must be created using the following API:

$API
### Specifying the Primary Edit

1. Based on the edit request, identify the parts that should be edited. 

2. For each part, determine if the part should be entirely edited or only a specific face or edge of the part should be edited.

3. Now, for each part, identify the applicable edit class from the API which adhere to the edit request. 

4. If there are multiple edit classes which adhere to the edit request, first consider the effect of editing the identified part with each candidate class. Then, based on this analysis, select the edit class that is the most appropriate.

5. Now, for each part and its selected edit class, consider each of its attributes one-by-one, and specify what they should be, and why. Then, use functions such as `center`, `face_center`, `edge_center`, `direction`, `RightUnitVector` etc. to define these parameters.

6. Finally, consider the amount variable for each edit carefully. The amount variable should be defined based on the part cuboid specifications. Using all the determined variables define the edits, and create the `program` variable.

### Guidelines

1. **Analytical Approach**: Provide a step-by-step rationale for the edits, explaining the logic behind your choices.

2. **Code Format**: Specify the edits in a executable Python code snippet, strictly adhering to the specified API format. Provide the code inside code snippets (starting with "```python" and ending with "```"). Remember to specify the `program` variable at the end.

3. **No Imports**: Do not use any import statements in your code. The functions and classes required to create the primary edit are already available.

4. **Function Use**: Employ functions like `center` and `face_center` to define the edits precisely. Use only expressions containing sympy.Symbol("X") to define the amount attribute. "X" will then be specified by the user interface at the run time. Specify the amount variable based on part cuboid specifications. Further, when using the edit function use named arguments.

5. **Part Selection**: Use the `shape.get(label)` function to access the part with label `label`, where the label is the path from the root part to the part to be accessed (e.g., `left_front_leg = shape.get("legs/leg_front_left")`). Do not use shape.get("$shape_class").

6. **Cuboidal Parts**: Remember that all the parts are simple cuboids.

7. **Scaling / Rotating Parts** When scaling or rotating parts, always carefully consider the value of the `origin` parameter. When considering rotation, consider if it results in a unstable shape. If so, consider using face translation instead (unless the edit request explicitly specifies rotation).

8. **Natural Language Directions**: Remember that all the directional variables "front", "back", "left", "right", "up", "down" are defined with respect to the global coordinate system. Therefore, the bottom face of the seat approximately faces the 'down' direction and the top face of the seat faces the 'up' direction approximately. Futhermore, remember that 'width' corresponds to the 'left-right' direction, 'height' corresponds to the 'up-down' direction, and 'depth' corresponds to the 'front-back' direction.

Now, please specify the `program` variable by following the instructions above."""

variable_list = [
    "API",
    "shape_specification",
    "edit_request",
    "shape_class"
]
