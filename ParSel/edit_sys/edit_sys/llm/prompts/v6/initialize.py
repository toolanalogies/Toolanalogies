instructions_primary_edit = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. You must identify the primary (first) part level edit that should be performed to fulfill the edit request. A edit propagation system will generate a complete edit based on this primary edit.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Output Specification

Specify the `primary_edit` variable in python using the API specified below. The `primary_edit` should be specified in a code snippet in the following format:

```python
primary_edit = ... # must be created via the API
```

The `primary_edit` variable must be created using the following API:

$API
### Specifying the Primary Edit

1. Based on the edit request, identify the primary part to edit. 

2. Identify if the part entirely should be edited or only a specific face or edge of the part should be edited.

3. For the identified part, identify edit class(es) from the API which adhere to the edit request. 

4. If there are multiple edit classes which adhere to the edit request, first consider the effect of editing the identified part with each candidate class. Then, based on this analysis, select the edit class that is the most appropriate.

5. Now, for the selected edit class, consider each of its attributes one-by-one, and specify what they should be, and why. Then, use functions such as `center`, `face_center`, `edge_center`, `direction`, `RightUnitVector` etc. to define these parameters.

### Guidelines

1. **Analytical Approach**: Provide a step-by-step rationale for the primary edit, explaining the logic behind your choices.

2. **Code Format**: Specify the primary edit in a executable Python code snippet, strictly adhering to the specified API format. Remember to specify only a single primary edit with the `primary_edit` variable.

3. **Function Use**: Employ functions like `center` and `face_center` to define the primary edit precisely. Do not use sympy, numpy etc. to create these variables. The amount of the edit will be controlled by the user interface and must not be specified.

4. **Part Selection**: Use the `shape.get(label)` function to access the part with label `label`, where the label is the path from the root part to the part to be accessed (e.g., `left_front_leg = shape.get("legs/leg_front_left")`). Do not use shape.get("$shape_class"). Use one of the parts from the shape specification.

5. **Cuboidal Parts**: Remember that all the parts are simple cuboids.

6. **Scaling / Rotating Parts** When scaling or rotating parts, always carefully consider the value of the `origin` parameter. When considering rotation, consider if it results in a unstable shape. If so, consider using face translation instead (unless the edit request explicitly specifies rotation).

7. **Natural Language Directions**: Remember that all the directional variables "front", "back", "left", "right", "up", "down" are defined with respect to the global coordinate system. Therefore, the bottom face of the seat approximately faces the 'down' direction and the top face of the seat faces the 'up' direction approximately. Futhermore, remember that 'width' corresponds to the 'left-right' direction, 'height' corresponds to the 'up-down' direction, and 'depth' corresponds to the 'front-back' direction.

Now, please specify a single primary edit by following the instructions above."""

variable_list = [
    "API",
    "shape_specification",
    "edit_request",
    "shape_class"
]

instructions_keep_fixed = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. You must identify which sub-parts of the shape are specified to be kept fixed.

## Shape Specification


Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Output Specification

Specify the following four lists containing the labels of sub-parts in python. The lists should be specified in a python code snippet in the following format:

```python
location_fixed_parts = ["", ...] # must contain full name of sub-parts.
scale_fixed_parts = ["", ...] # must contain full name of sub-parts.
rotation_fixed_parts = ["", ...] # must contain full name of sub-parts.
```

Specify an empty list if a particular list has no entries.

### Steps for Identifying the Fixed Parts

1. **Assess Explicit Restrictions**: Determine what parts are explicitly requested to be kept fixed in the edit request. For each explicitly mentioned part, analyse if the restriction is in location, scale, rotation, or all three.

2. **Select Appropriate Lists**: For the parts that that must be kept fixed, determine which list they should be in. If and only if a part must remain completely unaltered, add it to all the three lists.

### Guidelines

1. **Analytical Approach**: Follow the above instructions step-by-step and explaining your solution along the way.

2. **Code Format**: Specify all the lists in a single Python code snippet, strictly adhering to the instructions above. Ensure each entry in the different lists corresponds to a specific part's full name from the shape specification.

3. **Be Conservative**: Add parts to only the appropriate lists. Don't add parts to all the lists unless the edit request explicitly specifies that the part should remain fixed, unchanged, constant, or unaltered.

4. **Do not fix unspecified parts**: Do not fix parts which are not specified in the edit request. These parts may need to be adjusted to fulfill the edit request.

5. **Transformation Restrictions**: Please pay attention to the type of restriction specified in the edit request while adding parts to the different lists.

6. **Part Full Names**: A part's full name is its path from the root of the shape. For e.g., if the leg_front_left is under "legs", then full name is "legs/leg_front_left".

7. **Using Part Hierarchy**: When all sub-parts of a part are to be added to a list, simply add the parent part to the list. For example, if all the parts under "legs" are to be , then add "legs" to the list instead of adding each part under "legs" individually. Please only use the hierarchies present in the shape specification.

Now, please specify the three lists by following the instructions above."""

instructions_keep_fixed_extra = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. You must identify which sub-parts of the shape are specified to be kept fixed.

## Shape Specification


Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Output Specification

Specify the following four lists containing the labels of sub-parts in python. The lists should be specified in a python code snippet in the following format:

```python
location_fixed_parts = ["", ...] # must contain full name of sub-parts.
scale_fixed_parts = ["", ...] # must contain full name of sub-parts.
rotation_fixed_parts = ["", ...] # must contain full name of sub-parts.
```

Specify an empty list if a particular list has no entries.

### Steps for Identifying the Fixed Parts

1. **Assess Explicit Restrictions**: Determine what parts are explicitly requested to be kept fixed in the edit request. For each explicitly mentioned part, analyse if the restriction is in location, scale, rotation, or all three.

2. **Analyse Implicit Restrictions**: If the user wants to scale the entire object by moving certain other parts, then the parts which are opposite to these moved parts should be kept fixed. Consider what kind of restrictions are implied by the edit request.

3. **Select Appropriate Lists**: For the parts that that must be kept fixed, determine which list they should be in. If and only if a part must remain completely unaltered, add it to all the three lists.

### Guidelines

1. **Analytical Approach**: Follow the above instructions step-by-step and explaining your solution along the way.

2. **Code Format**: Specify all the lists in a single Python code snippet, strictly adhering to the instructions above. Ensure each entry in the different lists corresponds to a specific part's full name from the shape specification.

3. **Be Conservative**: Add parts to only the appropriate lists. Don't add parts to all the lists unless the edit request explicitly specifies that the part should remain fixed, unchanged, constant, or unaltered.

4. **Do not fix unspecified parts**: Do not fix parts which are not specified in the edit request. These parts may need to be adjusted to fulfill the edit request.

5. **Transformation Restrictions**: Please pay attention to the type of restriction specified in the edit request while adding parts to the different lists.

6. **Part Full Names**: A part's full name is its path from the root of the shape. For e.g., if the leg_front_left is under "legs", then full name is "legs/leg_front_left".

7. **Using Part Hierarchy**: When all sub-parts of a part are to be added to a list, simply add the parent part to the list. For example, if all the parts under "legs" are to be , then add "legs" to the list instead of adding each part under "legs" individually. Please only use the hierarchies present in the shape specification.

Now, please specify the three lists by following the instructions above."""


instructions_primary_edit_nocot = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. You must identify the primary (first) part level edit that should be performed to fulfill the edit request.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Output Specification

Specify the `primary_edit` variable in python using the API specified below. The `primary_edit` should be specified in a code snippet in the following format:

```python
primary_edit = ... # must be created via the API
```

The `primary_edit` variable must be created using the following API:

$API

Now, please specify a single primary edit by following the instructions above."""


instructions_keep_fixed_nocot = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts, and a user's edit request in natural language. You must identify which sub-parts of the shape are specified to be kept fixed.

## Shape Specification


Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Output Specification

Specify the following four lists containing the labels of sub-parts in python. The lists should be specified in a python code snippet in the following format:

```python
location_fixed_parts = ["", ...] # must contain full name of sub-parts.
scale_fixed_parts = ["", ...] # must contain full name of sub-parts.
rotation_fixed_parts = ["", ...] # must contain full name of sub-parts.
```

Specify an empty list if a particular list has no entries.

Now, please specify the three lists by following the instructions above."""

