instructions = """# Overview

You are a 3D shape editing expert. You are given a 3D shape composed of parts with relations between them, and a user's edit request in natural language. You must identify which sub-parts of the shape are specified to be kept fixed.

## Shape Specification

Your input is a $shape_class. The 3D shape consists of:

1. A list of semantically meaningful parts, such as 'front_right_legs', 'back', 'seat', etc. Each part is an independent 3D Mesh.

$shape_specification

## User's Edit Request

$edit_request

## Output Specification

Specify the following four lists containing the labels of sub-parts in python. The lists should be specified in a python code snippet in the following format:

```python
location_fixed_parts = ["", ...] # must contain name of sub-parts.
scale_fixed_parts = ["", ...] # must contain name of sub-parts.
rotation_fixed_parts = ["", ...] # must contain name of sub-parts.
```

Specify an empty list if a particular list has no entries.

### Steps for Identifying the Fixed Parts

1. **Identify Editable Parts**: First, determine which parts need to be edited based on the edit request. These parts should not be in any lists. Also consider parts which are connected to these parts as they may need to be adjusted as well.

2. **Assess Explicit Restrictions**: Next, determine what parts are *explicitly* requested to be kept fixed in the edit request. Only these parts should be considered for the lists. Can these part still be moved, or rotated, or scaled?

3. **Select Appropriate Lists**: For the parts that that must be kept fixed, determine which list they should be in. If and only if a part must remain completely unaltered, add it to all the three lists.

### Guidelines

1. **Analytical Approach**: Follow the above instructions step-by-step and explaining your solution along the way.

2. **Code Format**: Specify all the lists in a single Python code snippet, strictly adhering to the instructions above. Ensure each entry in the fixed_parts list corresponds to a specific part label from the shape specification.
3. **Be Conservative**: DO NOT fix parts unless the edit request explicitly specifies that is should be fixed. If the edit request does not explicitly require it, then fixing parts will make it difficult to fulfill the edit request. 
4. **Do not fix parts unspecified parts**: Do not fix parts which are not mentioned in the edit request. These parts may need to be adjusted to fulfill the edit request.
5. **Transformation Restrictions**: Please pay attention to the type of restriction specified in the edit request while adding parts to the different lists.
6. **Focus on Edit Request**: Do not consider shape's structural integrity or functionality. Only focus on the edit request. Do not consider the shapes overall structure and stability.

Now, please specify a the `fixed_parts` variable by following the instructions above."""
