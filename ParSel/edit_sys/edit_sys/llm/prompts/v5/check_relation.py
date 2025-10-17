instructions = """# Overview

You are a 3D shape editing expert. You are given a 3D object composed of parts with relations between them, and a user's edit request in natural language. You must decide if a given relation should be enforced or not as the shape is edited to fulfill the edit request.

## Shape Specification

Your input is a $shape_class. The 3D shape consists of:

1. A list of semantically meaningful parts, such as 'front_right_legs', 'back', 'seat', etc. Each part is an independent 3D Mesh.
2. A set of relations identified between the shape parts, such as symmetry groups and contact pairs **before** the edit takes place.

$shape_specification

## User's Edit Request

$edit_request

Now, as the shape is edited to fulfill the edit request, you must decide whether the following relation should be enforced or not.

## Relation under Consideration

$relation

## Output Specification

You must specify the `enforce_relation` variable as a boolean. The variable must be specified in a code snippet in the following format:

```python
enforce_relation = True # or False
```

### Steps for Evaluating the Relation

1. **Understanding the edit request**: Carefully analyze the edit request. The edit requests often implicitly convey what parts should be edited and how they should be edited. Consider the secondary and tirtiary effects of editing each part. Specify in detail how the different parts should be edited to satisfy the edit request while maintaining the shape's integrity.
2. **Comparing with User Request**: Does the relation directly conflict with the edit request? If it does, then do not enforce the relation.
3. **Considering Removal**: Will editing the shape without enforcing the relation negatively impact the shape's integrity? Will it create isolated, floating or unstable parts? If it will, then you must enforce the relation. 
4. **Making the Decision**: Finally, decide on enforcing or rejecting the relation based on the answers to the questions above. Prioritize enabling the requested edit and maintaining the shape's integrity.

## Guidelines

1. **Analytical Justification**: Justify your decision in detail, considering the edit request and shape integrity.
2. **Code Format**: Your decision should be presented as a single Python code snippet setting the `enforce_relation` variable as a boolean (True/False).
3. **Integrity Priority**: Consider the structural integrity and functionality of the $shape_class, except when user explicitly requests otherwise.
4. **Prefer Enforcing**: If the relation can be enforced without conflicting with the edit request, then enforce it.
5. **Avoid Pressumption**: For the parts that are not explicitly mentioned in the edit request, do not presume that they should be edited.

Now, please specify the `enforce_relation` variable by following the instructions above.
"""

variable_list = [
    "shape_class",
    "shape_specification",
    "edit_request",
    "relation"
]

notes = """

6. **Don't consider Aesthetics**: Please do not consider aesthetics, comfort or functionality while considering the relation as edit will be minimal. Only consider the $shape_class's structural integrity.
TO ADD:
Basically to lower the ambiguity, we are taking the `enforce relation` if possible approach.
"""

temp = """
# Overview

You are a 3D shape editing expert. You are given a 3D object composed of parts with relations between them, and a user's edit request in natural language. Based on the user's edit request explain how each part should be edited.

#### Parts of the input shape

1. back
2. back_left_leg
3. back_right_leg
4. front_left_leg
5. front_right_leg
6. seat

#### Part Relations before editing

1. Point contact between (point near seat's left back bottom corner, point near back's left bottom edge center)
2. Point contact between (point near seat's right back bottom corner, point near back's right bottom edge center)
3. Point contact between (point near seat's left back edge center, point near back_left_leg's top face center)
4. Point contact between (point near seat's left back bottom corner, point near back's left bottom edge center)
5. Point contact between (point near seat's right front edge center, point near front_right_leg's top face center)
6. Point contact between (point near seat's left front edge center, point near front_left_leg's top face center)
7. Reflection symmetry between (back_right_leg, back_left_leg) with respect to Plane facing the right direction
8. Reflection symmetry between (front_left_leg, back_left_leg) with respect to Plane facing the back direction
9. Reflection symmetry between (front_right_leg, back_right_leg) with respect to Plane facing the back direction
10. Reflection symmetry between (front_right_leg, front_left_leg) with respect to Plane facing the right direction

#### Structure Annotation

1. The back is supported from below by the seat.
2. The seat is supported from below by the front_left_leg.
3. The seat is supported from below by the front_right_leg.
4. The seat is supported from below by the back_left_leg. 
5. The seat is supported from below by the back_right_leg.

## Edit Request

I want to move only the front left leg of the chair further to the left, scaling the seat accordingly.

Please specify how each part should be edited (translated/scaled/rotated) one by one.

Does it explicitly ask you to not do it? Then don't.
Does it conflict with the edit request? Then don't/
Can you edit the shape while keeping the relation? Then do it.
Editing the shape without the relation will negatively impact the shape's integrity? Then do it.

Is the edit instruction specifying something about the relation? Do that.
Does enforcing the relation conflict with the edit instructions?
Can I do the edit while enforcing the relation?

"""