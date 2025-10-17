instructions = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of parts with relations between them, and a user's edit request in natural language. Your goal is to specify how the $part of the $shape_class should be edited. Towards this end, you must select the most appropriate edit from a list of edit candidates.

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

## Currently Planned Edits

The following part-level edits will be performed to fulfill the edit request:

$all_edits

$remaining_parts

## $part's Enforced Relation(s)

These part is being edited due to the user's edit request and to satisfy the following relation(s):

$broken_relations

## Edit candidates for the $part

Here are the edit candidates:

$options

IMPORTANT: All the edit candidates enforce the $part's enforced relation(s).

## Output specification

You must specify the `selected_ind` integer variable in a code snippet in the following format:

```python
selected_ind = ... # must be an integer
```

### Steps for Selecting an Edit

1. **Review Information**: First, summarize how the planned edits support the edit request. Additionally, note the part-level edits that the edit-request explicitly requires. Next, specify how the part $part should be edited based on the shape specification, the edit request, and the enforced relation(s).
2. **Initial Screening**: Identify appropriate types of edit from the API for the $part based on the edit request. Discard candidates which do not have the appropriate edit type. List their indices.
3. **Understand Remaining Edits**: Consult the API and for the remaining edit candidates summarize how they alter the $part based on their's parameters (including `amount`, `origin` etc.). Its important to analyze each candidate independently. If the edit is asymmetric, specify which faces of the $part remain unaltered by the edit. Complex, asymmetric edits might be preferable if the edit request discards the symmetries in the object as well.
4. **Analyze Future Edit**: Now, for each of the edit candidates, consider how they affect the $shape_class overall. An edit might necessitate edits to other parts to retain the shape's structural integrity. Analyze if those future edits may conflict with the user's request. If they will, reject the edit candidate, noting the indices that should be removed.
5. **Final Selection**: Finally, from the remaining edit candidates, choose the edit that best fulfills the edit request while maintaining the shape’s integrity.

### Guidelines

1. **Analytical Approach**: Carefully follow the steps above explaining the logic behind your choices.
2. **Code Format**: Specify the `selected_ind` in a single Python code snippet, strictly adhering to the specified format.
3. **Always Select**: You must select at least one of the edit candidates. If none of the candidates are appropriate, select the one which is least inappropriate.
4. **Plan Ahead**: Remember that parts which have not been edited yet may be edited in the later steps.
5. **Consider Complex Edits** Do not dismiss an edit candidate simply for being complex, like Shear or DirectionalVariableScale transforms. Most of the times they are preferable edits.
6. **Shape Symmetry**: Prefer asymmetric edits if the user's edit request discards the symmetries in the object as well.
7. **All Edits satisfy Constraints**: Note that all edits candidates have been designed to satisfy the enforced relations. Therefore, you do not need to worry about the enforced relations while selecting an edit.
8. **Don't consider Aesthetics**: Please do not consider aesthetics, comfort or functionality while considering the edits as the amount of edit (based on the variable X) will be adjusted by the user. Only consider the $shape_class's structural integrity.

Now, please specify the `selected_ind` variable by following the instructions above.
"""


steps = """
1. Does the part semantic/structure tell me something? 
2. Does the edit instruction tell me something about which edit to select?
3. 
31 - The seat should be DVS scaled about the corner.
32 - The leg should be scaled up so that it does not lose contact with the ground. If you move the seat up and the legs up then you have not "really" moved it up. 

"""

