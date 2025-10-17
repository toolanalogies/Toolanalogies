instructions_edit_selection = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts with relations between them, and a user's edit request in natural language. Your goal is to specify how a specific part/relation of the $shape_class should be edited. Towards this end, you must select the most appropriate edit from a list of edit candidates.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Part/Relation in Focus

$part_in_detail

$unedited_parts

## Edit candidates

$options

## Additional Information for Edit Candidates

$edit_info_str
## Output specification

You must specify the `selected_ind` integer variable in a code snippet in the following format:

```python
selected_ind = ... # must be an integer
```

### Steps for Selecting an Edit

1. **Consider The Edit Request**: Look at the edit request. Does it provide explicit clue about how this part should be edited?

2. **Consider Related Parts**: Now, look at the parts related to this edit. Given the edit request, how will all the related parts be edited? Does this information provide a clue about how this part should be edited?

3. **Consider Affect on Related Parts**: Once we edit this part, the parts with a relation to this part, which have not been edited yet may be edited in the later steps. Will the edits on the related parts directly contradict the user's edit request? Then consider an edit which avoids the need of future edits on the related parts.

4. **Simplicity**: Finally, consider that simpler edits are often preferable. For example, symmetric edits are perferable over asymmetric edits, unless they contradict the user's edit request. Additionally, edits with parameters defined with respect to the part itself are preferable over those defined with respect to other parts.

5. **Final Decision**: Finally, specify the `selected_ind` variable. Prioritize following the user's edit request.

### Guidelines

1. **Analytical Approach**: Carefully follow the steps above explaining the logic behind your choices.

2. **Code Format**: Specify the `selected_ind` in a single Python code snippet, strictly adhering to the specified format.

3. **Always Select**: You must select at least one of the edit candidates. If none of the candidates are appropriate, select the one which is least inappropriate.

4. **All Edits satisfy Constraints**: Note that all edits candidates have been designed to satisfy the enforced relations. Therefore, you do not need to worry about the enforced relations while selecting an edit.

Now, please specify the `selected_ind` variable by following the instructions above.
"""
# For example, part level edits are preferable over face level edits, which are in turn preferable over edge level edits. 
instructions_class_selection = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts with relations between them, and a user's edit request in natural language. Your goal is to specify how a specific part of the $shape_class should be edited. Towards this end, you must select the most appropriate edit type from a list of edit types.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Part in Focus

$part_in_detail

$unedited_parts

## Edit Types

Here are the edit types:

$options

## Additional Information for Edit Types

$edit_info_str
## Output specification

You must specify the `selected_ind` integer variable in a code snippet in the following format:

```python
selected_ind = ... # must be an integer
```

### Steps for Selecting an Edit Type

1. **Consider The Edit Request**: Look at the edit request. Does it provide explicit clue about how this part should be edited?

2. **Consider Related Parts**: Now, look at the parts related to this edit. Given the edit request, how will all the related parts be edited? Does this information provide a clue about how this part should be edited?

3. **Consider Affect on Related Parts**: Once we edit this part, the parts with a relation to this part, which have not been edited yet, may be edited in the later steps. What edits will be necessary on the related parts? Will the edits on the related parts directly contradict the user's edit request? Then consider an edit which avoids the need of edits on the related parts.

4. **Simplicity**: Finally, consider that simpler edits are often preferable. For example, part level edits are preferable over face level edits, which are in turn preferable over edge level edits. Similarly, symmetric edits are perferable over asymmetric edits, unless they contradict the user's edit request. Finally, edits with parameters defined with respect to the part itself are preferable over those defined with respect to other parts.

5. Finally, specify the `selected_ind` variable. Prioritize following the user's edit request.

### Guidelines

1. **Analytical Approach**: Carefully follow the steps above explaining the logic behind your choices.

2. **Code Format**: Specify the `selected_ind` in a single Python code snippet, strictly adhering to the specified format.

3. **Always Select**: You must select at least one of the edit candidates. If none of the candidates are appropriate, select the one which is least inappropriate.

4. **All Edits satisfy Constraints**: Note that all edits candidates have been designed to satisfy the enforced relations. Therefore, you do not need to worry about the enforced relations while selecting an edit.

Now, please specify the `selected_ind` variable by following the instructions above.
"""


# How to solve this task:
# Look at the edit request, and understand how different parts will be edited. Does it provide explicit clue about how this part should be edited? # COnsider edit request
# If not, Then look at the parts related to this one, and consider how they will be edited based on the edit request. Given this, how should this part be edited? # Consider future edits.
# If both the steps don't give a clear answer, then consider that simpler edits are preferable, PartEdit over FaceEdit over EdgeEdit.

variables = [
    "shape_class",
    "shape_specification",
    "edit_request",
    "all_edits_in_detail", # To Remove.
    "part_in_detail",
    "remaining_parts",
    "options",
    
]

additional_class_level_information = {
    "PartTranslate": "This edit is used to move a part in the specified direction.",
    "FaceTranslate": "This edit is used to a) expand/contract a part or b) shear a part. It is important to consider the direction of translation, and which face its applied to. For instance, translating the up face of a chair seat upwards will expand the seat (while keeping the down face fixed). In contrast, translating the down face of a chair leg towards the front will shear the leg (while keeping the up face fixed).",
    "EdgeTranslate": "This edit is used to expand or contract and object assymetrically. For example, moving the left front edge of a chair seat to the left will expand the front of the seat only to the left. It is important to carefully consider the direction of translation and the edge its applied to.",
    "PartRotate": "This edit is used to rotate a part. Carefully consider the direction and origin of the rotation axis",
    "PartScale1D": "This edit is used to expand or contract a part along a single direction. The origin of the scaling is important to consider. When applied with the origin at the center of a part, it will expand/contract the part in both directions (for example left and right, or up and down). However, when applied with the origin at a face center / edge center, it will expand/contract the part without changing the face/edge where the origin is present.",
    "FaceScale1D": "This edit is used to taper a part. It is import to consider the direction of scaling and the face its applied to. For example, scaling the top face of the chair back, along the right direction will make the top of the back wider, while keeping the bottom fixed.",
    "PartScale2D": "This edit is used to expand or contract a part along two directions (i.e. along a a plane). The origin of the scaling is important to consider. When applied with the origin at the center of a part, it will expand/contract the part in four directions. For example, if the plane normal points to up direction, then the part will expand/contract in the left, right, front and back directions. When applied with the origin at a face center / edge center, it will expand/contract the part without changing the face/edge where the origin is present.", 
    "FaceScale2D": "This edit is used to taper a part in two directions (i.e. along a plane). It is important to consider the direction of scaling and the face its applied to. For example, when this edit is applied to the down face of the chair seat with the plane normal pointing in the up/down direction, it will expand/contract the seat (in left, right, front and back directions) while keeping the top face of the seat fixed.",
    "PartScale3D": "This edit is used to expand or contract a part along all three directions. The origin of the scaling is important to consider. When applied with the origin at the center of a part, it will expand/contract the part in all six directions (i.e. left, right, up, down, front and back). When applied with the origin at a face center / edge center, it will expand/contract the part without changing the face/edge where the origin is present.",
    "PartShear": "This edit is used to shear a part. When the shear plane center is at the part's center, the part will be sheared such that two of its faces (one along the shear plane direction, and the other opposite to it) will be sheared in the direction of shearing. When the shear plane center is at a face center / edge center, the part will be sheared such that the face/edge where the origin is present will not be sheared, but the opposite face will be sheared.",
    "ChangeDelta": "This edit is used to change the distance between the sub-parts within a part.",
    "ChangeCount": "This edit is used to change the number of instances of the sub-parts within a part.",
    "FaceRotate": "This edit is used to expand or contract a part by  rotating one of its faces. Carefully consider the direction and origin of the rotation axis",
}
improved_class_level_information = {
    "PartTranslate": "Moves a part in a specified direction. Example: Shifting a table top sideways.",
    "FaceTranslate": "Used to either expand/contract a part or shear it, depending on the face and direction. Example: Raising the top face of a chair seat expands it, while shifting the bottom face of a leg forward shears the leg.",
    "EdgeTranslate": "Expands or contracts a part asymmetrically. Example: Moving the left edge of a chair seat leftward expands the seat's front left side.",
    "PartRotate": "Rotates a part around a specified axis. Consider the rotation's direction and origin. Example: Rotating the rungs to a bunker bed.",
    "PartScale1D": "Scales a part along one direction. The scaling origin affects the outcome. Example: Scaling a rod lengthwise from its center elongates it equally on both sides.",
    "FaceScale1D": "Tapers a part along one direction. The direction and face of scaling are crucial. Example: Scaling the top face of a chair's backrest rightward/leftward widens its top.",
    "PartScale2D": "Scales a part along two directions in a plane. The scaling's origin is key. Example: Scaling a square table top from the center enlarges it in all four directions.",
    "FaceScale2D": "Tapers a part in two directions within a plane. The face and direction matter. Example: Scaling the bottom face of a chair seat with plane normal pointing in up/down direction expands the seat while the top remains fixed.",
    "PartScale3D": "Scales a part in all three dimensions. The origin of scaling impacts the result. Example: Scaling a cube from its center enlarges it uniformly in all directions.",
    "PartShear": "Shears a part, altering its shape. The position of the shear plane affects which faces are sheared. Example: Shearing a rectangular prism can slant its top face while keeping the bottom face unchanged.",
    "FaceRotate": "Expands or contracts a part by rotating one of its faces. Consider the rotation's direction and origin. Example: rotating the cabinet doors of a wardrobe about its hinges."
}

def get_edit_info_str(unique_option_classes):
    info_str = ""
    for ind, option_class in enumerate(unique_option_classes):
        info_str += f"{ind+1}. {option_class}: {additional_class_level_information[option_class]}\n"
    return info_str

# part_in_detail contains -> Part name, all its relations - broken etc. All Edits till now.

instructions_edit_selection_nocot = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts with relations between them, and a user's edit request in natural language. Your goal is to specify how a specific part/relation of the $shape_class should be edited. Towards this end, you must select the most appropriate edit from a list of edit candidates.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Part/Relation in Focus

$part_in_detail

$unedited_parts

## Edit candidates

$options

## Additional Information for Edit Candidates

$edit_info_str
## Output specification

You must specify the `selected_ind` integer variable in a code snippet in the following format:

```python
selected_ind = ... # must be an integer
```

Now, please specify the `selected_ind` variable by following the instructions above.
"""
# For example, part level edits are preferable over face level edits, which are in turn preferable over edge level edits. 
instructions_class_selection_nocot = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of cuboidal parts with relations between them, and a user's edit request in natural language. Your goal is to specify how a specific part of the $shape_class should be edited. Towards this end, you must select the most appropriate edit type from a list of edit types.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Part in Focus

$part_in_detail

$unedited_parts

## Edit Types

Here are the edit types:

$options

## Additional Information for Edit Types

$edit_info_str
## Output specification

You must specify the `selected_ind` integer variable in a code snippet in the following format:

```python
selected_ind = ... # must be an integer
```

Now, please specify the `selected_ind` variable by following the instructions above.
"""

additional_information_option_select_v1 = """### Steps for Selecting an Edit

1. **Consider The Edit Request**: Look at the edit request. Does it provide explicit clue about how this part should be edited?

2. **Consider Related Parts**: Now, look at the parts related to this edit. Given the edit request, how will all the related parts be edited? Does this information provide a clue about how this part should be edited?

3. **Consider Affect on Related Parts**: Once we edit this part, the parts with a relation to this part, which have not been edited yet may be edited in the later steps. Will the edits on the related parts directly contradict the user's edit request? Then consider an edit which avoids the need of future edits on the related parts.

4. **Simplicity**: Finally, consider that simpler edits are often preferable. For example, symmetric edits are perferable over asymmetric edits, unless they contradict the user's edit request. Additionally, edits with parameters defined with respect to the part itself are preferable over those defined with respect to other parts.

5. **Final Decision**: Finally, specify the `selected_ind` variable. Prioritize following the user's edit request.

### Guidelines

1. **Code Format**: Specify the `selected_ind` in a single Python code snippet, strictly adhering to the specified format.

2. **Always Select**: You must select at least one of the edit candidates. If none of the candidates are appropriate, select the one which is least inappropriate.

3. **All Edits satisfy Constraints**: Note that all edits candidates have been designed to satisfy the enforced relations. Therefore, you do not need to worry about the enforced relations while selecting an edit.

Now, please specify the `selected_ind` variable by following the instructions above.
"""

additional_information_option_select_v2 = """### Steps for Selecting an Edit

1. **Consider The Edit Request**: Look at the edit request. Does it provide explicit clue about how this part should be edited?

2. **Simplicity**: If there is no clue in the edit request, prefer simpler edits: Translation > Rotation > Scaling > Shearing. Carefully consider the type of face-translations to judge if its simply scaling or shearing. Within scaling, prefer ones with lower number of dimensions.

3. **Lower Amounts**: Prefer edits which have lower amount expression.

4. **Final Decision**: Finally, specify the `selected_ind` variable. Prioritize following the user's edit request.

### Guidelines

1. **Code Format**: Specify the `selected_ind` in a single Python code snippet, strictly adhering to the specified format.

2. **Always Select**: You must select at least one of the edit candidates. If none of the candidates are appropriate, select the one which is least inappropriate.

3. **All Edits satisfy Constraints**: Note that all edits candidates have been designed to satisfy the enforced relations. Therefore, you do not need to worry about the enforced relations while selecting an edit.

Now, please specify the `selected_ind` variable by following the instructions above.
"""