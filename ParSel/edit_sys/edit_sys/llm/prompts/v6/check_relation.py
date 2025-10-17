from string import Template
instructions = """# Overview

You are a 3D shape editing expert. You are given a 3D object composed of cuboidal parts with relations between them, and a user's edit request in natural language. You must decide if a given relation should be enforced, updated, or deleted.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Relation under Consideration

$relation_in_detail

## Output Specification

You must specify the `selected_option` variable as a integer (should be a one of the options). The variable must be specified in a code snippet in the following format:

```python
selected_option = 0 # or 1 or 2
```

## Options

$options_string

## Steps for Relation Evaluation

* Interpreting the Edit Request: Examine the edit request carefully and elaborate the edit request in detail.

* $option_tips

* Final Decision: Conclude by considering the previous steps. Prioritize implementing the edit as requested.

## Guidelines

1. **Analytical Justification**: Justify your decision in detail, considering the edit request followed by the shape integrity.

2. **Code Format**: Your decision should be presented as a single Python code snippet setting the `selected_option` variable as a Integer. It should be one of the options mentioned above.

3. **Explicit Mention**: Not mentioning a part in the edit request does not imply that the part should not be edited. When considering which parts to edit or not, do not assume that the unmentioned parts should not be edited.

Now, please explain your reasoning step-by-step and specify the `selected_option` variable by following the instructions above."""

variable_list = [
    "shape_class",
    "shape_specification",
    "edit_request",
    "relation"
]

variables = [
    "shape_class",
    "shape_specification",
    "edit_request",
    "relation_in_detail",
    "options_string",
    "option_tips"
]
option_0_str = """0. This relation should be rejected. The parts $unedited_parts will not be edited to enforce this relation."""

option_1_sym_group_str = """1. This relation should be enforced. This implies that the the following edits: $potential_new_edits will be performed to enforce this relation."""

option_2_sym_group_str = """2. The parameters of this relation should be updated. This implies for Translational or Rotational symmetry, either the number of entities in the symmetry relation, or the spacing between them will be updated. For Reflection symmetry, this implies that the plane of reflection will be adjusted."""

option_1_contact_str = """1. This relation should be enforced. This implies that the parts $unedited_parts will be edited to enforce this relation."""

option_2_contact_str = """2. The relation should be rejected. The part $unedited_parts will not be edited and remain in the exact same location with the same orientation and size even if there is inter penetration or unstability between $edited_part and $unedited_parts."""

option_1_fixed_str = """1. This relation should be enforced. While searching for edit candidates for $unedited_parts, we will enforce this relation. This implies that only edit candidates which maintain this relation will be considered."""

option_2_fixed_str = """2. The parameters of this relation should be updated. This means while searching for edit candidates for $unedited_parts, we will not enforce this relation. The relation parameters will be updated later if possible else it will be broken."""



option_0 = Template(option_0_str)
option_1_contact = Template(option_1_contact_str)
option_1_sym_group = Template(option_1_sym_group_str)
option_2_sym_group = Template(option_2_sym_group_str)
option_2_contact = Template(option_2_contact_str)
option_1_fixed = Template(option_1_fixed_str)
option_2_fixed = Template(option_2_fixed_str)

def relation_option_set_to_string(relation, sym_group, option_set, potential_new_edits="", has_fixed=False):
    string_set = []
    part = relation.get_unedited_parts()
    edited_part = [x for x in relation.parts if x not in part]
    if len(edited_part) > 0:
        edited_part = edited_part[0]
        edited_part_label = edited_part.full_label
    else:
        edited_part_label = ""
    part_string = ", ".join([x.full_label for x in part])
    part_string = "(" + part_string + ")" 
    substitute_dict = {"unedited_parts": part_string,
                       "edited_part": edited_part_label,
                       "potential_new_edits": potential_new_edits}
    if 0 in option_set:
        string_set.append(option_0.substitute(substitute_dict))
    if 1 in option_set:
        if sym_group:
            string_set.append(option_1_sym_group.substitute(substitute_dict))
        else:
            if has_fixed:
                string_set.append(option_1_fixed.substitute(substitute_dict))
            else:
                string_set.append(option_1_contact.substitute(substitute_dict))
    if 2 in option_set:
        if sym_group:
            string_set.append(option_2_sym_group.substitute(substitute_dict))
        else:
            if has_fixed:
                string_set.append(option_2_fixed.substitute(substitute_dict))
            else:
                string_set.append(option_2_contact.substitute(substitute_dict))
    
    option_string = "\n".join(string_set)
    return option_string
    
# super specific tips
ss_tip_contact = """Follow these steps to select an option:

1. If the user's request explicitly requires that the part $unedited_parts remain unedited, then select option 2.
2. If the request does not explicitly mention the part $unedited_parts, then consider the following: What will happen if we edit part $edited_part while keeping $unedited_parts fixed? If the shape will become unstable, or there could be inter-penetration between $unedited_parts and $edited_part, then selection option 1. Option 2 should only be selected if maintaining this relation as given is not important."""
ss_tip_contact = Template(ss_tip_contact)

# super specific tips
ss_tip_contact_fixed = """Follow these steps to select an option:

1. Consider how the part $unedited_parts will be edited in the future. Is it important to perform the edit when enforcing this relation exactly as it is? i.e. Should the point in $edited_part which contacts $unedited_parts remain exactly the same? If that is the case, select option 1.

2. If its unclear how the part $unedited_parts will be edited in the future, then its preferable to update the relation. Select option 2."""
ss_tip_contact_fixed = Template(ss_tip_contact_fixed)


ss_tip_sym_01 = """Follow these steps to select an option: 

1. If the user's request explicitly requires that the unedited part $unedited_parts remain unedited, then select option 0.

2. If the request does not make it clear if the part is to be edited or not, then consider what will happen when we edit $unedited_parts with the edit shown above. If the edits will conflict with the user's request, select option 0.

3. If its still unclear which option to pick, select option 1."""

ss_tip_sym_02 = """Follow these steps to select an option:

1. If the user's request explicitly requires that the symmetry relation be maintained, then select option 2. 

2. If the user's request does not explicitly require the symmetry to be maintained, consider what will happen if we do not maintain the symmetry relation. If that can lead to an unstable structure, then select option 2. 

3. If its still not clear, then select option 0."""

ss_tip_sym_12 = """This combination should not happen!"""

ss_tip_sym_012 = """Follow these steps to select an option:

1. Consider the potential edit on $unedited_parts due to enforcing the relation. If this will interfere with the user's request, then reject option 1.

2. Updating the relation is preferable over rejecting the relation."""


sym_mapper = {
    (0, 1): Template(ss_tip_sym_01),
    (0, 2): Template(ss_tip_sym_02),
    (1, 2): Template(ss_tip_sym_12),
    (0, 1, 2): Template(ss_tip_sym_012)
}

def get_option_tips(relation, sym_group, valid_options, has_fixed=False):
    m = [x for x in valid_options]
    m.sort()
    
    part = relation.get_unedited_parts()
    edited_part = [x for x in relation.parts if x not in part]
    if len(edited_part) > 0:
        edited_part = edited_part[0]
        edited_part_label = edited_part.full_label
    else:
        edited_part_label = ""
    part_string = ", ".join([x.full_label for x in part])
    part_string = "(" + part_string + ")" 
    substitute_dict = {"unedited_parts": part_string,
                       "edited_part": edited_part_label}
    
    if sym_group:
        tip = sym_mapper[tuple(m)]
    else:
        if has_fixed:
            tip = ss_tip_contact_fixed
        else:
            tip = ss_tip_contact
    tip = tip.substitute(substitute_dict)
    return tip

# Steps
# Contacts -> Edit Part to fix the relation.
#          -> Create a new relation based on the new configuration.

# Sym groups

# Question Consider only contacts:
# When would you want enforce
# When would you want to update
# If the user clearly requests the other part to remain unedited, then select update. Otherwise, select enforce. 

# Question to consider for sym groups:

# Decide between Break, enforce, and update
# Decide between Break and update
# Decide between Break, enforce,
# Decide between enforce or update.


instructions_nocot = """# Overview

You are a 3D shape editing expert. You are given a 3D object composed of cuboidal parts with relations between them, and a user's edit request in natural language. You must decide if a given relation should be enforced, updated, or deleted.

## Shape Specification

Your input is a $shape_class. The part-hierarchy of the $shape_class is described as follows:

$shape_specification
## User's Edit Request

$edit_request

## Relation under Consideration

$relation_in_detail

## Output Specification

You must specify the `selected_option` variable as a integer (should be a one of the options). The variable must be specified in a code snippet in the following format:

```python
selected_option = 0 # or 1 or 2
```

## Options

$options_string

Now, please specify the `selected_option` variable by following the instructions above.
"""

additional_info = """## Steps for Relation Evaluation

* Interpreting the Edit Request: Examine the edit request carefully.

* $option_tips

* Final Decision: Conclude by considering the previous steps. Prioritize implementing the edit as requested.

## Guidelines

1. **Code Format**: Your decision should be presented as a single Python code snippet setting the `selected_option` variable as a Integer. It should be one of the options mentioned above.

2. **Explicit Mention**: Not mentioning a part in the edit request does not imply that the part should not be edited. When considering which parts to edit or not, do not assume that the unmentioned parts should not be edited.

Now, please explain your reasoning step-by-step and specify the `selected_option` variable by following the instructions above.
"""