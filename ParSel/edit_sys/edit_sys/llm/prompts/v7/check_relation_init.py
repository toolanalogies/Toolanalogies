from string import Template
from .settings import NOCOT

start = """# Overview

You are a 3D shape editing expert. You are given a 3D object composed of cuboidal parts with relations between them, and a user's edit request in natural language. You must decide if a given relation will be broken, kept-as-is, or updated as the shape is edited as per the user's request.

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

# Where 0 implies the relation will be broken, 
# 1 implies the relation will be kept as is, ie. even after editing, the parts retain the symmetry relation with the same parameters,
# 2 implies the relation will be kept, but its parameters will be updated.
```

## Options

$options_string

## Steps for Relation Evaluation

* Interpreting the Edit Request: Examine the edit request carefully and elaborate the edit request in detail. Elaborate how the parts in the relation will be edited to fulfill the user's request.

* $option_tips

* Final Decision: Decide which option to pick based on the above steps. Remember that the shape will be edited exactly as the user requests."""
guidelines_list = [
    "**Analytical Justification**: Justify your decision in detail, considering the edit request followed by the shape integrity.",
    "**Code Format**: Your decision should be presented as a single Python code snippet setting the `selected_option` variable as a Integer. It should be one of the options mentioned above.",
    "**Explicit Mention**: Not mentioning a part in the edit request does not imply that the part should not be edited. When considering which parts to edit or not, do not assume that the unmentioned parts should not be edited.",
]

end = """Now, please explain your reasoning step-by-step and specify the `selected_option` variable by following the instructions above."""

def instructions(nocot=NOCOT):
    if nocot:
        selected_guidelines = guidelines_list[1:]
    else:
        selected_guidelines = guidelines_list
    guideline_str = "### Guidelines\n\n"
    for ind, guideline in enumerate(selected_guidelines):
        guideline_str += f"{ind + 1}. {guideline}\n\n"
    instructions = "\n\n".join([start, guideline_str, end])
    return instructions

option_0_str = """0. The relation will be broken. This implies that after editing the parts will not retain this symmetry relation."""

option_1_sym_group_str = """1. The relation will be kept as is. This implies that even after editing, the parts retain the symmetry relation with the same parameters."""

option_2_sym_group_str = """2. The relation will be kept, but its parameters will be updated. This implies that after editing the parts will retain the symmetry relation, but the parameters of the symmetry relation will change. For Translational or Rotational symmetry, either the number of entities in the symmetry relation, or the spacing between them will be updated. For Reflection symmetry, the plane of reflection will be adjusted."""


option_0 = Template(option_0_str)
option_1_sym_group = Template(option_1_sym_group_str)
option_2_sym_group = Template(option_2_sym_group_str)

def relation_option_set_to_string(relation, option_set):
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
                       "edited_part": edited_part_label}
    if 0 in option_set:
        string_set.append(option_0.substitute(substitute_dict))
    if 1 in option_set:
        string_set.append(option_1_sym_group.substitute(substitute_dict))
    if 2 in option_set:
        string_set.append(option_2_sym_group.substitute(substitute_dict))
    
    option_string = "\n".join(string_set)
    return option_string
    
ss_tip_sym_01 = """Follow these steps to select an option:

1. If the user's request explicitly requires that any of the parts in the relation remain unedited, then select option 0.

2. If the request does not make it clear if the part is to be edited or not, then consider if the edit can be performed while enforcing this relation. If fulfilling the user's request while enforcing this relation (as it is) is impossible, select option 0.

3. If its still unclear which option to pick, select option 1."""

ss_tip_sym_02 = """Follow these steps to select an option:

1. If the user's request explicitly requires that this symmetry relation be maintained or updated, then select option 2. 

2. If the user's request does not explicitly require the symmetry to be maintained, consider what will happen if we do not maintain the symmetry relation. If that can lead to an unstable structure, then select option 2. 

3. If its still not clear, then select option 0."""

ss_tip_sym_012 = """Follow these steps to select an option:

1. Consider the potential edit that must be performed to enforce this relation, while also fulfilling the user's edit request. If this will interfere with the user's request, then reject option 1.

2. Updating the relation is preferable over rejecting the relation."""

ss_tip_sym_12 = """This combination should not happen!"""



sym_mapper = {
    (0, 1): Template(ss_tip_sym_01),
    (0, 2): Template(ss_tip_sym_02),
    (1, 2): Template(ss_tip_sym_12),
    (0, 1, 2): Template(ss_tip_sym_012)
}

def get_option_tips(relation, valid_options):
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
    
    tip = sym_mapper[tuple(m)]
    tip = tip.substitute(substitute_dict)
    return tip
