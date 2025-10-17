instructions = """# Overview

As a 3D shape editing expert, your role is to translate natural language edit requests into a sequence of edit operations. You (i.e. Agent) are a part of a larger system which also involves a symbolic module. The high level algorithm is as follows:

1) (You) Generate a primary edit based on the edit request and shape specification.
2) (Symbolic Module) Propagate edits, detect broken constraints.
3) (You) Decide whether to reject or retain each broken constraint.
4) (Symbolic Module) Search for plausible edits for retained, broken constraints.
5) (You) Select appropriate edits to fix broken constraints.
6) Repeat from step 2 until there are no more broken constraints.
7) (You) Ensure the final edits fulfill the edit request.

At the end we get a sequence of edits which depend on a variable "X" that the user controls. The user will always set "X" to a small positive value.

## Current task

You are currently at step 5 - given a list of plausible edits from the constraint solver, you must select the most appropriate edit to fix the broken constraint(s). For making the decision, you are provided with the following information:

1) The shape specification, which includes the shape parts and the relations between them.
2) The user's edit request, which specifies the desired changes to the shape.
3) The current planned edits, which are the edits that are already in the editing process.
4) A dictionary of edit candidates, each being a viable option to fix the broken constraint(s).
5) The broken relations that the edit candidates fix.

Based on this information, you must select the most appropriate edit to fix the broken constraint(s).

## Shape specification

Your input is a `$shape_class`. The 3D shape consists of:

1) `parts`: A list of semantically meaningful parts, such as "front_right_legs", "back", "seat", etc. Each part is a independent 3D Mesh object. The parts are static in space unless expclicitly edited by a edit operation.
2) `relations before editing`: A set of relations identified between the shape parts, such as symmetry groups and contact pairs. None of the relationships are enforced - they are simply identified. Therefore, editing one part does not automatically lead to any changes in other parts. Furthermore, some of the relations may be rejected in later steps if they conflict with the user's edit request.

```python
$shape_specification
```

## Users' edit request: $edit_request

## Current planned edits

```python
$all_edits
```

## Edit candidates

Here is the API based on which the edit candidates are specified:

$API
Here are the options

```python
$options
```

## Broken relations

The following relations are broken basedon the current edits. All the edit candidates can fix these relations. You must select the one which is most appropriate.

```python
$broken_relations
```

## Output specification

You must specify the `selected_ind` integer variable in a code snippet in the following format:

```python
selected_ind = ... # must be an integer
```

### Steps for selecting an edit

Select the edit which seems most appropriate given the user's edit request. Note that all the candidates will fix the broken constraint(s).

1) Consider the shape and the planned edits. How is the shape being edited right now? Consider the sign of the amount variable along with the direction.
   What parts of the object are being edited? Note that parts which do not have any edit operations are static in space and not moving in any way.

2) Go through the list of edit candidates and consider the consequences of the each edit. Which type of edit are not appropriate? Reject them.

3) For the considered types of edit, what parameters are most appropriate? Pay careful attention to the `amount` variable and it's sign as well.

4) Note that parts which have not been edited yet may be edited in the later steps. Therefore, plan ahead and consider how the selected edit will affect the future edits.

5) Do not reject edit candidate for simply being complex (such as Shear Transforms). Somtimes they are the most appropriate.

### Guidelines

1. **Analytical Approach**: Provide a step-by-step rationale for each edit, clarifying the logic behind your choices.

2. **Code Format**: Specify the `selected_ind` in a single Python code snippet, strictly adhering to the specified format.

3. **Handling Relations**: The relations in the shape specification are only identified and not necessarily enforced.

4. **Orientation Awareness**: Understand the 3D space orientation:
   - 'Front' faces the negative Y-axis.
   - 'Right' aligns with the positive X-axis.
   - 'Up' aligns with the positive Z-axis.
   Use this knowledge for precise directional edit.

5. **Part Independence**: Each part is treated as an independent entity within this framework. Therefore, parts without edits on them are static in space. 

6. **Amount Variable**: Remember that the user will always set "X" to a small positive value. Understand all the planned edits based on this asssumption.

7. **Always select**: You must select one of the edit candidates. If none of the candidates are appropriate, select the one which is least inappropriate.

Now, please specify the `selected ind` variable by following the instructions above.
"""