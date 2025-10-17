instructions = """# Overview

You are a 3D shape editing expert. You are given a 3D $shape_class composed of parts with relations between them, and a user's edit request in natural language. Given a list of part-level edits, you must determine if they completely fulfill the user's edit request.

## Shape Specification

Your input is a 3D $shape_class. It consists:

1. A list of semantically meaningful parts, such as 'front_right_legs', 'back', 'seat', etc. Each part is an independent 3D Mesh.
2. A set of relations identified between the shape parts, such as symmetry groups and contact pairs **before** the edit takes place.

$shape_specification

## User's Edit Request

$edit_request

## Edit API

The edits are defined using the following API:

$API

## List of Part-level Edits

$all_edits

$remaining_parts

## Output specification

You must specify the `edit_complete` boolean variable in a code snippet in the following format:

```python
edit_complete = ... # must be an boolean
```

### Steps for Determining Edit Completeness

1. **Review Edit Request**: Summarize the edit request in your own words.

2. **Review Planned Edits**: Summarize how the planned edits support the edit request.

3. **Review Unedited parts**: Carefully analyze the unedited parts. Did the edit request explicitly require them to be edited? If even a single part which should have been edited is not edited, then the list of edits is incomplete. Note that the relations are already checked and enforced so you don't need to factor them in your decision focus on the edit request.

### Guidelines

1. **Analytical Approach**: Carefully follow the steps above explaining the logic behind your choices.
2. **Code Format**: Specify the `edit_complete` variable in a single Python code snippet, strictly adhering to the specified format.
3. **Edit Completeness Priority**: Prioritize enabling the requested edit and maintaining the shape's integrity. If either of these cannot be achieved, then the list of edits is incomplete.
4. **Shape Relations**: Note that all the part relations are stable, and you don't need to consider them.

Now, please specify the `edit_complete` variable by following the instructions above.
"""