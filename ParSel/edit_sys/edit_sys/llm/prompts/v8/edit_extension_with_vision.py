instructions = """# Overview

You are given a 3D $shape_class object composed of well-labelled semantically meaningful parts and a user's edit request. Your task is suggest extensions of the input edits that lead to more interesting variations of the shape.

## Output Specification

Please specify an `extended_edits` python list variable which contains natural language edit suggestions. 

```python
extended_edits = ["", ...] # must contain edits as natural language strings
```

The suggestions should be based to the shape specification provided in the prompt.

## Examples

Here are some examples of what kind of edit extensions are expected:

### Example 1

Shape Specification: 

```markdown
chair/
    chair_back
    chair_seat
    legs (rotation symmetry)/
        leg_back_right
        leg_front_right
        leg_front_left
        leg_back_left
```

User Edit Request: "I want to widen the chair seat."

Extended Edits: 

```python
extended_edits = [
    "Make the chair seat wider while tilting the legs.",
    "Make the chair seat wider while keeping the legs fixed.",
    "Make the chair seat wider while keeping the back fixed.",
    ...
    ]
```

### Example 2

Shape Specification: 

```markdown
table/
    legs (rotation symmetry)/
        leg_back_right
        leg_back_left
        leg_front_left
        leg_front_right
    tabletop
    tabletop_connector
```

User Edit Request: "I want to shorten the table legs."

Extended Edits:

```python
extended_edits = [
    "Shorten the table legs while thickening the tabletop from the bottom",
    "Shorten the table legs while keeping the tabletop fixed.",
    "Shorten the table legs while expanding the tabletop connector.",
    ...
    ]
```

### Example 3
Shape Specification: 

```markdown
chair/
    back_frame
    back_surface_horizontal_bar
    back_surface_vertical_bars (translation symmetry)/
        back_surface_vertical_bar_left
        back_surface_vertical_bar_center_left
        back_surface_vertical_bar_center
        back_surface_vertical_bar_center_right
        back_surface_vertical_bar_right
    legs (rotation symmetry)/
        leg_front_right
        leg_front_left
        leg_back_left
        leg_back_right
    seat_support
    seat_surface
```
User Edit Request: "I want to make the back frame wider."

Extended Edits: 

```python
extended_edits = [
    "widen the back frame while keeping the legs fixed.",
    "widen the back frame while tilting the legs.",
    "widen the back frame while keeping the front legs fixed."
    ]
```


# Current Task:

Please specify a `extended_edits` list containting $n_edits edit requests for the following shape. A render of the shape is also provided for reference.

## Shape Specification

The part-hierarchy of the shape is as follows:

$shape_specification

## Steps for deriving the edits

1. **Consider Shape Design Variations**: First, consider the given $shape_class and think about the possible edit variations that mesh well with users edit request. How can you extend the user's request to create more interesting variations of the shape?

2. **Consider given Shape**: Now, carefully examine the $shape_class image, as well as the part-hierarchy provided. List down the edit extensions that will create interesting variations of the shape. 

3. Due to system restrictions, only consider two types of edit extensions: (a) Where certain related parts can be kept fixed while the edit is applied to other parts, and (b) Where certain related parts are either scaled/rotated/translated in conjunction with the edit. These edits should be considered on the basis of the shape category and what kind of variations are reasonable for the shape category.

3. **Formatting the edits**: Now, for each of the suggestions, try to format it as shown in the examples above. These edits will be executed by an edit solver, so its important to clearly specify if certain parts are to remain fixed, or edited in a certain fashion.

4. **Write the edits**: Create a python snippet that contains the `extended_edits` list which should contain the formatted natural language edit suggestions from step 3.

## Guidelines

1. *Analytical Approach*: Follow the above instructions step-by-step and explaining your solution along the way.

2. *Code Format*: Specify the list in a single Python code snippet, strictly adhering to the instructions above.

3. *Shape Specific*: Ensure that the edits are based on the shape shown and described above. 

4. *Edit Types*: Only provide edit extensions that involve editing the shape structure like translation, scaling, rotating, widening, etc. Do not make edit extensionsthat add new components or edit the parts beyond these operations.

6. *Wide Coverage*: Try to provide edit extensions that target different parts of the shape to create a diverse set of suggestions. Do not focus on the same part for multiple extensions.

8. *Strictly follow examples*: Strictly follow the format and types of edit shown in the examples.

Now, please specify the `extended_edits` list by following the instructions above."""
