instructions = """# Overview

You are given a 3D $shape_class object composed of well-labelled semantically meaningful parts. Your task is to suggest edits that can be applied to the $shape_class to create interesting and plausible variations of the shape.

## Output Specification

Please specify a `procedural_edits` python list variable which contains natural language edit suggestions. 

```python
procedural_edits = ["", ...] # must contain edits as natural language strings
```

The suggestions should be based to the shape specification provided in the prompt.

## Examples

Here are some examples of what kind of edits are expected:

### Example 1

Shape Specification: 

```markdown
chair/
    chair_back
    chair_seat
    ground
    legs (rotation symmetry)/
        leg_back_right
        leg_front_right
        leg_front_left
        leg_back_left
```

Procedural Edits: 

```python
procedural_edits = [
    "Make the top of the chair back wider.",
    "Shorten chair legs from the top and scale the seat accordingly.",
    "Make the legs radially thicker",
    "Make the seat wider in the left-right direction, moving the legs subsequently.",
    ...
    ]
```

### Example 2

Shape Specification: 

```markdown
table/
    ground
    legs (rotation symmetry)/
        leg_back_right
        leg_back_left
        leg_front_left
        leg_front_right
    tabletop
    tabletop_connector
```

Procedural Edits: 

```python
procedural_edits = [
    "Make the table top radially wider. Subsequently, increase the number of legs as the table top gets wider.",
    "I want to shift the bottom tips of the legs outwards in the left-right direction, while keeping the rest of the table as it is.",
    "Make the table connectors thicker in the down-up direction, scaling the legs and moving the table top accordingly.",
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
    ground
    legs (rotation symmetry)/
        leg_front_right
        leg_front_left
        leg_back_left
        leg_back_right
    seat_support
    seat_surface
```

Procedural Edits: 

```python
procedural_edits = [
    "Make the entire chair wider in the left-right direction and increase the number of vertical bars in the back surface accordingly.",
    "Make each leg thicker radially from its center while keeping the rest of the chair as it is.",
    "Rotate the back frame outwards, while keeping the rest of the chair as it is.",
    ...
    ]
```


# Current Task:

Please specify a `procedural_edits` list containting $n_edits edit requests for the following shape. A render of the shape is also provided for reference.

## Shape Specification

The part-hierarchy of the shape is as follows:

$shape_specification

## Steps for deriving the edits

1. **Consider Shape Design Variations**: First, consider the given $shape_class and think about the possible design variations that lead to interesting alterations of the shape.

2. **Consider given Shape**: Now, carefully examine the $shape_class image, as well as the part-hierarchy provided. List down the design variations that will lead to interesting variations of the shape. Note that due to system restrictions, we can only consider edits that involve editing the shape parts (entirely or its face/edges/corner points etc.) with translation, scaling, rotating, widening, etc. Do not make edits that add new components or edit the parts beyond these operations.

3. **Formatting the edits**: Now, for each of the suggestions, try to format it as shown in the examples above. These edits will be executed by an edit solver, so its important to clearly specify if certain parts are to remain fixed, or edited in a certain fashion.

4. **Write the edits**: Create a python snippet that contains the `procedural_edits` list which should contain the formatted natural language edit suggestions from step 3.

## Guidelines

1. *Analytical Approach*: Follow the above instructions step-by-step and explaining your solution along the way.

2. *Code Format*: Specify the list in a single Python code snippet, strictly adhering to the instructions above.

3. *Shape Specific*: Ensure that the edits are based on the shape shown and described above. 

4. *Edit Types*: Only provide edits that only involve editing the shape structure like translation, scaling, rotating, widening, etc. Do not make edits that add new components or edit the parts beyond these operations.

5. *Edit Amounts*: Do not provide a numerical value for how much the edit should be done. Instead, provide a qualitative description of the edit. 

6. *Wide Coverage*: Try to provide edits that target different parts of the shape to create a diverse set of suggestions. Do not focus on the same part for multiple edits.

7. *Isolated Edits*: Try to provide isolated part specific edits. Additional for each edit specify what is to be kept fixed (i.e. all the parts that are to be kept as it for the edit).

8. *Strictly follow examples*: Strictly follow the format and types of edit shown in the examples.


Now, please specify the `procedural_edits` list by following the instructions above."""
