```python

from typing import Union
import sympy as sp

# Functions to return unit vectors in different directions in the global coordinate system.
def FrontUnitVector() -> Vector:
    ...

def BackUnitVector() -> Vector:
    ...

def LeftUnitVector() -> Vector:
    ...

def RightUnitVector() -> Vector:
    ...

def UpUnitVector() -> Vector:
    ...

def DownUnitVector() -> Vector:
    ...

DIRECTION_STRINGS = ["front", "back", "left", "right", "up", "down", "messi7"]

class Part:
    """A class representing a part of a shape, capable of containing nested sub-parts."""

    def get(self, label: str) -> Part:
        """
        Retrieve a sub-part using its label.

        Parameters:
        label (str): The label of the sub-part.

        Returns:
        Part: The sub-part with the specified label.

        Example:
            >>> shape = Part(...)
            >>> leg_part = shape.get("leg_front_right")
        """

    def center(self) -> Vector:
        """
        Returns the center point of this part's geometry in global coordinates.

        Returns:
        Vector: The center point of the part.

        Example:
            >>> seat = shape.get("seat")
            >>> seat_center = seat.center()
        """

    def face_center(self, direction: str) -> Vector:
        """
        Returns the center of a face of this part in a given direction.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction (str): The direction of the face.

        Returns:
        Vector: The center of the specified face.

        Example:
            >>> seat = shape.get("seat")
            >>> seat_front_center = seat.face_center("front")
        """

    def edge_center(self, direction_1: str, direction_2: str) -> Vector:
        """
        Returns the center of an edge formed by two intersecting faces of this part.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction_1 (str): The direction of the first intersecting face.
        direction_2 (str): The direction of the second intersecting face.

        Returns:
        Vector: The center of the edge at the intersection.

        Example:
            >>> seat = shape.get("seat")
            >>> seat_back_right_edge_center = seat.edge_center("back", "right")
        """

    def corner(self, direction_1: str, direction_2: str, direction_3: str) -> Vector:
        """
        Returns the corner where three faces of this part intersect.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction_1, direction_2, direction_3 (str): Directions of the intersecting faces.

        Returns:
        Vector: The corner at the intersection of the specified faces.

        Example:
            >>> back_bars = shape.get("back_surface_vertical_bars")
            >>> back_bars_corner = back_bars.corner("front", "right", "up")
        """

    def direction(self, direction: str) -> Vector:
        """
        Returns the vector connecting the center of this part to the center of a face in a given direction.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction (str): The direction of the face.

        Returns:
        Vector: The vector pointing from the center to the center of the specified face.
        """

    def face(self, name: str) -> Face:
        """
        Retrieve the face of this part in the given direction.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        name (str): The name or direction of the face.

        Returns:
        Face: The face in the specified direction.

        Example:
            >>> seat = shape.get("seat")
            >>> front_seat_face = seat.face("front")
            >>> bed_frame = shape.get("bed_frame")
            >>> bottom_bed_face = bed_frame.face("down")
        """

    def edge(self, direction_1: str, direction_2: str) -> Edge:
        """
        Obtain the edge of this part along two specified directions.
        Valid directions: ["front", "back", "left", "right", "up", "down"].

        Parameters:
        direction_1, direction_2 (str): The directions defining the edge.

        Returns:
        Edge: The edge along the specified directions.

        Example:
            >>> seat = shape.get("legs/leg_front_right")
            >>> back_right_seat_edge = seat.edge("back", "right")
        """
        
## Edits
type_operand = Union[Part, Face, Edge]

class Translate:
    """
    Translate a geometric entity (part, face, or edge) in a given direction.

    Parameters:
    operand (Part|Face|Edge): The geometric entity (part, face, or edge) to be translated.
    direction (Vector): The vector specifying the direction and magnitude of the translation.
    amount (sp.Expr): The symbolic amount of translation as an expression of symbol "X". The user will later assign a value to this symbol to perform the translation.

    Examples:
        # Raise the seat of a chair.
        amount = sp.Symbol("X")
        seat = shape.get("seat_surface")
        edit = Translate(seat, UpUnitVector(), amount=amount)

        # Move the front legs of the chair backwards.
        amount = 0.5 * sp.Symbol("X")
        leg_front_left = shape.get("legs/leg_front_left")
        edit_1 = Translate(leg_front_left, BackUnitVector(), amount=amount)
        leg_front_right = shape.get("legs/leg_front_right")
        edit_2 = Translate(leg_front_right, BackUnitVector(), amount=amount)

        # Lower the backrest of the chair along the vertical bars in contact with it.
        amount = 2 * sp.Symbol("X")
        backrest = shape.get("backrest")
        back_vertical_bar = shape.get("back_surface_vertical_bar_left")
        edit = Translate(backrest, back_vertical_bar.direction("down"), amount=amount)

        # Translate a face to scale a part.
        # Make the seat thicker by translating its bottom face downwards.
        amount = sp.Symbol("X")
        seat = shape.get("seat_surface")
        edit = Translate(seat.face("down"), DownUnitVector(), amount=amount)

        # Stretch a part in one direction by translating a face.
        amount = sp.Symbol("X")
        bed_frame = shape.get("bed_frame")
        edit = Translate(bed_frame.face("right"), RightUnitVector(), amount=amount)

        # Shorten the front legs of the chair from the top.
        amount = 1.5 * sp.Symbol("X")
        leg_front_left = shape.get("legs/leg_front_left")
        edit = Translate(leg_front_left.face("up"), DownUnitVector(), amount=amount)

        # Move the base of a part: Move the base of the armrests inwards.
        amount = sp.Symbol("X")
        armrest_left = shape.get("armrest_left")
        edit = Translate(armrest_left.face("down"), RightUnitVector(), amount=amount)

        # Tilt the front of a chair's armrest downwards.
        amount = 2 * sp.Symbol("X")
        armrest_left = shape.get("armrest_left")
        edit = Translate(armrest_left.face("front"), DownUnitVector(), amount=amount)
        
        # Taper a part by translating an edge.
        # Extend only the top right side of the chair's back.
        amount = 0.2 * sp.Symbol("X")
        back = shape.get("back_surface")
        edit = Translate(back.edge("back", "right"), RightUnitVector(), amount=amount)

        # Shear a part by translating a face while keeping another part stationary.
        amount = 0.5 * sp.Symbol("X")
        # Move the base of the front left leg forward while keeping its attachment to the seat unchanged.
        leg_front_left = shape.get("legs/leg_front_left")
        edit = Translate(leg_front_left.face("down"), FrontUnitVector(), amount=amount)
    """

class Rotate:
    """
    Rotate a geometric entity (part, face, or edge) around a specified axis.

    Parameters:
    operand (Part|Face|Edge): The entity to be rotated.
    rotation_axis_origin (Vector): The origin point of the rotation axis.
    rotation_axis_direction (Vector): The direction vector of the rotation axis.
    amount (sp.Expr): The symbolic amount of rotation as an expression of symbol "X". The user will later assign a value to this symbol to perform this operation.

    Examples:
        # Rotate the back frame of the bed backwards
        amount = sp.Symbol("X")
        bed_back_frame = shape.get("bed_frame_back")
        bed_back_frame_bottom_center = bed_back_frame.face_center("bottom")
        # To rotate towards the back, the axis of rotation is the left unit vector.
        edit = Rotate(bed_back_frame, bed_back_frame_bottom_center, LeftUnitVector(), amount=amount)

        # Tilt the armrests of the chair downwards
        amount = 2 * sp.Symbol("X")
        armrest_left = chair.get("armrest_left")
        armrest_left_top_back_center = armrest_left.edge_center("up", "back")
        edit = Rotate(backrest, armrest_left_top_back_center, RgithUnitVector(), amount=amount)
    """

class Scale:
    """
    Scale a geometric entity in different dimensions, either expanding or contracting.

    This class allows for scaling along one, two, or three dimensions based on the number of axis direction vectors provided:
    - To scale along a single axis, provide axis_direction_1.
    - To scale along two axes, provide both axis_direction_1 and axis_direction_2.
    - To scale uniformly in all directions (i.e. volume scaling), do not provide any axis_direction vectors.

    Parameters:
    operand (Part|Face|Edge): The entity to be scaled.
    scaling_type (str): The type of scaling, either "expand" or "contract".
    scale_origin (Vector): The origin point of the scaling.
    axis_direction_1 (Vector, optional): A direction vector along the first axis of scaling.
    axis_direction_2 (Vector, optional): A direction vector along the second axis of scaling, used along with axis_direction_1 for a two-dimensional scaling. Must be perpendicular to axis_direction_1.
    amount (sp.Expr): The symbolic amount of scaling as an expression of symbol "X". The user will later assign a value to this symbol to perform this operation.

    Examples:
        # Scaling along a single axis

        # Lengthen the table top along the left-right direction
        amount = sp.Symbol("X")
        table_top = shape.get("table_top")
        table_top_center = table_top.center()
        edit = Scale(table_top, scaling_type="expand", scale_origin=table_top_center, 
                axis_direction_1=RightUnitVector(), amount=amount)

        # Reduce the width of the front of a lamp head.
        amount = 0.5 * sp.Symbol("X")
        lamp_head = shape.get("lamp_head")
        edit = Scale(lamp_head.face("front"), scaling_type="contract", scale_origin=lamp_head.center(), 
            axis_direction_1=RightUnitVector(), amount=amount)

        # Scaling along two axes

        # Bring the table legs closer together, given the table is in upright position.
        amount = 0.25 * sp.Symbol("X")
        table_legs = shape.get("legs")
        table_legs_center = table_legs.center()
        edit = Scale(table_legs, scaling_type="contract", scale_origin=table_legs_center, axis_direction_1=FrontUnitVector(), axis_direction_2=RightUnitVector(), amount=amount)

        # Scale down the armrests along both the back-front and up-down directions
        amount = 0.5 * sp.Symbol("X")
        armrest_left = shape.get("armrest_left")
        scale_origin = armrest_left.edge_center("down", "back")
        edit = Scale(armrest_left, scaling_type="contract", scale_origin=scale_origin, axis_direction_1=DownUnitVector(), axis_direction_2=BackUnitVector(),  amount=amount)

        # Uniform scaling in all directions

        # Shrink a lamp cover uniformly from its top
        amount = 0.1 * sp.Symbol("X")
        lamp_cover = shape.get("lamp_cover")
        scale_center = lamp_cover.face_center("up")
        edit = Scale(lamp_cover, scaling_type="contract", scale_origin=scale_center, amount=amount)

    Note:
    - Provide axis_direction_1 for single-axis scaling.
    - Provide both axis_direction_1 and axis_direction_2 for two-dimensional scaling.
    - Do not provide any axis_direction vectors for uniform scaling in all directions.
    """

class Shear:
    """
    Shear a geometric entity in a specified direction along a defined plane.
    If the shear plane origin is the center of the entity, the shear will effect the entity in both the directions along the shear direction.

    Parameters:
    operand (Part|Face|Edge): The entity to be sheared.
    shear_direction (Vector): The direction vector of the shearing.
    shear_plane_origin (Vector): The origin point of the shear plane.
    shear_plane_direction (Vector): The direction vector of the shear plane.
    amount (sp.Expr): The symbolic amount of shearing as an expression of symbol "X". The user will later assign a value to this symbol to perform the shear.

    Examples:
        # Shear the backrest of a chair, bringing the top forward and the bottom backward.
        amount = sp.Symbol("X")
        backrest = shape.get("backrest")
        backrest_center = backrest.center()
        edit = Shear(backrest, RightUnitVector(), backrest_center, UpUnitVector(), amount=amount)

        # Shear the bed to make it slanted vertically
        amount = 2 * sp.Symbol("X")
        bed_frame = shape.get("bed_frame")
        frame_side_center = bed_frame.get_edge_center("left", "down")
        edit = Shear(bed_frame, RightUnitVector(), frame_side_center, FrontUnitVector(), amount=amount)
    """

class SymGroupEdit:
    """
    Modify the symmetry group of a geometric entity.

    Parameters:
    operand (Part|Face|Edge): The entity with the symmetry group.
    change_type (str): Type of change - 'count' or 'delta' between symmetry group elements.
    extend_from (str): Specifies expansion method for translation symmetry. Options: "start", "end", "center", "keep_fixed". When extend_from is specified as "start", the symmetry group extends towards the direction of the first instance in the symmetry group. When extend_from is specified as "end", the symmetry group extends towards the direction of the last instance in the symmetry group. When extend_from is specified as "center", the symmetry group extends on both sides of the symmetry group. When extend_from is specified as "keep_fixed", the symmetry group remains of fixed size, while the number of instances change.
    amount (sp.Expr): The symbolic amount of shearing as an expression of symbol "X". The user will later assign a value to this symbol to perform the shear.

    Examples:
        # Change the number of legs in a swivel chair - given they are in a rotational symmetry group.
        amount = sp.Symbol("X")
        legs = shape.get("legs")
        edit = SymGroupEdit(legs, "count", "keep_fixed", amount=amount)

        # Adjust the spacing between the vertical bars in the back of a chair - given they are in a translational symmetry group.
        amount = sp.Symbol("X")
        back_vertical_slats = shape.get("back_surface_vertical_bars")
        edit = SymGroupEdit(back_vertical_slats, "delta", "center", amount=amount)
    """

```
