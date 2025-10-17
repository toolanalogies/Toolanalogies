```python

from typing import Union

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

DIRECTION_STRINGS = ["front", "back", "left", "right", "up", "down", "messi0"]

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

    Examples:
        # Example 1: Raise the seat of a chair.
        seat = shape.get("seat_surface")
        edit = Translate(seat, UpUnitVector())

        # Example 2: Move the front legs of the chair backwards.
        leg_front_left = shape.get("legs/leg_front_left")
        edit_1 = Translate(leg_front_left, BackUnitVector())
        leg_front_right = shape.get("legs/leg_front_right")
        edit_2 = Translate(leg_front_right, BackUnitVector())

        # Example 3: Translate a face to scale a part.
        # Make the seat thicker by translating its bottom face downwards.
        seat = shape.get("seat_surface")
        edit = Translate(seat.face("down"), DownUnitVector())

        # Example 4: Stretch a part in one direction by translating a face.
        bed_frame = shape.get("bed_frame")
        edit = Translate(bed_frame.face("right"), RightUnitVector())

        # Example 5: Taper a part by translating an edge.
        # Extend only the top right side of the chair's back.
        back = shape.get("back_surface")
        edit = Translate(back.edge("back", "right"), RightUnitVector())

        # Example 6: Shear a part by translating a face while keeping another part stationary.
        # Move the base of the front left leg forward while keeping its attachment to the seat unchanged.
        leg_front_left = shape.get("legs/leg_front_left")
        edit = Translate(leg_front_left.face("down"), RightUnitVector())
    """

class Rotate:
    """
    Rotate a geometric entity (part, face, or edge) around a specified axis.

    Parameters:
    operand (Part|Face|Edge): The entity to be rotated.
    rotation_axis_origin (Vector): The origin point of the rotation axis.
    rotation_axis_direction (Vector): The direction vector of the rotation axis.

    Examples:
        # Rotate the back frame of the bed backwards
        bed_back_frame = shape.get("bed_frame_back")
        bed_back_frame_bottom_center = bed_back_frame.face_center("bottom")
        # To rotate towards the back, the axis of rotation is the left unit vector.
        edit = Rotate(bed_back_frame, bed_back_frame_bottom_center, LeftUnitVector())

        # Tilt the armrests of the chair upwards
        armrest_left = chair.get("armrest_left")
        armrest_left_top_back_center = armrest_left.edge_center("up", "back")
        edit = Rotate(backrest, armrest_left_top_back_center, LeftUnitVector())
    """

class AxisScale:
    """
    Scale a geometric entity along an axis, either expanding or contracting.

    Parameters:
    operand (Part|Face|Edge): The entity to be scaled.
    axis_origin (Vector): The origin point of the scaling axis.
    axis_direction (Vector): The direction vector of the scaling axis.
    scaling_type (str): The type of scaling, either "expand" or "contract".

    Examples:
        # Lengthen the table top along the left-right direction
        table_top = shape.get("table_top")
        table_top_center = table_top.center()
        edit = AxisScale(table_top, table_top_center, table_top.direction("right"), "expand")

        # Make the front of a lamp head wider.
        lamp_head = shape.get("lamp_head")
        edit = AxisScale(lamp_head.face("front"), lamp_head.center(), lamp_head.direction("right"), "contract")
        # Note: This edit is equivalent to translating the top face of the lamp stand downwards.

        # Expand only the top face of the back (taper the back) of a chair
        back = shape.get("back_surface")
        edit = AxisScale(back.face("up"), back.face_center("down"), back.direction("right"), "expand")

        # Widen the backest of the chair.
        backrest = shape.get("backrest")
        edit = AxisScale(backrest, backrest.center(), backrest.direction("right"), "expand")

    """

class PlaneScale:
    """
    Scale a geometric entity along a plane, either expanding or contracting. It scales the entity along two axis perpendicular to the plane normal.
    This edit should not be used for scaling along a single axis. AxisScale should be used for such cases.

    Parameters:
    operand (Part|Face|Edge): The entity to be scaled.
    scale_origin (Vector): The origin point of the scaling.
    plane_normal (Vector): The normal vector of the scaling plane.
    scaling_type (str): The type of scaling, either "expand" or "contract".

    Examples:

        # Bring the table legs closer together
        table_legs = shape.get("legs")
        table_legs_center = table_legs.center()
        edit = PlaneScale(table_legs, table_legs_center, FrontUnitVector(), "contract")

        # Scale down the armests along both the back-front direction and the up-down direction
        armrest_left = shape.get("armrest_left")
        scale_origin = armrest_left.edge_center("down", "back")
        plane_normal_direction = armrest_left.direction("right")
        edit = PlaneScale(armrest_left, scale_origin, plane_normal_direction, "contract")

        # Make the base of the bed larger only on the bottom side
        bed_base = shape.get("bed_base")
        bed_base_top_center = bed_base.face_center("up")
        edit = PlaneScale(bed_base, bed_base_top_center, UpUnitVector(), "expand")
    """

class VolumeScale:
    """
    Scale a geometric entity uniformly in all directions.

    Parameters:
    operand (Part|Face|Edge): The entity to be scaled.
    scale_origin (Vector): The center point of the scaling.
    scaling_type (str): The type of scaling, either "expand" or "contract".

    Examples:
        # Make the back vertical bars of the chair smaller.
        back_bars = shape.get("back_surface_vertical_bars")
        back_bars_bottom_center = back_bars.face_center("down")
        edit = VolumeScale(back_bars, back_bars_bottom_center, "contract")

        # Shrink a lamp cover uniformly
        lamp_cover = shape.get("lamp_cover")
        lamp_cover_center = lamp_cover.center()
        edit = VolumeScale(lamp_cover, lamp_cover_center, "contract")
    """

class Shear:
    """
    Shear a geometric entity in a specified direction along a defined plane.
    If the shear plane origin is the center of the entity, the shear will effect the entity in both the directions along the shear direction.

    Parameters:
    operand (Part|Face|Edge): The entity to be sheared.
    shear_plane_origin (Vector): The origin point of the shear plane.
    shear_plane_direction (Vector): The direction vector of the shear plane.
    shear_direction (Vector): The direction vector of the shearing.

    Examples:
        # Shear the backrest of a chair, bringing the top forward and the bottom backward.
        backrest = shape.get("backrest")
        backrest_center = backrest.center()
        edit = Shear(backrest, backrest_center, UpUnitVector(), RightUnitVector())

        # Shear the bed to make it slanted vertically
        bed_frame = shape.get("bed_frame")
        frame_side_center = bed_frame.get_edge_center("left", "down")
        edit = Shear(bed_frame, frame_side_center, FrontUnitVector(), RightUnitVector())
    """

class SymGroupEdit:
    """
    Modify the symmetry group of a geometric entity.

    Parameters:
    operand (Part|Face|Edge): The entity with the symmetry group.
    change_type (str): Type of change - 'count' or 'delta' between symmetry group elements.
    extend_from (str): Specifies expansion method for translation symmetry. Options: "start", "end", "center", "keep_fixed". When extend_from is specified as "start", the symmetry group extends towards the direction of the first instance in the symmetry group. When extend_from is specified as "end", the symmetry group extends towards the direction of the last instance in the symmetry group. When extend_from is specified as "center", the symmetry group extends on both sides of the symmetry group. When extend_from is specified as "keep_fixed", the symmetry group remains of fixed size, while the number of instances change.

    Examples:
        # Change the number of legs in a swivel chair - given they are in a rotational symmetry group.
        legs = shape.get("legs")
        edit = SymGroupEdit(legs, "count", "keep_fixed")

        # Adjust the spacing between the vertical bars in the back of a chair - given they are in a translational symmetry group.
        back_vertical_slats = shape.get("back_surface_vertical_bars")
        edit = SymGroupEdit(back_vertical_slats, "delta", "center")
    """

```
