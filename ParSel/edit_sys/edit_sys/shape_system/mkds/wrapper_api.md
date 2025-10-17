```python
import sympy as sp
from typing import List
from .shape_atoms import Part

DIRECTIONS = ["front", "back", "left", "right", "top", "bottom", "messi8"]

# Functions to return unit vectors in different directions
def FrontUnitVector():
    ...

def BackUnitVector():
    ...

def LeftUnitVector():
    ...

def RightUnitVector():
    ...

def UpUnitVector():
    ...

def DownUnitVector():
    ...

class Part():
    """Class for defining a part of a Shape, which can contain sub-parts."""

    def __init__(self, label: str, bound_geom: Primitive, sub_parts: set = None):
        """
        Initialize a part of a shape.

        :param label: String uniquely identifying this part.
        :param bound_geom: The geometric primitive defining the shape of this part. Typically a cuboid
        :param sub_parts: A set of sub-parts contained within this part (optional).
        """

    def get(self, label):
        """
        Returns a sub-part with the specified label.

        :param label: String label of the sub-part to be returned.

        Example:
            >>> shape = Part(...)
            >>> leg_part = shape.get("front_right_leg")
        """
        part_to_traverse = [self]
        while part_to_traverse:
            part = part_to_traverse.pop()
            if part.label == label:
                return part
            else:
                part_to_traverse.extend(part.sub_parts)
        raise ValueError(f"Part with label {label} not found.")

    def center(self):
        """
        Returns the center of the bound_geom.

        Example:
            >>> # Get the center of the seat of a chair
            >>> seat = shape.get("seat")
            >>> seat_center = seat.center()
        """

    
    def face_center(self, direction):
        """
        Returns the center of the bound_geom face in the specified direction.

        :param direction: String specifying the direction ('front', 'back', etc.)
        
        Example:
            >>> # Get the center of the front face of the seat of a chair
            >>> seat = shape.get("seat")
            >>> seat_front_center = seat.face_center("front")

        """

    def edge_center(self, direction_1, direction_2):
        """
        Returns the center of the bound_geom face in the specified direction.

        :param direction: String specifying the direction ('front', 'back', etc.)
        
        Example:
            >>> # Get the center of the back right edge of the seat
            >>> seat = shape.get("seat")
            >>> seat_front_center = seat.edge_center("back", "right")

        """

    def corner(self, direction_1, direction_2, direction_3):
        """
        Returns the corner of the bound_geom formed by the intersection of three specified directions.

        :param direction_1, direction_2, direction_3: Strings specifying the directions ('front', 'back', etc.)

        Example:
            >>> # Get the corner formed by the intersection of the front, right and up faces of the seat of a chair
            >>> seat = shape.get("seat")
            >>> seat_corner = seat.corner("front", "right", "up")
        """

    def get_direction(self, direction):
        """
        Returns the direction vector from the center to a bound_geom face.

        :param direction: String specifying the direction ('front', 'back', etc.)

        Example:
            >>> # Get the direction vector from the center to the front face of the seat of a chair
            >>> seat = shape.get("seat")
            >>> seat_front_direction = seat.get_direction("front")
        """
    #TODO Add part
    def axis(self):
        """
        Returns the first principal axis of the bound_geom.

        Example:
            >>> # Get the axis of the seat of a chair
            >>> seat = shape.get("seat")
            >>> seat_axis = seat.axis()
        """

## Edits

def translate(part:Part, direction:str, amount:sp.Expr):
    """
    Translates a shape part in a specified direction. This edit cannot be used on a sub-part to scale an object. Use `multidirectional_scale`, `multidirectional_face_scale` or `uniform_scale` for such edits.

    Parameters:
        part (Part): The shape part to be translated.
        direction (str): Key representing the direction of translation. Must be in DIRECTIONS.
        amount (symbolic, optional): The scaling factor. Defaults to sympy.Symbol("X").
    Example:
        >>> # Translate the seat of a chair to the right
        >>> seat_part = shape.get("seat")
        >>> new_edit = translate(part=seat_part, direction="right")
    """

def rotate(part: Part, rotation_origin: sp.Matrix, rotation_axis_direction: str, amount: sp.Expr):
    """
    Rotates a geometric part around a specified axis.

    Parameters:
        part (Part): The geometric part to be rotated.
        rotation_origin (sp.Matrix): The origin point of the rotation axis.
        rotation_axis_direction (str): Key for the direction of the rotation axis. Must be in DIRECTIONS.
        amount (symbolic, optional): The magnitude of rotation in degrees. Defaults to sympy.Symbol("X").
    Example:
        >>> # Rotate the front right leg of a chair from its top towards the front.
        >>> front_right_leg = shape.get("front_right_leg")
        >>> # The origin should be the center of the top face, so that it remains fixed there.
        >>> rotation_origin = front_right_leg.face_center("up")
        >>> # The axis will face the left direction since the rotation is towards the front.
        >>> new_edit = rotate(part=front_right_leg, rotation_origin=rotation_origin, rotation_axis_direction="left")
    """

def uniform_scale(part: Part, origin: sp.Matrix, amount: sp.Expr):
    """
    Uniformly scales a geometric part from a specified origin. Note that the scale factor is `1 + amount`.

    Parameters:
        part (Part): The geometric part to be scaled.
        origin (sp.Matrix): The origin point for scaling.
        amount (symbolic, optional): Scaling factor. Defaults to sympy.Symbol("X").

    Example:
        >>> # shrinking the seat of a chair isotropically from its center
        >>> seat_part = shape.get("seat")
        >>> center = seat_part.center()
        >>> # To reduce the size, keep the amount negative.
        >>> amount = -sp.Symbol("X")
        >>> new_edit = uniform_scale(seat_part, origin=center, amount=amount)
    """

def multidirectional_scale(part: Part, directions: List[str], amount: sp.Expr):
    """
    Scales a part in the give list of directions. Supports scaling in one to six directions.  Note that the scale factor is `1 + amount`. Ideal edit for lengthening, shortening etc.

    Parameters:
        part (Part): The geometric part to be scaled.
        directions (list of str): Keys representing scaling directions. Each must be in DIRECTIONS.
        amount (symbolic, optional): Scaling factor or factors. Defaults to sympy.Symbol("X").
    Examples:
        >>> # Scaling the chair-back to be thicker by extending its back
        >>> back_part = shape.get("back")
        >>> new_edit = multidirectional_scale(back_part, directions=["back"])
        >>> # Scaling the chair-back to be thicker by extending its back and front
        >>> new_edit = multidirectional_scale(back_part, directions=["back", "front"])
        >>> # Scaling the chair-back by extending its top and right faces.
        >>> new_edit = multidirectional_scale(back_part, directions=["top", "right"])
        >>> # Scaling the chair-back by extending it in left, top, and right directions while keeping the base fixed.
        >>> new_edit = multidirectional_scale(back_part, directions=["left", "top", "right"])
        >>> # Scaling the chair-front-leg radially from its center.
        >>> front_leg_part = shape.get("front_leg")
        >>> new_edit = multidirectional_scale(front_leg_part, directions=["front", "back", "left", "right"])
        >>> # Scaling the chair-front-leg by extending its top face.
        >>> new_edit = multidirectional_scale(front_leg_part, directions=["top"])
        >>> # scaling the seat of the chair in all directions while keeping its bottom fixed
        >>> seat_part = shape.get("seat")
        >>> new_edit = multidirectional_scale(seat_part, directions=["front", "back", "left", "right", "top"])
    """

def multidirectional_face_scale(part: Part, face: str, directions: List[str], amount: sp.Expr):
    """
    Performs multidirectional scaling of a geometric part by scaling a face. The face opposite to `face` is kept fixed. Other faces can remain fixed based on the `directions` variable. Note that the scale factor is `1 + amount`.

    Parameters:
        part (Part): The geometric part to be scaled.
        face (str): Key representing the face that will be scaled. It must be in DIRECTIONS.
        directions (list of str): Keys representing scaling directions. Each must be in DIRECTIONS.
        amount (symbolic, optional): Scaling factor or factors.  Defaults to sympy.Symbol("X").

    Example:
        >>> # Stretching the seat along the left-right direction in the front, while keeping its back fixed.
        >>> seat_part = shape.get("seat")
        >>> new_edit = multidirectional_face_scale(seat_part, face="front", directions=["left", "right"])
        >>> # Note: if it was to be stretched at both the front and the back, then use the multidirectional_scale function.
        >>> # To scale both front and back:
        >>> new_edit = multidirectional_scale(seat_part, directions=["left", "right"])
        >>> # Make the chair-front-left leg thicker in the bottom.
        >>> leg_part = shape.get("front_right_leg")
        >>> new_edit = multidirectional_face_scale(leg_part, face="bottom", directions=["front", "back", "left", "right"])
        >>> # Make the chair-back larger in the front, which remaining fixed at the bottom.
        >>> back_part = shape.get("back")
        >>> new_edit = multidirectional_face_scale(back_part, face="front", directions=["left", "up", "right"]) 
    """

def shear(part, face, shear_direction, amount=None):
    """
    Applies shear transformation to a geometric part, such that the `face` face of the part is shreaded towards the `shear_direction` by amount `amount`.

    Parameters:
        part (Part): The geometric part to be sheared.
        face (str): Key indicating the face normal to the shearing plane.
        shear_direction (str): Key indicating the direction of shear.
        amount (symbolic, optional): The magnitude of shear. Defaults to None.
    Example:
        >>> # Shearing the top of a chair towards the front direction
        >>> back_part = shape.get("back")
        >>> new_edit = shear(part=back_part, face="top", shear_direction="front")
        >>> # Shearing the front-left leg of the chair forward while keeping its top fixed.
        >>> seat_part = shape.get("seat")
        >>> new_edit = shear(part=seat_part, face="bottom", shear_direction="front")
    """

def bidirectional_shear(part, faces, shear_direction, amount=None):
    """
    Applies a bidirectional shear transformation to a geometric part. Face `face` is sheared towards the `shear_direction` by amount `amount`. The face opposite to `face` is sheared in the opposite direction by the same amount.

    Parameters:
        part (Part): The geometric part to be sheared.
        faces (list of str): Keys indicating the faces involved in the shearing.
        shear_direction (str): Key indicating the direction of shear.
        amount (symbolic, optional): The magnitude of shear. Defaults to None.
    Example:
        >>> # Shear the seat so that its raised in the front and lower in the back
        >>> seat_part = shape.get("seat")
        >>> new_edit = bidirectional_shear(part=seat_part, faces=["front", "back"], shear_direction="up")
    """
```