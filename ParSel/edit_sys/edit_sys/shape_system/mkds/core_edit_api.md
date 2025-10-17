```python

import sympy as sp

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

# For creating edits
class Edit:
    """The base class for all edit operations.
    """
    ...

class DirectionalTranslate(Edit):
    """Performs a translation edit operation on a part in a specified direction. This edit cannot be used on a sub-part to scale an object. Use `DirectionalConstantScale` for edits of that nature.

    Attributes:
        operand (Part): The part to be translated.
        direction (sp.Matrix): The direction vector of the translation.
        amount (symbolic, optional): The magnitude of the translation. Defaults to sympy.Symbol("X").

    Example:
        >>> # Translating the seat of a chair to the right
        >>> seat_part = shape.get("seat")
        >>> direction = RightUnitVector()
        >>> new_edit = DirectionalTranslate(seat_part, direction)
    """

class DirectionalRotate(Edit):
    """Performs a rotational edit operation on a part around a specified axis.

    Attributes:
        operand (Part): The part to be rotated.
        rotation_origin (sp.Matrix): The point about which the rotation occurs.
        rotation_axis (sp.Matrix): The axis of rotation.
        amount (symbolic, optional): The angle of rotation. Defaults to sympy.Symbol("X").

    Example:
        >>> # Rotating the front right leg of a chair around its upper face center
        >>> front_right_leg = shape.get("front_right_leg")
        >>> rotation_origin = front_right_leg.face_center("up")
        >>> rotation_axis = LeftUnitVector()
        >>> new_edit = DirectionalRotate(front_right_leg, rotation_origin, rotation_axis)
    """

class IsotropicScale(Edit):
    """Performs an isotropic scaling operation on a part, scaling equally in all directions. Use only when a part has to be uniformly scaled along all three axis. Note that scale factor is is as `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs.
        amount (symbolic, optional): The scaling factor. Defaults to sympy.Symbol("X").

    Example:
        >>> # Scaling the seat of a chair isotropically from its center
        >>> seat_part = shape.get("seat")
        >>> center = seat_part.center()
        >>> new_edit = IsotropicScale(seat_part, center)
    """

class DirectionalConstantScale(Edit):
    """Scales a part along a specified direction and its opposite, with no scaling in the plane perpendicular to this direction. Ideal for scaling in one dimension, such as lengthening or shortening a part without altering its thickness.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs.
        direction (sp.Matrix): The axis along which the scaling is performed.
        amount (symbolic, optional): The scaling factor. Defaults to sympy.Symbol("X").

    Example:
        >>> # Scaling the back of a chair along the front axis
        >>> back_part = shape.get("back")
        >>> center = back_part.center()
        >>> direction = FrontUnitVector()
        >>> new_edit = DirectionalConstantScale(back_part, center, direction)
        >>> # Scaling the back right leg of a chair such that its top face is contracted.
        >>> leg_part = shape.get("back_right_leg")
        >>> # The origin should center of the bottom face, so that it remains fixed.
        >>> origin = back_part.face_center("down")
        >>> direction = UpUnitVector()
        >>> new_edit = DirectionalConstantScale(leg_part, origin, direction, amount=-sp.Symbol("X"))
    """

class DirectionalVariableScale(Edit):
    """Performs a variable directional scaling operation on a part along a specified axis. The part is scaled along the specified direction (and its opposite direction), while having no scaling in the plane perpendicular to the specified direction. More importantly, the scaling amount increases as we move away from the origin along the variation direction. The scaling amount is zero when the distance from the origin is zero along the variation direction.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs.
        direction (sp.Matrix): The axis along which the scaling is performed.
        variation_direction (sp.Matrix): The direction along which the scaling factor varies.
        amount (symbolic, optional): The base scaling factor. Defaults to sympy.Symbol("X").

    Example:
        >>> # Stretching the seat along the left-right direction in the front, while keeping its back fixed.
        >>> seat_part = shape.get("seat")
        >>> scale_origin = seat_part.face_center("back")
        >>> right_direction = RightUnitVector()
        >>> front_direction = FrontUnitVector()
        >>> new_edit = DirectionalVariableScale(back_part, scale_origin, right_direction, front_direction)
    """

class PlanarConstantScale(Edit):
    """Performs a constant planar scaling operation on a part along a specified plane.
    The part is scaled along the specified plane, while having no scaling in the direction perpendicular to the plane. PlanarScale is more appropriate than DirectionalScale when a part has to be thickened, thinned or radially scaled.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs.
        plane_normal (sp.Matrix): The normal vector of the scaling plane.
        amount (symbolic, optional): The scaling factor. Defaults to sympy.Symbol("X").

    Example:
        >>> # Thickening the front right leg of a given chair shape.
        >>> leg_part = shape.get("front_right_leg")
        >>> scale_origin = leg_part.center()
        >>> plane_normal = UpUnitVector()
        >>> new_edit = PlanarConstantScale(leg_part, scale_origin, plane_normal)
    """

class PlanarVariableScale(Edit):
    """Performs a variable planar scaling operation on a part along a specified plane.
    The part is scaled along the specified plane, while having no scaling in the direction perpendicular to the plane. PlanarScale is more appropriate than DirectionalScale when a part has to be thickened/thinned or radially scaled.
    More importantly, the scaling amount increases as we move away from the origin along the perpendicular. Note that the scaling amount is zero when the distance from the origin is zero along the perpendicular.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs.
        plane_normal (sp.Matrix): The normal vector of the scaling plane.
        amount (symbolic, optional): The scaling factor. Defaults to sympy.Symbol("X").

    Example:
        >>> # Variable planar scaling of the front right leg of a chair
        >>> leg_part = shape.get("front_right_leg")
        >>> scale_origin = leg_part.face_center("up")
        >>> plane_normal = UpUnitVector()
        >>> new_edit = PlanarVariableScale(leg_part, scale_origin, plane_normal)
    """

class Shear(Edit):
    """Performs a shear edit operation on a part along a specified plane. The shear origin is the point on the shear plane that remains fixed during the shear operation.

    Attributes:
        operand (Part): The part to be sheared.
        shear_plane_origin (sp.Matrix): The origin of the shear plane.
        shear_plane_normal (sp.Matrix): The normal vector of the shear plane. This should be perpendicular to the shear direction.
        shear_direction (sp.Matrix): The direction of the shear.
        amount (symbolic, optional): The magnitude of the shear. Defaults to sympy.Symbol("X").

    Example:
        >>> # Shearing the top of a chair towards the back direction
        >>> back_part = shape.get("back")
        >>> shear_plane_origin = back_part.face_center("down")
        >>> shear_plane_normal = UpUnitVector()
        >>> shear_direction = BackUnitVector()
        >>> new_edit = Shear(back_part, shear_plane_origin, shear_plane_normal, shear_direction)
        >>> # Shearing the front of the seat downwards while keeping the back fixed.
        >>> seat_part = shape.get("seat")
        >>> shear_plane_origin = seat_part.face_center("back")
        >>> shear_plane_normal = FrontUnitVector()
        >>> shear_direction = DownUnitVector()
        >>> new_edit = Shear(seat_part, shear_plane_origin, shear_plane_normal, shear_direction)
        
    """
```