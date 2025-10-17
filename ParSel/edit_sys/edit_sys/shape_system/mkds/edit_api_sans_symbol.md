```python

# Functions to return unit vectors in various directions
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
    """Class for defining a part of a Shape Hierarchy, which can contain sub-parts."""

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
            >>> # Get the front right leg of a chair
            >>> leg_part = shape.get("front_right_leg")
        """

    def get_center(self):
        """
        Returns the center of the bound_geom.

        Example:
            >>> # Get the center of the seat of a chair
            >>> seat = shape.get("seat")
            >>> seat_center = seat.get_center()
        """
    
    def get_face_center(self, direction):
        """
        Returns the center of the bound_geom face in the specified direction.

        :param direction: String specifying the direction ('front', 'back', etc.)
        
        Example:
            >>> # Get the center of the front face of the seat of a chair
            >>> seat = shape.get("seat")
            >>> seat_front_center = seat.get_face_center("front")

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

    def get_corner(self, direction_1, direction_2, direction_3):
        """
        Returns the corner of the bound_geom formed by the intersection of three specified directions.

        :param direction_1, direction_2, direction_3: Strings specifying the directions ('front', 'back', etc.)

        Example:
            >>> # Get the corner formed by the intersection of the front, right and up faces of the seat of a chair
            >>> seat = shape.get("seat")
            >>> seat_corner = seat.get_corner("front", "right", "up")
        """

# For creating edits
class Edit:
    """The base class for all edit operations.

    Attributes:
        operand (Part): The part to be edited.
        amount (symbolic, optional): The magnitude of the edit operation. Defaults to a symbolic variable 'X' if not specified.
    """

class DirectionalTranslate(Edit):
    """Performs a translation edit operation on a part in a specified direction. 
    This edit cannot be used to scale an object. Use scaling edits for such operations.

    Attributes:
        operand (Part): The part to be translated.
        direction (sp.Matrix): The direction vector of the translation.
        amount (symbolic, optional): The magnitude of the edit operation. Defaults to a symbolic variable 'X' if not specified.

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
        amount (symbolic, optional): The magnitude of the edit operation. Defaults to a symbolic variable 'X' if not specified.

    Example:
        >>> # Rotating the front right leg of a chair around its upper face center
        >>> front_right_leg = shape.get("front_right_leg")
        >>> rotation_origin = front_right_leg.get_face_center("up")
        >>> rotation_axis = LeftUnitVector()
        >>> new_edit = DirectionalRotate(front_right_leg, rotation_origin, rotation_axis)
    """

class IsotropicScale(Edit):
    """Performs an isotropic scaling operation on a part about the `origin`, scaling equally in all directions. Use only when a part has to be uniformly scaled along all three axis. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        amount (symbolic, optional): The magnitude of the edit operation. Defaults to a symbolic variable 'X' if not specified.

    Example:
        >>> # Scaling the seat of a chair isotropically from its center
        >>> seat_part = shape.get("seat")
        >>> center = seat_part.get_center()
        >>> new_edit = IsotropicScale(seat_part, center)
    """

class DirectionalConstantScale(Edit):
    """Scales a part along a specified direction and its opposite, with no scaling in the plane perpendicular to this direction. Ideal for scaling in one dimension, such as lengthening or shortening a part without altering its thickness. Scaling is performed about the `origin`. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        direction (sp.Matrix): The axis along which the scaling is performed.
        amount (symbolic, optional): The magnitude of the edit operation. Defaults to a symbolic variable 'X' if not specified.

    Example:
        >>> # Scaling the back of a chair along the front axis
        >>> back_part = shape.get("back")
        >>> center = back_part.get_center()
        >>> direction = FrontUnitVector()
        >>> new_edit = DirectionalConstantScale(back_part, center, direction)
    """

class DirectionalVariableScale(Edit):
    """Performs a variable directional scaling operation on a part along a specified axis. The part is scaled along the specified direction (and its opposite direction), while having no scaling in the plane perpendicular to the specified direction. More importantly, the scaling amount increases as we move away from the origin along the variation direction. that the scaling amount is zero when the distance from the origin is zero along the variation direction. Scaling is performed about the `origin`. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        direction (sp.Matrix): The axis along which the scaling is performed.
        variation_direction (sp.Matrix): The direction along which the scaling factor varies.
        amount (symbolic, optional): The magnitude of the edit operation. Defaults to a symbolic variable 'X' if not specified.

    Example:
        >>> # Stretching the seat along the left-right direction in the front, while keeping its back fixed.
        >>> seat_part = shape.get("seat")
        >>> scale_origin = seat_part.get_face_center("back")
        >>> right_direction = RightUnitVector()
        >>> front_direction = FrontUnitVector()
        >>> new_edit = DirectionalVariableScale(back_part, scale_origin, right_direction, front_direction)
    """

class PlanarConstantScale(Edit):
    """Performs a constant planar scaling operation on a part along a specified plane.
    The part is scaled along the specified plane, while having no scaling in the direction perpendicular to the plane. PlanarScale is more appropriate than DirectionalScale when a part has to be thickened, thinned or radially scaled. Scaling is performed about the `origin`. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        plane_normal (sp.Matrix): The normal vector of the scaling plane.
        amount (symbolic, optional): The magnitude of the edit operation. Defaults to a symbolic variable 'X' if not specified.

    Example:
        >>> # Thickening the front right leg of a given chair shape.
        >>> leg_part = shape.get("front_right_leg")
        >>> scale_origin = leg_part.get_center()
        >>> plane_normal = UpUnitVector()
        >>> new_edit = PlanarConstantScale(leg_part, scale_origin, plane_normal)
    """

class PlanarVariableScale(Edit):
    """Performs a variable planar scaling operation on a part along a specified plane.
    The part is scaled along the specified plane, while having no scaling in the direction perpendicular to the plane. PlanarScale is more appropriate than DirectionalScale when a part has to be thickened/thinned or radially scaled.
    More importantly, the scaling amount increases as we move away from the origin along the perpendicular. Note that the scaling amount is zero when the distance from the origin is zero along the perpendicular. Scaling is performed about the `origin`. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        plane_normal (sp.Matrix): The normal vector of the scaling plane.
        amount (symbolic, optional): The magnitude of the edit operation. Defaults to a symbolic variable 'X' if not specified.

    Example:
        >>> # Variable planar scaling of the front right leg of a chair
        >>> leg_part = shape.get("front_right_leg")
        >>> scale_origin = leg_part.get_face_center("up")
        >>> plane_normal = UpUnitVector()
        >>> new_edit = PlanarVariableScale(leg_part, scale_origin, plane_normal)
    """

class Shear(Edit):
    """Performs a shear edit operation on a part along a specified plane. The part is sheared along the specified plane, with no shearing at the origin. Typically, the `shear_plane_origin` should be the point where we want it to stay attached. Shear is more appropriate than DirectionalRotate when the size of the part along the `shear_plane_normal` must be preserved. 

    Attributes:
        operand (Part): The part to be sheared.
        shear_plane_origin (sp.Matrix): The origin of the shear plane.
        shear_plane_normal (sp.Matrix): The normal vector of the shear plane.
        shear_direction (sp.Matrix): The direction of the shear.
        amount (symbolic, optional): The magnitude of the edit operation. Defaults to a symbolic variable 'X' if not specified.

    Example:
        >>> # Shearing the back of a chair towards the back direction
        >>> back_part = shape.get("back")
        >>> # set plane origin at the point where we want it to stay attached.
        >>> shear_plane_origin = back_part.get_face_center("down")
        >>> shear_plane_normal = UpUnitVector()
        >>> shear_direction = BackUnitVector()
        >>> new_edit = Shear(back_part, shear_plane_origin, shear_plane_normal, shear_direction)
    """

class KeepFixed(Edit):
    """Specifies that a part must remain unedited.
    This must be used for parts that must be kept unedited explicitly.

    Attributes:
        operand (Part): The part to remain fixed.

    Example:
        >>> # Keeping the armrests of a chair fixed
        >>> armrest_part = shape.get("armrest")
        >>> new_edit = KeepFixed(armrest_part)
    """

# IMPORTANT: For all scaling operations, the scale factor is as `1 + amount`. Therefore, to shorten a part, specify amount = -sp.Symbol("X").

# IMPORTANT: For all scaling operations, region around the `origin` will scale the least. Therefore, to scale a part in one direction only, set the `origin` to be the center of the face opposite to that direction.
```