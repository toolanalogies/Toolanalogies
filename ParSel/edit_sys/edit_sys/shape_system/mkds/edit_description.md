```python

class DirectionalTranslate(Edit):
    """Performs a translation edit operation on a part in a specified direction. 
    This edit cannot be used to scale an object. Use scaling edits for such operations.

    Attributes:
        operand (Part): The part to be translated.
        direction (sp.Matrix): The direction vector of the translation.
        amount (symbolic): The magnitude of the edit operation.
    """

class DirectionalRotate(Edit):
    """Performs a rotational edit operation on a part around a specified axis.

    Attributes:
        operand (Part): The part to be rotated.
        rotation_origin (sp.Matrix): The point about which the rotation occurs.
        rotation_axis (sp.Matrix): The axis of rotation.
        amount (symbolic): The magnitude of the edit operation.
    """

class IsotropicScale(Edit):
    """Performs an isotropic scaling operation on a part about the `origin`, scaling equally in all directions. Use only when a part has to be uniformly scaled along all three axis. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        amount (symbolic): The magnitude of the edit operation.
    """

class DirectionalConstantScale(Edit):
    """Scales a part along a specified direction and its opposite, with no scaling in the plane perpendicular to this direction. Ideal for scaling in one dimension, such as lengthening or shortening a part without altering its thickness. Scaling is performed about the `origin`. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        direction (sp.Matrix): The axis along which the scaling is performed.
        amount (symbolic): The magnitude of the edit operation.
    """

class DirectionalVariableScale(Edit):
    """Performs a variable directional scaling operation on a part along a specified axis. The part is scaled along the specified direction (and its opposite direction), while having no scaling in the plane perpendicular to the specified direction. More importantly, the scaling amount increases as we move away from the origin along the variation direction. that the scaling amount is zero when the distance from the origin is zero along the variation direction. Scaling is performed about the `origin`. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        direction (sp.Matrix): The axis along which the scaling is performed.
        variation_direction (sp.Matrix): The direction along which the scaling factor varies.
        amount (symbolic): The magnitude of the edit operation.
    """

class PlanarConstantScale(Edit):
    """Performs a constant planar scaling operation on a part along a specified plane.
    The part is scaled along the specified plane, while having no scaling in the direction perpendicular to the plane. PlanarScale is more appropriate than DirectionalScale when a part has to be thickened, thinned or radially scaled. Scaling is performed about the `origin`. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        plane_normal (sp.Matrix): The normal vector of the scaling plane.
        amount (symbolic): The magnitude of the edit operation.
    """

class PlanarVariableScale(Edit):
    """Performs a variable planar scaling operation on a part along a specified plane.
    The part is scaled along the specified plane, while having no scaling in the direction perpendicular to the plane. PlanarScale is more appropriate than DirectionalScale when a part has to be thickened/thinned or radially scaled.
    More importantly, the scaling amount increases as we move away from the origin along the perpendicular. Note that the scaling amount is zero when the distance from the origin is zero along the perpendicular. Scaling is performed about the `origin`. Note that scale factor is `1 + amount`.

    Attributes:
        operand (Part): The part to be scaled.
        origin (sp.Matrix): The point about which the scaling occurs. Region near the origin is scaled less than the region far from the origin.
        plane_normal (sp.Matrix): The normal vector of the scaling plane.
        amount (symbolic): The magnitude of the edit operation.
    """

class Shear(Edit):
    """Performs a shear edit operation on a part along a specified plane. The part is sheared along the specified plane, with no shearing at the origin. Typically, the `shear_plane_origin` should be the point where we want it to stay attached. Shear is more appropriate than DirectionalRotate when the size of the part along the `shear_plane_normal` must be preserved. 

    Attributes:
        operand (Part): The part to be sheared.
        shear_plane_origin (sp.Matrix): The origin of the shear plane.
        shear_plane_normal (sp.Matrix): The normal vector of the shear plane.
        shear_direction (sp.Matrix): The direction of the shear.
        amount (symbolic): The magnitude of the edit operation.
    """
```