
from .edits import *
from .shape_atoms import *
from .relations import *
from .constants import DOT_THRESHOLD

def Translate(operand, direction, amount=None, amount_from_operand=None):
    identifier = None 
    if amount is None:
        if amount_from_operand is not None:
            amount = get_amount(amount_from_operand)
        else:
            amount = None
    else:
        amount = amount
    if isinstance(operand, Part):
        edit = PartTranslate(operand,
                             direction=direction,
                             amount=amount,
                             identifier=identifier)
    elif isinstance(operand, FaceFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = FaceTranslate(part,
                             direction=direction,
                             amount=amount,
                             identifier=identifier,
                             restrictor_name=restrictor_name)
    elif isinstance(operand, EdgeFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = EdgeTranslate(part,
                             direction=direction,
                             amount=amount,
                             identifier=identifier,
                             restrictor_name=restrictor_name)
    else:
        raise ValueError(f"Unsupported operand type: {type(operand)}")
    return edit


def Rotate(operand, rotation_axis_origin, rotation_axis_direction, amount=None, amount_from_operand=None):
    identifier = None 
    if amount is None:
        if amount_from_operand is not None:
            amount = get_amount(amount_from_operand)
        else:
            amount = None
    else:
        amount = amount
    if isinstance(operand, Part):
        edit = PartRotate(operand,
                          rotation_origin=rotation_axis_origin,
                          rotation_axis=rotation_axis_direction,
                          amount=amount,
                          identifier=identifier)
    elif isinstance(operand, FaceFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = FaceRotate(part,
                          rotation_origin=rotation_axis_origin,
                          rotation_axis=rotation_axis_direction,
                          amount=amount,
                          identifier=identifier,
                          restrictor_name=restrictor_name)
    elif isinstance(operand, EdgeFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = EdgeRotate(part,
                          rotation_origin=rotation_axis_origin,
                          rotation_axis=rotation_axis_direction,
                          amount=amount,
                          identifier=identifier,
                          restrictor_name=restrictor_name)
    else:
        raise ValueError(f"Unsupported operand type: {type(operand)}")
    return edit

def Scale(operand, scaling_type, scale_origin,  
          axis_direction_1=None, axis_direction_2=None, axis_direction_3=None, 
          amount=None, amount_from_operand=None, 
          *args, **kwargs):
    # see what type it is:

    if axis_direction_1 is None:
        scale = VolumeScale(operand, scaling_type, scale_origin, amount, amount_from_operand)
    elif axis_direction_2 is None:
        scale = AxisScale(operand, scaling_type, scale_origin, axis_direction_1, amount, amount_from_operand)
    elif axis_direction_3 is not None:
        if len(args) or len(kwargs):
            # could be plane or volume.
            normal_dir = axis_direction_1.cross(axis_direction_2)
            normal_dir = normal_dir.normalized()
            scale = PlaneScale(operand, scaling_type, scale_origin, normal_dir, amount, amount_from_operand)
        else:
            scale = VolumeScale(operand, scaling_type, scale_origin, amount, amount_from_operand)
    else:
        dot_prod = axis_direction_1.dot(axis_direction_2)
        if sp.Abs(dot_prod) > DOT_THRESHOLD:
            scale = AxisScale(operand, scaling_type, scale_origin, axis_direction_1, amount, amount_from_operand)
        else:
            normal_dir = axis_direction_1.cross(axis_direction_2)
            normal_dir = normal_dir.normalized()
            scale = PlaneScale(operand, scaling_type, scale_origin, normal_dir, amount, amount_from_operand)
    return scale


def AxisScale(operand, scaling_type, axis_origin, axis_direction, amount=None, amount_from_operand=None):
    identifier = None 
    if amount is None:
        if amount_from_operand is not None:
            amount = get_amount(amount_from_operand)
        else:
            amount = MAIN_VAR
    else:
        amount = amount

    if scaling_type == "expand":
        amount = amount
    elif scaling_type == "contract":
        amount = -amount
    else:
        raise ValueError(f"Scale scaling_type not properly defined: {scaling_type}")
    
    if isinstance(operand, Part):
        edit = PartScale1D(operand,
                           origin=axis_origin,
                           direction=axis_direction,
                           amount=amount,
                           identifier=identifier)
    elif isinstance(operand, FaceFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = FaceScale1D(part,
                           origin=axis_origin,
                           direction=axis_direction,
                           amount=amount,
                           identifier=identifier,
                           restrictor_name=restrictor_name)
    elif isinstance(operand, EdgeFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = EdgeScale1D(part,
                           origin=axis_origin,
                           direction=axis_direction,
                           amount=amount,
                           identifier=identifier,
                           restrictor_name=restrictor_name)
    else:
        raise ValueError(f"Unsupported operand type: {type(operand)}")
    return edit


def PlaneScale(operand, scaling_type, scale_origin, plane_normal, amount=None, amount_from_operand=None):
    identifier = None 
    if amount is None:
        if amount_from_operand is not None:
            amount = get_amount(amount_from_operand)
        else:
            amount = MAIN_VAR
    else:
        amount = amount
        
    if scaling_type == "expand":
        amount = amount
    elif scaling_type == "contract":
        amount = -amount
    else:
        raise ValueError(f"Scale scaling_type not properly defined: {scaling_type}")
    
    if isinstance(operand, Part):
        edit = PartScale2D(operand,
                           origin=scale_origin,
                           plane_normal=plane_normal,
                            amount=amount,
                            identifier=identifier)
    elif isinstance(operand, FaceFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = FaceScale2D(part,
                           origin=scale_origin,
                           plane_normal=plane_normal,
                           amount=amount,
                           identifier=identifier,
                           restrictor_name=restrictor_name)
    elif isinstance(operand, EdgeFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = EdgeScale2D(part,
                           origin=scale_origin,
                           plane_normal=plane_normal,
                           amount=amount,
                           identifier=identifier,
                           restrictor_name=restrictor_name)
    else:
        raise ValueError(f"Unsupported operand type: {type(operand)}")
    return edit


def VolumeScale(operand, scaling_type, scale_origin, amount=None, amount_from_operand=None):
    identifier = None 
    if amount is None:
        if amount_from_operand is not None:
            amount = get_amount(amount_from_operand)
        else:
            amount = MAIN_VAR
    else:
        amount = amount
        
    if scaling_type == "expand":
        amount = amount
    elif scaling_type == "contract":
        amount = -amount
    else:
        raise ValueError(f"Scale scaling_type not properly defined: {scaling_type}")
    
    if isinstance(operand, Part):
        edit = PartScale3D(operand,
                           origin=scale_origin, 
                           amount=amount,
                           identifier=identifier)
    elif isinstance(operand, FaceFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = FaceScale3D(part,
                           origin=scale_origin,
                           amount=amount,
                           identifier=identifier,
                           restrictor_name=restrictor_name)
    elif isinstance(operand, EdgeFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = EdgeScale3D(part,
                           origin=scale_origin,
                           amount=amount,
                           identifier=identifier,
                           restrictor_name=restrictor_name)
    else:
        raise ValueError(f"Unsupported operand type: {type(operand)}")
    return edit

def Shear(operand, shear_direction, shear_plane_origin, shear_plane_normal, amount=None, amount_from_operand=None):
    if amount is None:
        if amount_from_operand is not None:
            amount = get_amount(amount_from_operand)
        else:
            amount=MAIN_VAR
    else:
        amount = amount
    identifier = None
    if isinstance(operand, Part):
        edit = PartShear(operand,
                            shear_plane_origin=shear_plane_origin,
                            shear_plane_normal=shear_plane_normal,
                            shear_direction=shear_direction,
                            amount=amount,
                            identifier=identifier)
    elif isinstance(operand, FaceFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = FaceShear(part,
                            shear_plane_origin=shear_plane_origin,
                            shear_plane_normal=shear_plane_normal,
                            shear_direction=shear_direction,
                            amount=amount,
                            identifier=identifier,
                            restrictor_name=restrictor_name)
    elif isinstance(operand, EdgeFeature):
        part = operand.primitive.part
        restrictor_name = operand.name
        edit = EdgeShear(part,
                            shear_plane_origin=shear_plane_origin,
                            shear_plane_normal=shear_plane_normal,
                            shear_direction=shear_direction,
                            amount=amount,
                            identifier=identifier,
                            restrictor_name=restrictor_name)
    else:
        raise ValueError(f"Unsupported operand type: {type(operand)}")
    return edit

def SymGroupEdit(operand, change_type="count", extend_from="center", amount=None, amount_from_operand=None):
    if amount is None:
        if amount_from_operand is not None:
            amount = get_amount(amount_from_operand)
        else:
            amount=None
    else:
        amount=amount
        
    identifier=None
    # get relation from the operand
    relations = operand.all_relations(only_active=True)
    relation = [x for x in relations if isinstance(x, (TranslationSymmetry, RotationSymmetry))][0]
    if extend_from == "start":
        stretch_annotation = "end"
    elif extend_from == "end":
        stretch_annotation = "start"
    else:
        stretch_annotation = "center"
    edits = sym_group_wrapper(relation, stretch_annotation=stretch_annotation, scaling_type=change_type,
            amount=amount, identifier=identifier)
    return edits


def get_amount(amount_from_operand):
    if isinstance(amount_from_operand, (Part, FaceFeature, EdgeFeature)):
        amount = amount_from_operand.primitive.edit_sequence[0].amount
    elif isinstance(amount_from_operand, Relation):
        amount = amount_from_operand.edit_sequence[0].amount
    else:
        amount = None
    return amount


def ChangeAngleBetween(operand, other_part, amount=None, amount_from_operand=None):
    identifier = None 
    if amount is None:
        if amount_from_operand is not None:
            amount = get_amount(amount_from_operand)
        else:
            amount = None
    else:
        amount = amount
    # Get rotation axis and origin from the relation
    # Gather the contact point. 
    one_relations = [x for x in operand.all_relations() if isinstance(x, FeatureRelation)]
    two_relations = [x for x in other_part.all_relations() if isinstance(x, FeatureRelation)]
    matching_relation = [x for x in one_relations if x in two_relations][0]
    # This should be just one relation
    global_points = [x.static_expression() for x in matching_relation.features]
    # stack them sp
    global_points = sp.Matrix.vstack(*global_points)
    mean_axis_0 = sp.Matrix([
        sum(global_points.col(j)) / global_points.rows for j in range(global_points.cols)
    ])

    # mean_axis_0 is a column vector; if you want a row vector, transpose it
    origin = mean_axis_0.T

    # Now we need principal axis.
    pca_operand = operand.primitive.sorted_principal_axis[0]
    pca_other_part = other_part.primitive.sorted_principal_axis[0]

    # get cross vector
    cross_vector = pca_operand.cross(pca_other_part)
    if isinstance(operand, Part):
        edit = PartRotate(operand,
                          rotation_origin=origin,
                          rotation_axis=cross_vector,
                          amount=amount,
                          identifier=identifier)
    else:
        raise ValueError(f"Unsupported operand type: {type(operand)}")
    return edit


def CurveInCenter(operand, direction, amount=None, amount_from_operand=None):
    identifier = None 
    if amount is None:
        if amount_from_operand is not None:
            amount = get_amount(amount_from_operand)
        else:
            amount = None
    else:
        amount = amount
    if isinstance(operand, Part):
        edit = PartCenterBendBeta(operand,
                             direction=direction,
                             amount=amount,
                             identifier=identifier)
    else:
        raise ValueError(f"Unsupported operand type: {type(operand)}")
    return edit