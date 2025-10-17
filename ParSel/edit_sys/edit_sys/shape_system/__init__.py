
import os
from .geometric_atoms import *
from .shape_atoms import *
from .relations import *
from .edits import KeepFixed
from .edit_wrapper import Translate, Rotate, AxisScale, PlaneScale, VolumeScale, Scale, Shear, SymGroupEdit, ChangeAngleBetween, CurveInCenter
from .edit_gen_wrapper import Translate as TranslateGen, Rotate as RotateGen, Scale as ScaleGen, Shear as ShearGen, SymGroupEdit as SymGroupEditGen, ChangeAngleBetween as ChangeAngleBetweenGen, CurveInCenter as CurveInCenterGen

all_items = [Plane, Line, Hexahedron, Part,
             PointFeature, FaceFeature, LineFeature,
             PartRelation, FeatureRelation, ReflectionSymmetry, RotationSymmetry,
             TranslationSymmetry, PointContact, FaceContact,
             FrontUnitVector, BackUnitVector, LeftUnitVector, RightUnitVector, UpUnitVector, DownUnitVector,
             LeftToRightLine, FrontToBackLine, DownToUpLine, RightToLeftLine, BackToFrontLine, UpToDownLine,
             ]

GEOMETRIC_ATOMS = {x.__name__: x for x in all_items}

all_edit_atoms = [Translate, Rotate, AxisScale, PlaneScale, VolumeScale, Scale, Shear, SymGroupEdit, KeepFixed, ChangeAngleBetween, CurveInCenter]

EDIT_ATOMS = {x.__name__: x for x in all_edit_atoms}

all_edit_atoms_wrapper = [TranslateGen, RotateGen, ScaleGen, ShearGen, SymGroupEditGen, ChangeAngleBetweenGen, CurveInCenterGen]

EDIT_ATOMS_WRAPPER = {x.__name__: x for x in all_edit_atoms_wrapper}

api_file = os.path.join(os.path.dirname(__file__), "mkds", "edits_api.md")
INIT_EDIT_API = open(api_file, "r").read()

api_file = os.path.join(os.path.dirname(__file__), "mkds", "secondary_edits_api.md")
SECONDARY_EDIT_API = open(api_file, "r").read() 

singlepass_api_file = os.path.join(os.path.dirname(__file__), "mkds", "single_pass_api.md")
SINGLE_PASS_API = open(singlepass_api_file, "r").read()


api_file = os.path.join(os.path.dirname(__file__), "mkds", "shape_only.md")
SHAPE_API = open(api_file, "r").read() 

api_file = os.path.join(os.path.dirname(__file__), "mkds", "edits_only.md")
EDIT_API = open(api_file, "r").read() 
# api_file = os.path.join(os.path.dirname(__file__), "mkds", "core_edit_api.md")
# EDIT_API = open(api_file, "r").read()
# api_file = os.path.join(os.path.dirname(__file__),
#                         "mkds", "edit_api_sans_symbol.md")
# EDIT_API_SANS_SYMBOL = open(api_file, "r").read()

# api_file = os.path.join(os.path.dirname(__file__),
#                         "mkds", "edit_description.md")
# EDIT_DESC = open(api_file, "r").read()
