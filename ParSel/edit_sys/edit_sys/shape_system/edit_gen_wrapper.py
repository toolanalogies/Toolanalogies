
from .edits import *
from .shape_atoms import *
from .relations import *
from .constants import DOT_THRESHOLD
from .edit_wrapper import (Translate as EditTranslate, Rotate as EditRotate, 
                           Scale as EditScale, Shear as EditShear, 
                           SymGroupEdit as EditSymGroupEdit, 
                           ChangeAngleBetween as EditChangeAngleBetween,
                           CurveInCenter as EditCurveInCenter,)

def Translate(*args, **kwargs):
    edit = EditTranslate(*args, **kwargs)
    edit_gen = EditGen(
        edit_class=edit.__class__,
        param_dict=edit.params,
    )
    return edit_gen

def Rotate(*args, **kwargs):
    edit = EditRotate(*args, **kwargs)
    edit_gen = EditGen(
        edit_class=edit.__class__,
        param_dict=edit.params,
    )
    return edit_gen

def Scale(*args, **kwargs):
    edit = EditScale(*args, **kwargs)
    edit_gen = EditGen(
        edit_class=edit.__class__,
        param_dict=edit.params,
    )
    return edit_gen

def Shear(*args, **kwargs):
    edit = EditShear(*args, **kwargs)
    edit_gen = EditGen(
        edit_class=edit.__class__,
        param_dict=edit.params,
    )
    return edit_gen

def SymGroupEdit(*args, **kwargs):
    edit = EditSymGroupEdit(*args, **kwargs)
    edit_gen = EditGen(
        edit_class=edit.__class__,
        param_dict=edit.params,
    )
    return edit_gen

def ChangeAngleBetween(*args, **kwargs):
    edit = ChangeAngleBetween(*args, **kwargs)
    edit_gen = EditGen(
        edit_class=edit.__class__,
        param_dict=edit.params,
    )
    return edit_gen

def CurveInCenter(*args, **kwargs):
    edit = CurveInCenter(*args, **kwargs)
    edit_gen = EditGen(
        edit_class=edit.__class__,
        param_dict=edit.params,
    )
    return edit_gen
