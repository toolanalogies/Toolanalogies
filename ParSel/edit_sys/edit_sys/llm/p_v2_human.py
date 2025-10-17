"""
This system is / was used for the manual annotation of edit programs for each (shape, edit request) pair.
The user plays the role of the LLM and provides answers for various prompts. 
This is the updated workflow where the details to be inferred are different from the version in human_prompter.py.
"""
import _pickle as cPickle

import edit_sys.shape_system.shape_atoms as sa
import edit_sys.shape_system.edit_wrapper as ew
from edit_sys.shape_system.edits import *

from .prompter_v2 import LLMPrompterV2

class LLMPrompterHuman(LLMPrompterV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relation_count = 0
        self.edit_type_count = 0
        self.end_counter = 1
        self.internal_save_path = kwargs.get('internal_save_path', None)
        if self.internal_save_path is None:
            self.internal_save_path = ""
        self.load_internal_state()

    def _get_fixed_parts(self, shape, messages, n_tries=0):
        if self.fixed_parts is None:
            
            fixed_parts = [  
                #"lamp_base"
                # "backrest",
                # "backrest_connector",
                # "vertical_back_panel",
                # "backrest_connector_left_down",
                # "backrest_connector_right_up",
                # "backrest_connector_left_up",
                # "backrest_frame"
                
                # "leg_horizontal_back",
                # "leg_horizontal_front",
                # "shelf_up",
                # "shelf_down",
                # "stretcher_back",
                # "leg_back_left",
                # "leg_back_right",
                # "backrest_frame_left",
                # "backrest_frame_right",
                # 'horizontal_top_panel',
                # 'horizontal_bottom_panel',
                # 'vertical_side_panel_left',
                # 'vertical_side_panel_right',
                # 'leg_horizontal_front',
                # 'leg_horizontal_back',
                # 'leg_vertical_right',
                # 'leg_vertical_left',
                # "stretcher_center"
                # "shelf_up",
                # "shelf_down",
                # "vertical_back_panel"
                # 'drawer_front',
                # 'drawer_side_right',
                # 'drawer_back',
                # 'drawer_bottom',
                # 'drawer_side_left',
                # 'cabinet_door_surface',
                # 'frame_vertical_bar_left_down',
                # 'base_side_panel_left',
                # 'frame_horizontal_bar_front_left',
                # 'vertical_side_panel_left',
                # 'frame_vertical_bar_left_up',
            ]
            self.fixed_parts = fixed_parts
        else:
            fixed_parts = self.fixed_parts
        response = ''

        # This should be done later.
        return fixed_parts, response
    

    def _get_primary_edits(self, shape, messages, n_tries=0):
        # Define the new edits here
        if self.primary_edits is None:

            part = shape.get("stick_head")
            # new_edit = ew.Translate(part.face("up"), sa.DownUnitVector())

            # new_edit = ew.Scale(part, part.center(), 
            #                     "expand", sa.UpUnitVector())
            # part = shape.get("pedestal")

            #part = shape.get("leg_front_right")
            # item=input("What part to change?")
    

            # part = shape.get(item)

            # change=input("Input Scale, Translate, Shear, Rotate")
            # while change not in ["Scale","Translate","Shear","Rotate"]:
            #         change=input("Input Scale, Translate, Shear, Rotate")



            # if change=="Scale":
            #     axes=input("Number of axes? (1, 2 or 3)")
            #     while axes not in ["1","2","3"]:
            #         axes=input("Number of axes? (1, 2 or 3)")
            #     if axes=="1":
            #         # rl=input("Right or left (r,l)")
            #         new_edit = ew.Scale(part, "expand", part.center(), sa.RightUnitVector())
            #         # elif rl=="l":
            #         # if rl=="r":
            #         #     new_edit = ew.Scale(part, "expand", part.center(), sa.RightUnitVector())
            #         # elif rl=="l":
            #         #     new_edit = ew.Scale(part, "expand", part.center(), sa.UpUnitVector())
            #     elif axes=="2":
            #         new_edit = ew.Scale(part, "expand", part.center(), sa.RightUnitVector(), sa.LeftUnitVector())
            #     elif axes=="3":
            #         new_edit = ew.Scale(part, "expand", part.center(), sa.RightUnitVector(), sa.UpUnitVector(), sa.FrontUnitVector())



            # elif change=="Translate":
            #     new_edit = ew.Translate(part, sa.UpUnitVector())
            # elif change=="Shear":
            #     new_edit = ew.Shear(part, "expand", part.center(), sa.RightUnitVector())
            # elif change=="Rotate":
            #     new_edit = ew.Rotate(part, rotation_axis_origin=part.face_center("left"), 
            #                       rotation_axis_direction=sa.DownUnitVector()) 

            new_edit = ew.Scale(part, "expand", part.center(),part.face_center("front")-part.center())

            # new_edit = ew.Rotate(part, rotation_axis_origin=part.face_center("left"), 
            #                       rotation_axis_direction=sa.DownUnitVector()) 
            #print("get the human primary edits")
            #new_edit = ew.Rotate(part, rotation_axis_origin=part.face_center("left"), 
            #                      rotation_axis_direction=sa.DownUnitVector()) 

            #new_edit = ew.Translate(part, sa.UpUnitVector())
            #new_edit = ew.Scale(part, "expand", part.center(), sa.RightUnitVector())
            #new_edit = ew.Scale(part, "expand", part.center(), sa.RightUnitVector(), sa.FrontUnitVector())
            # new_edit = ew.SymGroupEdit(part, "count", extend_from="center")
            # new_edit = ew.SymGroupEdit(coat_rack_hangers, change_type="delta", extend_from="end")

            self._end_edits = [
                # (EditGen(), "part", part.part_index)
            ]
            primary_edits = new_edit
            if isinstance(new_edit, list):
                self.primary_edits = []
                for edit in new_edit:
                    if isinstance(edit.operand, Part):
                        self.primary_edits.append((EditGen(edit.__class__, edit.params, edit.amount), "part", edit.operand.part_index))
                    else:
                        self.primary_edits.append((EditGen(edit.__class__, edit.params, edit.amount), "relation", edit.operand.relation_index))
            else:
                if isinstance(new_edit.operand, Part):
                    self.primary_edits = [(EditGen(new_edit.__class__, new_edit.params, new_edit.amount), "part", part.part_index)]
                else:
                    self.primary_edits = [(EditGen(new_edit.__class__, new_edit.params, new_edit.amount), "relation", new_edit.operand.relation_index)]
        else:
            editgen_obj = self.primary_edits[0]
            primary_edits = self.generate_edit(shape, editgen_obj)
        response = ''
        return primary_edits, response
    
    def _get_init_edit_hints(self, shape, messages, n_tries=0):
        if self.init_edit_type_hints is None:
            init_edit_type_hints = {
                # "leg_right": "translate",
                # "leg_left": "translate",
                # "leg_back_left": "tilt",
                # "leg_back_right": "tilt",
                # "vertical_divider_panel": "translate",
                # "backrest_frame_right": "scale",
                # "stretcher_back": "scale",
                # "stretcher_front": "scale",
                # "vertical_divider_panel": "scale",
                # "vertical_side_panel_left": "scale",
                # "vertical_side_panel_right": "scale",
                # "vertical_back_panel": "scale",
                # "horizontal_top_panel": 'translate',
                # "leg_vertical_left": "scale",
                # "leg_vertical_right": "scale",
                # "cabinet_door_surface_left": "tilt",
                # "cabinet_door_surface_right": "tilt",
                # 'handle_left': 'tilt',
                # 'handle_right': 'tilt',
                # "leg_back_right": "scale",
                # "leg_back_left": "scale",
                # 'vertical_front_panel': 'scale',
                # 'base_side_panel_left': 'scale',
                # 'vertical_side_panel_right': 'scale',
                # 'frame_vertical_bar_right_up': 'scale',
                # 'frame_vertical_bar_left_up': 'scale',
                # 'base_side_panel_right': 'scale',
                # 'base_side_panel_front': 'scale',
                # 'frame_horizontal_bar_down': 'scale',
                # 'frame_vertical_bar_right_down': 'scale',
                # 'shelf': 'scale',
                # 'bottom_panel': 'scale',
                # 'frame_horizontal_bar_back': 'scale',
                # 'back_panel': 'scale',
                # 'handle_up': 'scale',
                # 'base_side_panel_back': 'scale',
                # 'frame_horizontal_bar_front_left': 'scale',
                # 'frame_vertical_bar_left_down': 'scale',
                # 'frame_horizontal_bar_front_up': 'scale',
                # 'handle_down': 'scale'
            }
            init_edit_type_hints = {shape.get(x).full_label: y for x, y in init_edit_type_hints.items()}
            self.init_edit_type_hints = init_edit_type_hints
        else:
            init_edit_type_hints = self.init_edit_type_hints
        response = ""
        return init_edit_type_hints, response
    
    
    def _get_check_relation_option(self, shape, messages, relation=None, valid_states=None, n_tries=0):

        if self.relation_map is None:
            print(messages[1]['content'])
            print("Checking relation:", relation)
            print("Options:", valid_states)
            print("state anno: 0: NOTHING_TO_DO (break), 1: Resolve, 2: update")
            options = []
            variable = None
            while variable is None:
                try:
                    user_inp = input(f"{options}?")
                    variable = int(user_inp)
                    assert variable in valid_states, f"Selected option {variable} not in valid states {valid_states}"
                except:
                    print("Invalid input. Try again.")
                    variable = None
            self._rel_map[relation.relation_index] = variable
        else:
            try:
                variable = self.relation_map[relation.relation_index]
            except:
                variable = None
                print("Error in getting relation option.")
                import pdb
                pdb.set_trace()
        response = ''
        return variable, response
    

    def _get_edit_type(self, shape, messages, part_to_edit, n_tries=0):
        if self.edit_type_seq_map is None:
            print("Checking type for:", part_to_edit)
            options = "return edit type"
            variable = None
            valid_types = ['translate', 'rotate', 'scale', 'shear', 'change_count', 'change_delta']
            while variable is None:
                try:
                    user_inp = input(f"{options}?")
                    variable = str(user_inp)
                    assert variable in valid_types
                except:
                    print("Invalid input. Try again.")
                    variable = None
            if isinstance(part_to_edit, Part):
                self._edit_type_seq_map[part_to_edit.part_index] = variable
            else:
                self._edit_type_seq_map[part_to_edit.relation_index] = variable
        else:
            try:
                if isinstance(part_to_edit, Part):
                    variable = self.edit_type_seq_map[part_to_edit.part_index]
                else:
                    variable = self.edit_type_seq_map[part_to_edit.relation_index]
            except:
                variable = None
                print("Error in getting edit type.")
        response = ''
        return variable, response
    
    
    def api_call_finish_algo(self, shape, edit_request, all_edits):

        if self.end_edits is None:
            if len(self._end_edits) == 0:
                algorithm_end = True
                response = ''
                new_edit = None
            else:
                algorithm_end = False
                response = ''
                edit_gen_obj = self._end_edits.pop(0)
                new_edit = self.generate_edit(shape, edit_gen_obj)
                self.primary_edits.append(edit_gen_obj)
        else:
            if len(self.end_edits) == 0:
                algorithm_end = True
                response = ''
                new_edit = None
            else:
                algorithm_end = False
                response = ''
                edit_gen_obj = self.end_edits.pop(0)
                new_edit = self.generate_edit(shape, edit_gen_obj)
                self.primary_edits.append(edit_gen_obj)

        if algorithm_end:
            self.save_internal_state()
        return algorithm_end, new_edit, response

    def generate_edit(self, shape, edit_gen_obj):
        edit_gen, operand_type, index = edit_gen_obj
        if operand_type == "part":
            operand = [x for x in shape.partset if x.part_index == index]
            if len(operand) > 0:
                operand = operand[0]
        else:
            operand = [x for x in shape.all_relations() if x.relation_index == index][0]
        new_edit = edit_gen.employ(operand)
        return new_edit
    

    def save_internal_state(self):
        if self.relation_map is None:
            self.relation_map = self._rel_map
        if self.edit_type_seq_map is None:
            self.edit_type_seq_map = self._edit_type_seq_map
        if self.end_edits is None:
            self.end_edits = self._end_edits

        data = {
            'fixed_parts': self.fixed_parts,
            'primary_edits': self.primary_edits,
            'relation_map': self.relation_map,
            'init_edit_type_hints': self.init_edit_type_hints,
            'edit_type_seq_map': self.edit_type_seq_map,
            'end_edits': self.end_edits
        }
        cPickle.dump(data, open(self.internal_save_path, "wb"))
    
    def load_internal_state(self):
        #return None
        # if os.path.exists(self.internal_save_path):
        if False:
            data = cPickle.load(open(self.internal_save_path, "rb"))
            self.fixed_parts = data.get('fixed_parts', None)
            self.primary_edits = data.get('primary_edits', None)
            self.relation_map = data.get('relation_map', None)
            self.init_edit_type_hints = data.get('init_edit_type_hints', None)
            self.edit_type_seq_map = data.get('edit_type_seq_map', None)
            self.end_edits = data.get('end_edits', None)
        else:
            self.fixed_parts = None
            self.primary_edits = None
            self.relation_map = None
            self.init_edit_type_hints = None
            self.edit_type_seq_map = None
            self.end_edits = None
        self._rel_map = {}
        self._edit_type_seq_map = {}
        self._end_edits = []


class GeometricBase(LLMPrompterHuman):

    def _get_init_edit_hints(self, shape, messages, n_tries=0):
        edit_type_hints, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_init_edit_hints, error_template_map=None)
        return edit_type_hints, response
    