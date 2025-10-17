from collections import defaultdict
import os
import sqlite3
import time
import traceback
import _pickle as cPickle
import numpy as np
from .base_prompter import LLMPrompterV1
from .human_prompter import HumanPrompter
from edit_sys.shape_system.edits import PartEdit, EditGen, KeepFixed
from edit_sys.shape_system.shape_atoms import Part
from edit_sys.shape_system.relations import PrimitiveRelation, FeatureRelation
import edit_sys.shape_system.edit_wrapper as ew
import edit_sys.shape_system.edit_gen_wrapper as egw
import edit_sys.shape_system.shape_atoms as sa
from edit_sys.shape_system.final_annotation import (shape_to_file_hier,
                                                    get_relation_in_detail)
from edit_sys.shape_system.final_annotation import (shape_to_file_hier,
                                                    get_relation_in_detail,
                                                    get_part_in_details,
                                                    get_unedited_parts,
                                                    generate_least_breaking_str,
                                                    get_all_edits_in_detail,
                                                    get_options_in_detail)  
from edit_sys.shape_system import INIT_EDIT_API, SECONDARY_EDIT_API, GEOMETRIC_ATOMS, EDIT_ATOMS_WRAPPER
from edit_sys.shape_system.edits import HIGHER_TYPE_HINTS
from .utils import parallel_make_api_call, log_all, DB_FILE
from ..state import State
import multiprocessing as mp
from .base_prompter import APIFailure, response_to_snippet, snippet_to_variable, update_amount

from string import Template
from .prompts.v7 import initialize
from .prompts.v7 import check_relation
from .prompts.v7 import finish_algo
from .prompts.v7 import edit_type
from .prompts.v7 import edit_spec
from .prompts.v7 import edit_action
from .prompts.v7 import init_type_hinting

class LLMPrompterV2Base(LLMPrompterV1):

    def create_keep_fixed_prompt(self, shape, edit_request,):
        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                            "shape_specification": shape_specification,
                            "edit_request": edit_request}
        template = initialize.instructions_keep_fixed()
            
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": instructions_filled}
            ]
        
        return messages

    def create_initialize_prompt(self, shape, edit_request):
        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "API": INIT_EDIT_API}
        
        template = initialize.instructions_primary_edit()
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages
    
    def create_check_relation_prompt(self, shape, edit_request, relation, valid_states, resolved_new_edits=None):
        # check for fixed
        if isinstance(relation, FeatureRelation):
            all_relation_edits = []
            part = relation.get_unedited_parts()
            edited_part = [x for x in relation.parts if x not in part]
            edited_part = edited_part[0]
            edit = edited_part.primitive.edit_sequence[0]
            if isinstance(edit, KeepFixed):
                has_fixed = True
            else:
                has_fixed = False
        else:
            has_fixed = False

        shape_specification = shape_to_file_hier(shape)
        relation_in_detail = get_relation_in_detail(relation, shape)
        if isinstance(relation, (PrimitiveRelation)):
            sym_group = True
        else:
            sym_group = False
        option_tips = check_relation.get_option_tips(relation, sym_group, valid_states, has_fixed=has_fixed)
        if not resolved_new_edits is None:
            potential_new_edits = [x.signature(shape) for x in resolved_new_edits]
            potential_new_edits = "(" + ",".join(potential_new_edits) + ")"
        else:
            potential_new_edits = ""
        options_string = check_relation.relation_option_set_to_string(relation, sym_group, valid_states, potential_new_edits, has_fixed=has_fixed)
        
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "relation_in_detail": relation_in_detail,
                           "options_string": options_string,
                           "option_tips": option_tips,
                           }
        template = check_relation.instructions()
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages

    def api_call_initial_update(self, shape, edit_request):
        return super().api_call_initial_update(shape, edit_request)
    
    def api_call_initialize(self, shape, edit_request):
        return super().api_call_initialize(shape, edit_request)

    def _get_end_algo(self, shape, messages, all_edits, edit_request, n_tries=0):
        algorithm_end = True
        response = ''
        return algorithm_end, response
    
    def api_call_finish_algo(self, shape, edit_request, all_edits):
        cur_info_list = []
        messages = self.create_finish_algo_prompt(all_edits, shape, edit_request)
        algorithm_end, response = self._get_end_algo(shape, messages, all_edits, edit_request)
        cur_info = {
            'step': 'finish_algo',
            'algorithm_end': algorithm_end,
            'prompt': messages[1]['content'],
            'response': response
        }
        cur_info_list.append(cur_info)

        if algorithm_end:
            new_edit = None
        else:
            # Need to create new edits.
            messages = self.create_new_edit_prompt(all_edits, shape, edit_request)
            new_edit, response = self._get_new_edit(shape, messages, all_edits, edit_request)
            cur_info = self.get_new_edit_info(new_edit, messages, response)
            cur_info_list.append(cur_info)
        return algorithm_end, new_edit, cur_info_list
    
    def api_call_check_relation(self, shape, edit_request, all_edits, relation, valid_states, resolved_new_edit=None, type_hints=None):

        if isinstance(relation, PrimitiveRelation):
            messages = self.create_check_relation_prompt(shape, edit_request, relation, valid_states, resolved_new_edit)

            selected_option, response = self._get_check_relation_option(shape, messages, relation, valid_states)
            assert selected_option in valid_states, f"Selected option {selected_option} not in valid states {valid_states}"
            cur_info = self.get_check_relation_info(relation, valid_states, selected_option, messages, response)

            cur_info_list = [cur_info]
        else:
            selected_option = min(valid_states)
            cur_info_list = [{"step": "check_relation", "response": ""}]
        return selected_option, cur_info_list

class LLMPrompterV2(LLMPrompterV2Base):
        
        
        
    def api_call_generate_edit_spec(self, part_to_edit, shape, edit_request, all_edits, last_edit=None):
    
        print(f"Generating Edit Spec for {part_to_edit}")

        messages = self.create_edit_spec_prompt(shape, edit_request, part_to_edit, all_edits, last_edit)
        edit_spec, response = self._get_edit_spec(shape, messages, part_to_edit)
        cur_info = self.get_edit_spec_info(part_to_edit, edit_spec, messages, response)
        cur_info_list = [cur_info]
        return edit_spec, cur_info_list
    
    def _get_edit_spec(self, shape, messages, n_tries=0):
        edit_spec, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_edit_spec, error_template_map=None)
        return edit_spec, response
    
    def parse_edit_spec(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        edit_spec = snippet_to_variable_wt_wrapper(snippet,shape, "edit")

        return edit_spec
    
    
    def get_edit_spec_info(self, part_to_edit, edit_spec, messages, response):
        prompt = messages[1]['content']
        if isinstance(part_to_edit, Part):
            edit_item =  (edit_spec, part_to_edit.part_index)
        else:
            edit_item = (edit_spec, part_to_edit.relation_index)

        cur_info = {
            'step': 'edit_spec',
            'prompt': prompt,
            'edit_output': edit_item,
            'response': response
        }
        return cur_info

    
    def api_call_generate_edit_type(self, part_to_edit, shape, edit_request, all_edits, last_edit=None):
        print(f"Generating Edit Type for {part_to_edit}")

        messages = self.create_edit_type_prompt(shape, edit_request, part_to_edit, all_edits, last_edit)
        edit_type, response = self._get_edit_type(shape, messages, part_to_edit)
        cur_info = self.get_edit_type_info(part_to_edit, edit_type, messages, response)
        cur_info_list = [cur_info]

        return edit_type, cur_info_list

    def _get_edit_type(self, shape, messages, part_to_edit, n_tries=0):
        edit_type, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_edit_type, error_template_map=None)
        return edit_type, response
    
    def parse_edit_type(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        edit_type = snippet_to_variable(snippet,shape, "edit_type")
        return edit_type
    
    def get_edit_type_info(self, part_to_edit, edit_type, messages, response):
        prompt = messages[1]['content']
        if isinstance(part_to_edit, Part):
            edit_item =  (edit_type, part_to_edit.part_index)
        else:
            edit_item = (edit_type, part_to_edit.relation_index)
        cur_info = {
            'step': 'edit_type',
            'prompt': prompt,
            'edit_output': edit_item,
            'response': response
        }
        return cur_info


    def api_call_get_edit_action(self, part_to_edit, shape, edit_request, all_edits, least_breaking):
        print(f"Deciding to accept least breaking or not for {part_to_edit}")
        messages = self.create_edit_action_prompt(shape, edit_request, part_to_edit, all_edits, least_breaking)
        action, response = self._get_edit_action(shape, messages)
        cur_info = self.get_edit_action_info(part_to_edit, action, messages, response, least_breaking)
        cur_info_list = [cur_info]

        return action, cur_info_list

    def _get_edit_action(self, shape, messages, n_tries=0):
        edit_action, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_edit_action, error_template_map=None)
        return edit_action, response
    
    def parse_edit_action(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        edit_action = snippet_to_variable(snippet,shape, "action")

        return edit_action
    
    def get_edit_action_info(self, part_to_edit, action, messages, response, least_breaking):
        prompt = messages[1]['content']

        cur_info = {
            'step': 'edit_action',
            'prompt': prompt,
            'edit_output': action,
            'response': response
        }
        return cur_info

    def create_edit_spec_prompt(self, shape, edit_request, part_to_edit, all_edits, last_edits=None):
        # code for making the edit spec
        shape_specification = shape_to_file_hier(shape)
        unedited_parts = get_unedited_parts(shape)
        if isinstance(part_to_edit, Part):
            part_in_detail = get_part_in_details(part_to_edit, shape, edit_code=True)
        else:
            part_in_detail = get_relation_in_detail(part_to_edit, shape, edit_code=True)
        all_types = ['translate', 'rotate', 'scale', 'shear', 'change_count', 'change_delta']
        substitute_dict = {
            "shape_class": shape.label,
            "shape_specification": shape_specification,
            "edit_request": edit_request,
            "part_in_detail": part_in_detail,
            "unedited_parts": unedited_parts,
            "remaining_type_hints": all_types,
            "part": part_to_edit.label,
            "API": SECONDARY_EDIT_API
        }
        if last_edits is None:
            template = edit_spec.base_instructions()
            instructions_filled = Template(template).substitute(substitute_dict)
        else:
            # update subtitute dict
            failed_edit_str = get_failed_edits_str(last_edits, shape)
            substitute_dict['failed_edit'] = failed_edit_str
            template = edit_spec.with_failure_instructions()
            instructions_filled = Template(template).substitute(substitute_dict)

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages

    def create_edit_type_prompt(self, shape, edit_request, part_to_edit, all_edits, last_edit=None):
        # code for making the edit type
        shape_specification = shape_to_file_hier(shape)
        unedited_parts = get_unedited_parts(shape)
        if isinstance(part_to_edit, Part):
            part_in_detail = get_part_in_details(part_to_edit, shape)
            label = part_to_edit.label
            all_types = ['translate', 'rotate', 'scale', 'shear', 'change_count', 'change_delta']
        else:
            part_in_detail = get_relation_in_detail(part_to_edit, shape)
            label = f"{part_to_edit.__class__.__name__}({part_to_edit.parent_part.label})"
            all_types = ['change_count', 'change_delta']
        substitute_dict = {
            "shape_class": shape.label,
            "shape_specification": shape_specification,
            "edit_request": edit_request,
            "part_in_detail": part_in_detail,
            "unedited_parts": unedited_parts,
            "remaining_type_hints": all_types,
            "part": label,
        }
        if last_edit is None:
            template = edit_type.base_instructions()
            instructions_filled = Template(template).substitute(substitute_dict)
        else:
            # update subtitute dict
            prev_index = all_types.index(last_edit[-1])
            remaining_types = all_types[prev_index + 1: -2]
            substitute_dict["remaining_type_hints"] = remaining_types
            template = edit_type.with_failure_instructions()
            instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages

    def create_edit_action_prompt(self, shape, edit_request, part_to_edit, all_edits, least_breaking):
        
        # code for making the action_prompt
        shape_specification = shape_to_file_hier(shape)
        unedited_parts = get_unedited_parts(shape)
        if isinstance(part_to_edit, Part):
            part_in_detail = get_part_in_details(part_to_edit, shape)
        else:
            part_in_detail = get_relation_in_detail(part_to_edit, shape)
        least_breaking_str = generate_least_breaking_str(least_breaking, shape)
        substitute_dict = {
            "shape_class": shape.label,
            "shape_specification": shape_specification,
            "edit_request": edit_request,
            "part_in_detail": part_in_detail,
            "unedited_parts": unedited_parts,
            "part": part_to_edit.label,
            "minimally_relation_breaking_edit": least_breaking_str,
        }
        template = edit_action.base_instructions()
        instructions_filled = Template(template).substitute(substitute_dict)

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages


    def api_call_get_init_edit_hints(self, shape, edit_request, new_edits):
        
        messages = self.create_init_edit_hint_prompt(shape, edit_request)
        n_tries = 0

        edit_type_hints, response = self._get_init_edit_hints(shape, messages)
        cur_info = {
            'step': 'get_edit_hints',
            'edit_type_hints': edit_type_hints,
            'prompt': messages[1]['content'],
            'response': response,
            'type_hints': edit_type_hints
        }
        full_types = {}
        for name, edit_type in edit_type_hints.items():
            if edit_type in ['change_count', "change_delta"]:
                m = shape.get(name)
                if hasattr(m, 'core_relation'):
                    relation = m.core_relation
                else:
                    relation = m.parent.core_relation
                full_types[relation] = HIGHER_TYPE_HINTS[edit_type]
                edit_type = "scale"
            full_types[shape.get(name).full_label] = HIGHER_TYPE_HINTS[edit_type]
        cur_info_list = [cur_info]

        return full_types, cur_info_list
    
    def _get_init_edit_hints(self, shape, messages, n_tries=0):
        edit_type_hints, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_init_edit_hints, error_template_map=None)
        return edit_type_hints, response

    def parse_init_edit_hints(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        edit_hints = snippet_to_variable(snippet,shape, "edit_type_hints")
        # summary = snippet_to_variable(snippet,shape, "summary")
        return edit_hints
    

    def create_init_edit_hint_prompt(self, shape, edit_request):
        
        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request}
        template = init_type_hinting.instructions
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages

def snippet_to_variable_wt_wrapper(snippet, shape, variable_name):
    ldict = {}
    global_dict = {}
    global_dict['shape'] = shape
    global_dict.update(GEOMETRIC_ATOMS)
    global_dict.update(EDIT_ATOMS_WRAPPER)
    try: 
        exec(snippet, global_dict, ldict)
    except Exception as ex: 
        print("Failed to parse LLM Code.")
        print(ex)
        raise ex
    if not variable_name in ldict.keys():
        error_message = f"spec snippet did not specify the variable `{variable_name}`"
        print(f"Wrong snippet parsing.\n =========\n {error_message}")
        raise SyntaxError(error_message)
    desired_variable = ldict[variable_name]
    return desired_variable
