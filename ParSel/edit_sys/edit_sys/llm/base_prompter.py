from string import Template
import openai
import re
import sympy as sp
import numpy as np
import traceback
import copy
import _pickle as cPickle
# from .prompts.v7 import initialize
# from .prompts.v7 import check_relation
# from .prompts.v7 import select_option
# from .prompts.v7 import finish_algo
from .prompts.v6 import initialize
from .prompts.v6 import check_relation
from .prompts.v6 import select_option
from .prompts.v6 import finish_algo
from .prompts.v6 import edit_type_hints as hinting_prompts
import open3d as o3d
from edit_sys.shape_system import INIT_EDIT_API, EDIT_ATOMS, GEOMETRIC_ATOMS
from edit_sys.shape_system.relations import RELATION_RETAINED, RELATION_BROKEN
from edit_sys.shape_system.relations import ReflectionSymmetry, RotationSymmetry, TranslationSymmetry, PrimitiveRelation
from edit_sys.shape_system import UpUnitVector, LeftUnitVector, BackUnitVector, RightUnitVector, FrontUnitVector, DownUnitVector
from edit_sys.shape_system.edits import *
from edit_sys.shape_system.edit_wrapper import *
from edit_sys.shape_system.constants import MOVE_LIMIT
from edit_sys.data_loader.constants import ICP_THRESHOLD, MAX_ICP_ITER

from edit_sys.shape_system.prompt_annotations import (shape_signature,  
                                                      initialize_text_description, 
                                                      keep_fixed_shape_description,
                                                      shape_unedited_parts_str,
                                                      shape_remaining_parts_str)
from edit_sys.shape_system.final_annotation import (shape_to_file_hier,
                                                    get_relation_in_detail,
                                                    get_part_in_details,
                                                    get_unedited_parts,
                                                    get_all_edits_in_detail,
                                                    get_options_in_detail)  
import os
from edit_sys.shape_system.edits import EditGen
from .utils import make_api_call
from .errors import SyntaxError, SpecError, APIFailure
from ..state import State
import anthropic

N_OPTIONS_TO_CONSIDER = 20

class BaseLLMPrompter:
    def __init__(self, mode, key, model, temperature=0.0, seed=42, max_tries=2, *args, **kwargs):
        self.key = key
        self.model = model
        self.temperature = temperature
        self.max_tries = max_tries
        self.seed = seed
        self.mode = mode
        if mode == 'openai':
            openai.api_key = key
            self.client = None
            os.environ["OPENAI_API_KEY"] = key
        else:
            self.client = anthropic.Client(api_key=key)
        

class LLMPrompterV1(BaseLLMPrompter):
    # Version 1:

    def _error_tolerant_call(self, shape, messages, max_tries, n_tries, parsing_func, error_template_map, substitute_dict=None):
        if n_tries >= max_tries:
            print("FAILURE!!")
            raise APIFailure(f"Failure despite {n_tries} tries.")
        content, role, response_valid_stop = make_api_call(model=self.model, messages=messages, temperature=self.temperature, seed=self.seed)
        if not response_valid_stop:
            print(response)
            raise Exception("Error in finishing GPT response.")
        try:
            edit_instruction_set = parsing_func(shape, content)
        except Exception as ex:
            print(ex)
            self.seed = np.random.randint(0, 100000)
            # Try again with error information
            if error_template_map is None:
                error_template_map = defaultdict(lambda: "There is a error in your code. Please try again.\nHere is the error message:\n $error_message")
            error_template = error_template_map[type(ex)]
            if substitute_dict is None:
                error_message = Template(error_template).substitute(error_message=str(traceback.print_stack()))
            else:
                substitute_dict = {}
                substitute_dict["error_message"] = str(ex)
                # substitute_dict["error_message"] = str(traceback.print_stack())
                error_message = Template(error_template).substitute(substitute_dict)

            messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": error_message})
            State.error_count += 1
            # messages = messages[:2]
            edit_instruction_set, response = self._error_tolerant_call(shape,
                messages, max_tries=max_tries, n_tries=n_tries + 1, 
                parsing_func=parsing_func, error_template_map=error_template_map)
            print("Tried with error.")
        return edit_instruction_set, content
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_keep_fixed = True
        self.n_options_to_consider = N_OPTIONS_TO_CONSIDER

    def api_call_initial_update(self, shape, edit_request, *args, **kwargs):
        # Ideally give options, and let LLM decide
        # Reflection -> Translation
        # Rotation -> Reflection tuple or translation tuple
        cur_info = {
            'step': 'initial_update',
        }
        cur_info_list = [cur_info]
        return shape, cur_info_list
    
    def api_call_initialize(self, shape, edit_request):
        # Create Initial Edit operations.
        keep_fixed_edits = []
        cur_info_list = []
        if self.eval_keep_fixed:
            messages = self.create_keep_fixed_prompt(shape, edit_request)
            fixed_parts, response = self._get_fixed_parts(shape, messages)
            print("edit_request:", edit_request)
            keep_fixed_edits = [KeepFixed(operand=shape.get(part)) for part in fixed_parts]
            keep_fixed_info = self.get_keep_fixed_info(shape, keep_fixed_edits, messages, response)
            cur_info_list.append(keep_fixed_info)
        
        messages = self.create_initialize_prompt(shape, edit_request)
        primary_edits, response = self._get_primary_edits(shape, messages)
        
        if not isinstance(primary_edits, list):
            if isinstance(primary_edits, PartEdit):
                update_amount(primary_edits)
            primary_edits = [primary_edits]
        primary_edit_info = self.get_primary_edit_info(primary_edits, messages, response)
        cur_info_list.append(primary_edit_info)
        edits = keep_fixed_edits + primary_edits

        return edits, cur_info_list

                
    def _get_fixed_parts(self, shape, messages, n_tries=0):
        fixed_parts, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                        parsing_func=self.parse_fixed_parts, error_template_map=None)
        return fixed_parts, response

    def _get_primary_edits(self, shape, messages, n_tries=0):
        primary_edits, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_initial_instructions, error_template_map=None)
        return primary_edits, response
    
    def get_keep_fixed_info(self, shape, fixed_part_edits, messages, response):
        prompt = messages[1]['content']
        edit_info = []
        for fixed_part in fixed_part_edits:
            edit_info.append(fixed_part.operand.part_index)
        cur_info = {
            'step': 'keep_fixed',
            'prompt': prompt,
            'edit_output': edit_info,
            'response': response
        }
        return cur_info

    def get_primary_edit_info(self, primary_edits, messages, response):
        prompt = messages[1]['content']
        edit_info = []
        for edit in primary_edits:
            if isinstance(edit, PartEdit):
                edit_item =  (EditGen(edit.__class__, edit.params), edit.operand.part_index)
            else:
                edit_item = (EditGen(edit.__class__, edit.params), edit.operand.relation_index)
            edit_info.append(edit_item)

        cur_info = {
            'step': 'initialize',
            'prompt': prompt,
            'edit_output': edit_info,
            'response': response
        }
        return cur_info


    def create_keep_fixed_prompt(self, shape, edit_request, nocot=False):
        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                            "shape_specification": shape_specification,
                            "edit_request": edit_request}
        if nocot:
            template = initialize.instructions_keep_fixed_nocot
        else:
            template = initialize.instructions_keep_fixed
            
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": instructions_filled}
            ]
        
        return messages

    def create_initialize_prompt(self, shape, edit_request, nocot=False):
        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "API": INIT_EDIT_API}
        if nocot:
            template = initialize.instructions_primary_edit_nocot
        else:
            template = initialize.instructions_primary_edit
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages
    
    def parse_fixed_parts(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        # fixed_parts = snippet_to_variable(snippet,shape, "fully_fixed_parts")
        fixed_parts = []
        location_fixed_parts = snippet_to_variable(snippet,shape, "location_fixed_parts")
        scale_fixed_parts = snippet_to_variable(snippet,shape, "scale_fixed_parts")
        rotation_fixed_parts = snippet_to_variable(snippet,shape, "rotation_fixed_parts")
        additional_parts = set(location_fixed_parts).intersection(set(scale_fixed_parts)).intersection(set(rotation_fixed_parts))
        fixed_parts = list(set(fixed_parts).union(additional_parts))
        # Is the spec valid?
        # for edit in primary_edits:
        return fixed_parts

    def parse_initial_instructions(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        primary_edit = snippet_to_variable(snippet,shape, "primary_edit")
        return primary_edit
    
    def api_call_check_relation(self, shape, edit_request, all_edits, relation, valid_states, resolved_new_edit=None, type_hints=None):
        # Check if a relationship should be maintained or not.

        print(f"validating relation {relation.signature()}")

        messages = self.create_check_relation_prompt(shape, edit_request, relation, valid_states, resolved_new_edit)
        selected_option, response = self._get_check_relation_option(shape, messages, relation, valid_states)
        assert selected_option in valid_states, f"Selected option {selected_option} not in valid states {valid_states}"
        cur_info = self.get_check_relation_info(relation, valid_states, selected_option, messages, response)

        cur_info_list = [cur_info]
        return selected_option, cur_info_list

    def _get_check_relation_option(self, shape, messages, relation=None, valid_states=None, n_tries=0):
        selected_option, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_check_relation, error_template_map=None)
        return selected_option, response

    def get_check_relation_info(self, relation, valid_states, selected_option, messages, response):
        prompt = messages[1]['content']
        cur_info = {
            'step': 'check_relation',
            'relation': relation.relation_index,
            'valid_states': [x for x in valid_states],
            'selected_option': selected_option,
            'prompt': prompt,
            'response': response
        }
        return cur_info

    def create_check_relation_prompt(self, shape, edit_request, relation, valid_states, resolved_new_edits=None, nocot=False):
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
        if nocot:
            template = check_relation.instructions_nocot
        else:
            template = check_relation.instructions
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages

    def parse_check_relation(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        selected_option = snippet_to_variable(snippet,shape, "selected_option")
        # summary = snippet_to_variable(snippet,shape, "summary")
        return selected_option
    
    def api_call_select_edit(self, shape, edit_request, all_edits, edit_options, part_to_edit, type_hints=None):
        # Check if a relationship should be maintained or not.
        cur_info_list = []
        print("NUMBER OF OPTIONS:", len(edit_options))
        unique_option_classes, ind_to_unique_class, ind_to_options = self.partition_options(edit_options)
        n_options = len(edit_options)
        if n_options > self.n_options_to_consider:

            messages = self.create_edit_type_prompt(shape, edit_request, part_to_edit,
                                                    unique_option_classes, ind_to_unique_class, ind_to_options)
            selected_ind, response = self._get_edit_class(shape, messages, unique_option_classes, part_to_edit)
            
            cur_info = self.get_edit_type_info(part_to_edit, unique_option_classes, selected_ind, messages, response)
            cur_info_list.append(cur_info)

            selected_ind = selected_ind - 1
            # update edit_options and unique_option_classes
            edit_options = ind_to_options[selected_ind]
            # API CALL to to select one of the options.
            unique_option_classes = list(set([x.edit_class.__name__ for x in edit_options]))
            unique_option_classes.sort()
            unique_option_classes = unique_option_classes[::-1]
        if len(edit_options) == 1:
            # No need to ask for the type of edit.
            selected_ind = 0
        else:
        # local information for the part
            messages = self.create_edit_selection_prompt(shape, edit_request, edit_options, part_to_edit, unique_option_classes)
            selected_ind, response = self._get_edit_option(shape, messages, edit_options, part_to_edit)
            cur_info = self.get_edit_option_info(part_to_edit, edit_options, selected_ind, messages, response)
            cur_info_list.append(cur_info)
            selected_ind = selected_ind - 1
        new_edit_candidate = edit_options[selected_ind]
        new_edit = new_edit_candidate.employ(operand=part_to_edit)

        return new_edit, cur_info_list
    
    def _get_edit_option(self, shape, messages, edit_options=None, part_to_edit=None, n_tries=0):
        selected_ind, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                        parsing_func=self.parse_select_option, error_template_map=None)
        return selected_ind, response
    
    def _get_edit_class(self, shape, messages, unique_option_classes=None, part_to_edit=None, n_tries=0):

        selected_ind, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                        parsing_func=self.parse_select_option, error_template_map=None)
        return selected_ind, response
    
    def get_edit_option_info(self, part_to_edit, edit_options, selected_ind, messages, response):
        prompt = messages[1]['content']
        if isinstance(part_to_edit, Part):
            detail = ("part", part_to_edit.part_index)
        else:
            detail = ("relation", part_to_edit.relation_index)
        cur_info = {
            'step': 'select_edit_option',
            'part_to_edit': detail,
            'edit_options': [x for x in edit_options],
            'selected_ind': selected_ind,
            'prompt': prompt,
            'response': response
        }
        return cur_info
    
    def get_edit_type_info(self, part_to_edit, unique_option_classes, selected_ind, messages, response):
        prompt = messages[1]['content']

        if isinstance(part_to_edit, Part):
            detail = ("part", part_to_edit.part_index)
        else:
            detail = ("relation", part_to_edit.relation_index)
        cur_info = {
            'step': 'select_edit_type',
            'part_to_edit': detail,
            'unique_option_classes': [x for x in unique_option_classes],
            'selected_ind': selected_ind,
            'prompt': prompt,
            'response': response
        }
        return cur_info

    def create_edit_selection_prompt(self, shape, edit_request, edit_options, part_to_edit, unique_option_classes, nocot=False):
        shape_spec = shape_to_file_hier(shape)
        options_str = get_options_in_detail(part_to_edit, edit_options, shape)
        edit_info_str = select_option.get_edit_info_str(unique_option_classes)
        if isinstance(part_to_edit, Part):
            part_in_detail = get_part_in_details(part_to_edit, shape)
        else:
            part_in_detail = get_relation_in_detail(part_to_edit, shape)
        unedited_parts = get_unedited_parts(shape)
        substitute_dict = {"shape_class": shape.label,
                        "shape_specification": shape_spec,
                        "edit_request": edit_request,
                        'part_in_detail': part_in_detail,
                        'unedited_parts': unedited_parts,
                        'options': options_str,
                        'edit_info_str': edit_info_str,
                        }
        if nocot:
            template = select_option.instructions_edit_selection_nocot
        else:
            template = select_option.instructions_edit_selection
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages

    def create_edit_type_prompt(self, shape, edit_request, part_to_edit, 
                                unique_option_classes, ind_to_unique_class, ind_to_options,
                                nocot=False):

        shape_spec = shape_to_file_hier(shape)
        part_in_detail = get_part_in_details(part_to_edit, shape)
        unedited_parts = get_unedited_parts(shape)
            
        edit_info_str = select_option.get_edit_info_str(unique_option_classes)
            # API CALL once to get the unique class index.
        unique_class_options_str = ""
        for ind, unique_class in ind_to_unique_class.items():
            options = ind_to_options[ind]
            if len(options) > 1:
                unique_class_options_str += f"{ind+1}. {unique_class} - (Multiple Options)\n"
            else:
                edit_option = options[0]
                temp_edit = edit_option.employ(operand=part_to_edit)
                option_string = temp_edit.signature(shape)
                unique_class_options_str += f"{ind+1}. {unique_class} - (Single Option) {option_string}\n"


        substitute_dict = {"shape_class": shape.label,
                            "shape_specification": shape_spec,
                            "edit_request": edit_request,
                            'part_in_detail': part_in_detail,
                            'unedited_parts': unedited_parts,
                            'options': unique_class_options_str,
                            'edit_info_str': edit_info_str,
                            }
        if nocot:
            template = select_option.instructions_class_selection_nocot
        else:
            template = select_option.instructions_class_selection
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": instructions_filled}
            ]
        
        return messages

    def partition_options(self, edit_options):
        unique_option_classes = list(set([x.edit_class.__name__ for x in edit_options]))
        unique_option_classes.sort()
        unique_option_classes = unique_option_classes[::-1]
        ind_to_unique_class = {}
        ind_to_options = {}
        for ind, unique_class in enumerate(unique_option_classes):
            ind_to_unique_class[ind] = unique_class
            ind_to_options[ind] = [x for x in edit_options if x.edit_class.__name__ == unique_class]
        return unique_option_classes, ind_to_unique_class, ind_to_options
    
    def parse_select_option(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        selected_ind = snippet_to_variable(snippet,shape, "selected_ind")
        # Is the spec valid?
        
        return selected_ind
    
    def api_call_finish_algo(self, shape, edit_request, all_edits):
        # Check if a relationship should be maintained or not.
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
    
    def _get_end_algo(self, shape, messages, all_edits=None, edit_request=None, n_tries=0):
        algorithm_end, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                            parsing_func=self.parse_finish_algo_1, error_template_map=None)
        return algorithm_end, response

    def _get_new_edit(self, shape, messages, all_edits=None, edit_request=None, n_tries=0):

        new_edit, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                        parsing_func=self.parse_finish_algo_2, error_template_map=None)
        return new_edit, response
    
    def get_new_edit_info(self, new_edit, messages, response):

        edit_info = []
        if isinstance(new_edit, list):
            for edit in new_edit:
                if isinstance(edit, PartEdit):
                    edit_item =  (EditGen(edit.__class__, edit.params), edit.operand.part_index)
                else:
                    edit_item = (EditGen(edit.__class__, edit.params), edit.operand.relation_index)
                edit_info.append(edit_item)
        else:
            edit_info = (EditGen(new_edit.__class__, new_edit.params), new_edit.operand.part_index)
            new_edit = [new_edit]
        
        cur_info = {
            'step': 'new_edit',
            'new_edit': edit_info,
            'prompt': messages[1]['content'],
            'response': response
        }
        return cur_info
    
    def create_finish_algo_prompt(self, all_edits, shape, edit_request, nocot=False):

        shape_spec = shape_to_file_hier(shape)
        unedited_parts = get_unedited_parts(shape)
        all_edits_str = get_all_edits_in_detail(all_edits, shape)

        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_spec,
                           "edit_request": edit_request,
                           "all_edits_in_detail": all_edits_str,
                           "unedited_parts": unedited_parts,
                           }
        if nocot:
            template = finish_algo.instructions_over_or_not_nocot
        else:
            template = finish_algo.instructions_over_or_not
        instructions_filled = Template(template).substitute(substitute_dict)
        # instructions_filled = Template(instructions).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        return messages

    def create_new_edit_prompt(self, all_edits, shape, edit_request, nocot=False):

        shape_spec = shape_to_file_hier(shape)
        unedited_parts = get_unedited_parts(shape)
        all_edits_str = get_all_edits_in_detail(all_edits, shape)



        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_spec,
                           "edit_request": edit_request,
                           "all_edits_in_detail": all_edits_str,
                           "unedited_parts": unedited_parts,
                           "API": INIT_EDIT_API
                           }
        if nocot:
            template = finish_algo.instructions_new_edits_nocot
        else:
            template = finish_algo.instructions_new_edits
        instructions_filled = Template(finish_algo.instructions_new_edits).substitute(substitute_dict)
        # instructions_filled = Template(instructions).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        return messages
    
    def parse_finish_algo_1(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        algorithm_end = snippet_to_variable(snippet,shape, "edit_complete")
        return algorithm_end
    
    def parse_finish_algo_2(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        algorithm_end = snippet_to_variable(snippet,shape, "new_edit")
        return algorithm_end


def response_to_snippet(response):
    message = response# response.message.content
    pattern = r'```python\n(.*?)\n```'
    snippets = re.findall(pattern, message, re.DOTALL)
    if len(snippets) != 1:
        # error_message = f"The number of snippets returned is {len(snippets)}, but should be 1. Specify a single snippet with a code block in python.\n"
        # print(f"Wrong format.\n =========\n {error_message}")
        # raise SyntaxError(error_message)
        # lets just cat then all together
        snippet = "\n".join(snippets)
    else:
        snippet = snippets[0]
    print("LLM output:")
    for x in snippet.split('\n'):
        print(x)
    return snippet

def snippet_to_variable(snippet, shape, variable_name):
    ldict = {}
    global_dict = {}
    global_dict['shape'] = shape
    global_dict['sp'] = sp
    global_dict.update(GEOMETRIC_ATOMS)
    global_dict.update(EDIT_ATOMS)
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
    

def update_amount(primary_edit):
    operand = primary_edit.operand
    amount_enough = False
    count = 0
    while(not amount_enough):
        primary_edit.propagate()
        static_expr = operand.primitive.static_expression()
        dynamic_expr = operand.primitive.dynamic_expression()
        delta = dynamic_expr - static_expr
        broken = False
        n_points = delta.shape[0]
        for i in range(n_points):
            cur_delta = delta[i, :].norm()
            if not evaluate_equals_zero(cur_delta * 0.5, mode=2, order=1):
                print(cur_delta.subs({MAIN_VAR: MOVE_LIMIT}))
                broken = True
                break
        if broken:
            amount_enough = True
        else:
            print(f"Increase amount of {primary_edit} by 2 to {primary_edit.amount * 2}")
            primary_edit.amount = primary_edit.amount * 2
            count += 1
            if count == 2:
                break
        operand.primitive.edit_sequence = []
    return primary_edit