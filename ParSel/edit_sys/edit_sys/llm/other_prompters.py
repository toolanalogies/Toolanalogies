"""Depreciated
"""
from edit_sys.shape_system import Part, PrimitiveRelation

from .base_prompter import LLMPrompterV1 as BaseLLMPrompter
from edit_sys.shape_system.final_annotation import (shape_to_file_hier,
                                                    get_relation_in_detail,
                                                    get_part_in_details,
                                                    get_unedited_parts,
                                                    get_options_in_detail)  
from edit_sys.shape_system.proposer_utils import get_typed_distortion_energies, RELATION_EDIT_ENERGY
from edit_sys.shape_system.edits import ChangeDelta, ChangeCount
from string import Template

class PureGeometryPrompter(BaseLLMPrompter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_keep_fixed = True
        self.n_options_to_consider = 10000

    def _get_check_relation_option(self, shape, messages, relation=None, valid_states=None, n_tries=0):
        variable = 1
        if not variable in valid_states:
            if 2 in valid_states:
                variable = 2
            else:
                variable = 0

        # variable = 1
        print(f"Checking relation: {relation}")
        print(f"Selected option: {variable}")
        response = ''
        return variable, response

    def _get_edit_option(self, shape, messages, edit_options, part_to_edit, n_tries=0):
        print("Selecting Edit:")
        print(f"Operand: {part_to_edit}")

        if isinstance(part_to_edit, Part):
            distortion_energies = get_typed_distortion_energies(shape, part_to_edit, edit_options)
        else:
            distortion_energies = [RELATION_EDIT_ENERGY[x.edit_class] for x in edit_options]

        min_energy = min(distortion_energies)
        min_index = distortion_energies.index(min_energy)
        selected_ind = min_index
        print(f"Selected option: {selected_ind}")
        print(f"Energy: {min_energy}")
        print(f"Selected Edit: {edit_options[selected_ind]}")
        response = ''
        return selected_ind, response
    
    # def _get_end_algo(self, shape, messages, all_edits, edit_request, n_tries=0):
    #     algorithm_end = True
    #     response = ''
    #     return algorithm_end, response

    
class NoEditSelectLLMPrompter(BaseLLMPrompter):

    def _get_check_relation_option(self, shape, messages, relation=None, valid_states=None, n_tries=0):
        variable = 1
        if not variable in valid_states:
            if 2 in valid_states:
                variable = 2
            else:
                variable = 0

        # variable = 1
        response = ''
        return variable, response
    
class NoRelationSelectLLMPrompter(BaseLLMPrompter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_keep_fixed = True
        self.n_options_to_consider = 10000

    def _get_edit_option(self, shape, messages, edit_options, part_to_edit, n_tries=0):
        print("Selecting Edit:")
        print(f"Operand: {part_to_edit}")

        if isinstance(part_to_edit, Part):
            distortion_energies = get_typed_distortion_energies(part_to_edit, edit_options)
        else:
            distortion_energies = [RELATION_EDIT_ENERGY[x.edit_class] for x in edit_options]
        min_energy = min(distortion_energies)
        min_index = distortion_energies.index(min_energy)
        selected_ind = min_index
        response = ''
        return selected_ind, response

class NoEndLLMPrompter(BaseLLMPrompter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_keep_fixed = True
        self.n_options_to_consider = 10000

    def _get_end_algo(self, shape, messages, all_edits, edit_request, n_tries=0):
        algorithm_end = True
        response = ''
        return algorithm_end, response


class VisionAugmentedLLMPrompter(BaseLLMPrompter):
    ...

class NoAbstractionLLMPrompter(BaseLLMPrompter):
    """
    Give it everything in full details.
    Not particularly interesting.
    """
    ...
