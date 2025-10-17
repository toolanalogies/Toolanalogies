from pathlib import Path
import time
import sys
import os
import _pickle as cPickle
import multiprocessing as mp
from ..state import State
from edit_sys.shape_system.edits import PartEdit, EditGen, KeepFixed, HIGHER_TYPE_HINTS
from .parallel_prompter import ParallelLLMPrompterV2, UpfrontParallelLLMPrompterV2, get_n_rows, update_state
from .parallel_prompter import parallel_call_func, relation_parallel_call
from edit_sys.shape_system.relations import PrimitiveRelation
from edit_sys.shape_system.final_annotation import (shape_to_file_hier,
                                                    get_relation_in_detail_init,
                                                    get_part_in_details,
                                                    get_unedited_parts,
                                                    generate_least_breaking_str,
                                                    get_all_edits_in_detail,
                                                    get_options_in_detail)  

from .prompts.v8 import initialize
from .prompts.v8 import set_edit_types
from .prompts.v8 import set_relations
from string import Template
from edit_sys.shape_system import INIT_EDIT_API, EDIT_API, SHAPE_API
from .base_prompter import APIFailure, response_to_snippet, snippet_to_variable, update_amount
from edit_sys.shape_system.relations import ReflectionSymmetry, TranslationSymmetry, RotationSymmetry
from collections import defaultdict
import numpy as np

class UpfrontParallelLLMPrompterV3(UpfrontParallelLLMPrompterV2):

    def __init__(self, mode, key, model, temperature=0.0, seed=42, max_tries=2, repetitions=3, internal_save_path=None, *args, **kwargs):
        super(UpfrontParallelLLMPrompterV3, self).__init__(mode, key, model, temperature, seed, max_tries, *args, **kwargs)
        self.repetitions = repetitions
        if repetitions == 5:
            self.min_threshold = 3/5.0
        elif repetitions == 3:
            self.min_threshold = 1.5/3.0
        elif repetitions == 1:
            self.min_threshold = 0.5            
        self.internal_save_path = internal_save_path
            
    def create_new_init_prompt(self, shape, edit_request):
        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "SHAPE_API": SHAPE_API,
                           "EDIT_API": EDIT_API,
                           "API": INIT_EDIT_API}
        
        template = initialize.get_instructions()
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages
    
    def create_new_edit_types_prompt(self, shape, edit_request):
        
        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request}
        template = set_edit_types.get_instructions()
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages

    def _get_new_edit_types(self, shape, messages, n_tries=0):
        edit_type_hints, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_init_edit_hints, error_template_map=None)
        return edit_type_hints, response

    def parse_init_edit_hints(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        edit_hints = snippet_to_variable(snippet,shape, "type_hints")
        for key, value in edit_hints.items():
            try:
                m = shape.get(key)
            except:
                error_message = f"The part {key} is not present in the shape. Please correctly specify the part's *full label*."
                print(f"Wrong format.\n =========\n {error_message}")
                raise SyntaxError(error_message)
        
        edit_hints = {shape.get(key).full_label: value for key, value in edit_hints.items()}

        return edit_hints
    
    
    def create_check_relation_prompt(self, shape, edit_request, relation, valid_states=None, resolved_new_edits=None):

        shape_specification = shape_to_file_hier(shape)
        relation_in_detail = get_relation_in_detail_init(relation, shape)

        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "relation_in_detail": relation_in_detail,
                           }

        if isinstance(relation, ReflectionSymmetry):
            template = set_relations.ref_instructions()
        elif isinstance(relation, (TranslationSymmetry, RotationSymmetry)):
            template = set_relations.rot_trans_instructions()

        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        return messages
    
    def _get_check_relation_option(self, shape, messages, relation=None, valid_states=None, n_tries=0):
        selected_option, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_check_relation, error_template_map=None)
        return selected_option, response

    def parse_check_relation(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        selected_option = snippet_to_variable(snippet,shape, "relation_state")
        # summary = snippet_to_variable(snippet,shape, "summary")
        return selected_option
    
    
    def api_call_initialize_parallel(self, shape, edit_request, prompter_info):
        outputs = self._api_call_initialize_parallel(shape, edit_request, prompter_info)
        fixed_parts, init_type_hinting, primary_edit_gens, relation_to_option = outputs
        self.relation_to_option = relation_to_option
        new_edits = []
        for part_name in fixed_parts:
            try:
                part = shape.get(part_name)

                new_edits.append(KeepFixed(part))
            except:
                print(f"Could not find part {part_name}. Please carefully read the guildlines and try again.")
        print('primary_edit_gens:', primary_edit_gens)
        for edit_spec in primary_edit_gens:
            if edit_spec[1] == "part":
                part_index = edit_spec[2]
                print("all indices in shape.partset:", [p.part_index for p in shape.partset])
                print("looking for part_index:", part_index)
                part = [x for x in shape.partset if x.part_index == part_index][0]
                new_edits.append(edit_spec[0].employ(part))
            else:
                relation_index = edit_spec[2]
                relation = [x for x in shape.all_relations() if x.relation_index == relation_index][0]
                new_edits.append(edit_spec[0].employ(relation))
        
        cur_info_list = []
        return new_edits, init_type_hinting, cur_info_list

    def _api_call_initialize_parallel(self, shape, edit_request, prompter_info):
        # reverse get the 

        n_rows = get_n_rows()
        
        start_time = time.time()
        queue = mp.Manager().Queue()
        processes = []
        st = time.time()
        prompter_class= self.__class__
        for cur_mode in range(2):
            
            for j in range(self.repetitions):
                args = (prompter_class, prompter_info, shape, edit_request, cur_mode, queue)
                p = mp.Process(target=parallel_call_func_v3, args=args)
                processes.append(p)

        ### For each Primitive relation
        for relation in shape.all_relations():
            if isinstance(relation, PrimitiveRelation):
                # print("hello")
                # FOr reflection its only 0, 1.
                # For translation, Rotation its 0, 1, 2
                for j in range(self.repetitions):
                    args = (prompter_class, prompter_info, shape, edit_request, relation.relation_index, queue)
                    p = mp.Process(target=relation_parallel_call, args=args)
                    processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        outputs = []
        while not queue.empty():
            outputs.append(queue.get())
        end_time = time.time()
        print(f"Time taken for parallel call: {end_time - st}")
        State.api_time += end_time - start_time
        init_type_hints = [x[0] for x in outputs if x[1]['step'] == "TYPE_HINTS"]
        if len(init_type_hints) == 1:
            init_type_hints = init_type_hints[0]
        else:
            # TODO: Merging multiple calls
            fixed_score_counter = defaultdict(int)
            
            for init_type_hint in init_type_hints:
                for key, value in init_type_hint.items():
                    score_tuple = (key, value)
                    fixed_score_counter[score_tuple] += 1
                    
            final_selected_type_hints = {}
            n_success = len(init_type_hints)
            for key, value in fixed_score_counter.items():
                if value >= (self.min_threshold * n_success):
                    final_selected_type_hints[key[0]] = key[1]
            init_type_hints = final_selected_type_hints
            
        # parse fixed parts.
        fixed_parts = []
        real_type_hints = {}
        for part_name, edit_type in init_type_hints.items():
            if edit_type == "keep_fixed":
                fixed_parts.append(part_name)
            else:
                real_type_hints[part_name] = edit_type
        
        primary_edit_gens = [x[0] for x in outputs if x[1]['step'] == "INIT"]

        if len(primary_edit_gens) == 1:
            primary_edit_gens = primary_edit_gens[0]
        else:
            # TODO: Merging multiple calls
            edit_voter = defaultdict(int)
            edit_mapper = defaultdict(int)
            for primary_edit_gen in primary_edit_gens:
                cur_sym = ""
                for edit_spec in primary_edit_gen:
                    info = [edit_spec[0].edit_class.__name__, edit_spec[1], str(edit_spec[2])]
                    cur_sym += "_".join(info)

                edit_voter[cur_sym] += 1
                edit_mapper[cur_sym] = [primary_edit_gen]
            count = -1
            # May need secondary. So vote should be for sets.
            primary_edit_gens = None
            for key, value in edit_voter.items():
                if value >= count:
                    primary_edit_gens = edit_mapper[key]
                    count = value
            if isinstance(primary_edit_gens[0], list):
                primary_edit_gens = primary_edit_gens[0]


        all_relation_preds = [x[0] for x in outputs if x[1]['step'] == "check_relation"]
        relation_to_option = defaultdict(list)
        for relation_pred in all_relation_preds:
            relation_to_option[relation_pred[0]].append(relation_pred[1])
        real_relation_pred = {}
        for key, value in relation_to_option.items():
            unique, count = np.unique(value, return_counts=True)
            real_relation_pred[key] = unique[np.argmax(count)]
            relation = [x for x in shape.all_relations() if x.relation_index == key][0]
            print(f"Relation {relation} has been set to {real_relation_pred[key]}")
        relation_to_option = real_relation_pred
        
        # update state.
        # import pdb
        # pdb.set_trace()
        #update_state(n_rows)
        cur_info_list = []
        # self.save_info(fixed_parts, real_type_hints, primary_edit_gens, relation_to_option)

        return fixed_parts, real_type_hints, primary_edit_gens, relation_to_option
    
    def save_info(self, fixed_parts, real_type_hints, primary_edit_gens, relation_to_option):

        data = {
            'fixed_parts': fixed_parts,
            'primary_edits': primary_edit_gens,
            'relation_map': relation_to_option,
            'init_edit_type_hints': real_type_hints,
        }
        # Figure this part out.
        assert not "GT" in self.internal_save_path

        cPickle.dump(data, open(self.internal_save_path, "wb"))
        # exit
        # sys.exit(0)

def parallel_call_func_v3(prompter_class, prompter_info, shape, edit_request, mode, queue):
    
    prompter = prompter_class(**prompter_info)
    if mode % 2 == 0:
        messages = prompter.create_new_edit_types_prompt(shape, edit_request)
        fixed_parts, content = prompter._get_new_edit_types(shape, messages)
    
        for part in shape.partset:
            if len(part.partset) > 0:
                if part.full_label in fixed_parts:
                    for child in part.partset:
                        fixed_parts[child.full_label] = fixed_parts[part.full_label]
                else:
                    child_annotation = []
                    for child in part.partset:
                        child_annotation.append(fixed_parts.get(child.full_label, None))
                    if len(set(child_annotation)) == 1:
                        fixed_parts[part.full_label] = child_annotation[0]
                if hasattr(part, "core_relation"):
                    fixed_parts[part.core_relation.full_label] = fixed_parts[part.full_label]
            elif len(part.partset) == 0:
                ... 
        # keep_fixed_info = prompter.get_keep_fixed_info(shape, keep_fixed_edits, messages, response)
        output = fixed_parts, {"step": "TYPE_HINTS"}
    elif mode %2 == 1:
        
        messages = prompter.create_new_init_prompt(shape, edit_request)
        primary_edits, response = prompter._get_primary_edits(shape, messages)
        
        if not isinstance(primary_edits, list):
            if isinstance(primary_edits, PartEdit):
                update_amount(primary_edits)
            primary_edits = [primary_edits]

        edit_spec = []
        for edit in primary_edits:
            if isinstance(edit, PartEdit):
                edit_item =  (EditGen(edit.__class__, edit.params, amount=edit.amount), 'part', edit.operand.part_index)
            else:
                edit_item = (EditGen(edit.__class__, edit.params, amount=edit.amount), 'relation', edit.operand.relation_index)
            edit_spec.append(edit_item)
        # primary_edit_info = prompter.get_primary_edit_info(primary_edits, messages, response)
        output = edit_spec, {"step": "INIT"}
    
    queue.put(output)


class NoExtraUFPLLMPrompterV3(UpfrontParallelLLMPrompterV3):

    def _api_call_initialize_parallel(self, shape, edit_request, prompter_info):
        # reverse get the 
        n_rows = get_n_rows()
        start_time = time.time()
        queue = mp.Manager().Queue()
        processes = []
        st = time.time()
        prompter_class= self.__class__
        for j in range(self.repetitions):
            args = (prompter_class, prompter_info, shape, edit_request, queue)
            p = mp.Process(target=parallel_call_func_v4, args=args)
            processes.append(p)

        ### For each Primitive relation
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        outputs = []
        while not queue.empty():
            outputs.append(queue.get())
        end_time = time.time()
        print(f"Time taken for parallel call: {end_time - st}")
        State.api_time += end_time - start_time
        fixed_parts = []
        real_type_hints = {}

        primary_edit_gens = [x[0] for x in outputs if x[1]['step'] == "INIT"]

        if len(primary_edit_gens) == 1:
            primary_edit_gens = primary_edit_gens[0]
        else:
            # TODO: Merging multiple calls
            edit_voter = defaultdict(int)
            edit_mapper = defaultdict(int)
            for primary_edit_gen in primary_edit_gens:
                for edit_spec in primary_edit_gen:
                    edit_voter[(edit_spec[0].edit_class, edit_spec[1], edit_spec[2])] += 1
                    edit_mapper[(edit_spec[0].edit_class, edit_spec[1], edit_spec[2])] = [edit_spec]
            count = -1
            primary_edit_gens = None
            for key, value in edit_voter.items():
                if value >= count:
                    primary_edit_gens = edit_mapper[key]
                    count = value

        relation_to_option = {}
        for relation in shape.all_relations():
            relation_to_option[relation.relation_index] = 1
        
        # update state.
        #update_state(n_rows)
        return fixed_parts, real_type_hints, primary_edit_gens, relation_to_option
    

def parallel_call_func_v4(prompter_class, prompter_info, shape, edit_request, queue):
    
    prompter = prompter_class(**prompter_info)
    messages = prompter.create_new_init_prompt(shape, edit_request)
    primary_edits, response = prompter._get_primary_edits(shape, messages)
    
    if not isinstance(primary_edits, list):
        if isinstance(primary_edits, PartEdit):
            update_amount(primary_edits)
        primary_edits = [primary_edits]

    edit_spec = []
    for edit in primary_edits:
        if isinstance(edit, PartEdit):
            edit_item =  (EditGen(edit.__class__, edit.params, amount=edit.amount), 'part', edit.operand.part_index)
        else:
            edit_item = (EditGen(edit.__class__, edit.params, amount=edit.amount), 'relation', edit.operand.relation_index)
        edit_spec.append(edit_item)
    # primary_edit_info = prompter.get_primary_edit_info(primary_edits, messages, response)
    output = edit_spec, {"step": "INIT"}
    queue.put(output)

class NonParallel(UpfrontParallelLLMPrompterV3):



    def _api_call_initialize_parallel(self, shape, edit_request, prompter_info):

        # messages = self.create_new_edit_types_prompt(shape, edit_request)
        # fixed_parts, content = self._get_new_edit_types(shape, messages)
    
        # for part in shape.partset:
        #     if len(part.partset) > 0:
        #         if part.full_label in fixed_parts:
        #             for child in part.partset:
        #                 fixed_parts[child.full_label] = fixed_parts[part.full_label]
        #         else:
        #             child_annotation = []
        #             for child in part.partset:
        #                 child_annotation.append(fixed_parts.get(child.full_label, None))
        #             if len(set(child_annotation)) == 1:
        #                 fixed_parts[part.full_label] = child_annotation[0]
        #     elif len(part.partset) == 0:
                # ... 

        messages = self.create_new_init_prompt(shape, edit_request)
        primary_edits, response = self._get_primary_edits(shape, messages)
        
        if not isinstance(primary_edits, list):
            if isinstance(primary_edits, PartEdit):
                update_amount(primary_edits)
            primary_edits = [primary_edits]

        edit_spec = []
        for edit in primary_edits:
            if isinstance(edit, PartEdit):
                edit_item =  (EditGen(edit.__class__, edit.params, amount=edit.amount), 'part', edit.operand.part_index)
            else:
                edit_item = (EditGen(edit.__class__, edit.params, amount=edit.amount), 'relation', edit.operand.relation_index)
            edit_spec.append(edit_item)
        # primary_edit_info = prompter.get_primary_edit_info(primary_edits, messages, response)
        output = edit_spec, {"step": "INIT"}


        for relation in shape.all_relations():
            relation_index = relation.relation_index
            relation = [x for x in shape.all_relations() if x.relation_index == relation_index][0]

            messages = self.create_check_relation_prompt(shape, edit_request, relation)
            selected_option, response = self._get_check_relation_option(shape, messages, None, [1, 2, 3])
            relation_spec = (relation_index, selected_option)
            output = relation_spec, {"step": "check_relation"}

class PreLoadV3(UpfrontParallelLLMPrompterV3):


    def _api_call_initialize_parallel(self, shape, edit_request, prompter_info):
        # reverse get the 
        data = cPickle.load(open(self.internal_save_path, "rb"))
        fixed_parts = data['fixed_parts']
        real_type_hints = data['init_edit_type_hints']
        primary_edit_gens = data['primary_edits']
        relation_to_option = data['relation_map']
        # Just remove the types
        # fixed_parts = []
        # real_type_hints = {}
        # relation_to_option = {x:1 for x in relation_to_option}

        return fixed_parts, real_type_hints, primary_edit_gens, relation_to_option
    
class TESTERV3(UpfrontParallelLLMPrompterV3):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relation_count = 0
        self.edit_type_count = 0
        self.end_counter = 1
        self.internal_save_path = kwargs.get('internal_save_path', None)

    def __api_call_initialize_parallel(self, shape, edit_request, prompter_info):
        # reverse get the 
        n_rows = get_n_rows()
        
        start_time = time.time()
        queue = mp.Manager().Queue()
        processes = []
        st = time.time()
        prompter_class= self.__class__
        for cur_mode in range(3):
            args = (prompter_class, prompter_info, shape, edit_request, cur_mode, queue)
            p = mp.Process(target=parallel_call_func, args=args)
            processes.append(p)

        ### For each Primitive relation
        for relation in shape.all_relations():
            if isinstance(relation, PrimitiveRelation):
                # FOr reflection its only 0, 1.
                # For translation, Rotation its 0, 1, 2
                args = (prompter_class, prompter_info, shape, edit_request, relation.relation_index, queue)
                p = mp.Process(target=relation_parallel_call, args=args)
                processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        outputs = []
        while not queue.empty():
            outputs.append(queue.get())
        end_time = time.time()
        print(f"Time taken for parallel call: {end_time - st}")
        State.api_time += end_time - start_time
        fixed_parts = [x[0] for x in outputs if x[1]['step'] == "KF"][0]
        primary_edit_gens = [x[0] for x in outputs if x[1]['step'] == "INIT"][0]
        init_type_hinting = [x[0] for x in outputs if x[1]['step'] == "TYPE"][0]

        all_relation_preds = [x[0] for x in outputs if x[1]['step'] == "check_relation"]
        relation_to_option = {}
        for relation_pred in all_relation_preds:
            relation_to_option[relation_pred[0]] = relation_pred[1]

        # update state.
        #update_state(n_rows)
        cur_info_list = []

        return fixed_parts, init_type_hinting, primary_edit_gens, relation_to_option
    
    def api_call_initialize_parallel(self, shape, edit_request, prompter_info):
        # Do the task. Save it, print it and exit. 
        # get all the answers.
        fixed_parts, init_type_hinting, primary_edit_gens, relation_to_option = self._api_call_initialize_parallel(shape, edit_request, prompter_info)

        edited_part_index = [x[2] for x in self.edit_program if x[1] == 'part']
        self.load_internal_state()
        
        # fixed_parts = [x.operand.full_label for x in new_edits if isinstance(x, KeepFixed)]
        fixed_parts = [ shape.get(x).full_label for x in fixed_parts if x != "ground"]
        # All in the GT should be in here. 

        # primary edit
        primary_edit = primary_edit_gens[0]
        # what is primary is on an relation??
        if primary_edit[1] == 'relation':
            gt_edit = [x for x in self.edit_program if x[1] == 'relation' and x[2] == primary_edit[2]]
            if len(gt_edit) == 0:
                init_rate = 0.0
            else:
                gt_edit = gt_edit[0]
                if issubclass(primary_edit.__class__, gt_edit.__class__):
                    init_rate = 1.0
                else:
                    init_rate = 0.0
        else:

            gt_edit = [x for x in self.edit_program if x[1] == 'part' and x[2] == primary_edit[2]]
            if len(gt_edit) == 0:
                init_rate = 0.0
            else:
                gt_edit = gt_edit[0]
                part = [x for x in shape.partset if x.part_index == primary_edit[2]][0]
                pred_edit = primary_edit[0].employ(part)
                gt_edit = gt_edit[0].employ(part)
                pred_edit.amount = gt_edit.amount
                if pred_edit.merge_equal(gt_edit):
                    init_rate = 1.0
                else:
                    init_rate = 0.0
        print(f"Init Rate: {init_rate}")

        positive = 0
        negative = 0
        gt_fixed_parts = self.fixed_parts
        for part in gt_fixed_parts:
            if part not in fixed_parts:
                negative += 1
            else:
                positive += 1
        # edited parts
        for part in fixed_parts:
            if part not in gt_fixed_parts:
                part_index = shape.get(part).part_index
                if part_index in edited_part_index:
                    negative += 1
                else:
                    positive += 1
        if positive + negative == 0:
            final_kp_rate = "NIL"
        else:
            final_kp_rate = positive / (positive + negative)
        print(f"Final KP Rate: {final_kp_rate}")

        # Edit Type hints
        th_positive = 0
        th_negative = 0
        # for key, value in self.init_edit_type_hints.items():
        #     if value == "translate":
        #         self.init_edit_type_hints[key] == "move"
        #     elif value in ['rotate', 'shear']:
        #         self.init_edit_type_hints[key] == "tilt"
        for part_name, type_hint in init_type_hinting.items():
            gt_type_hint = self.init_edit_type_hints.get(part_name, None)
            if gt_type_hint:
                if gt_type_hint == type_hint:
                    th_positive += 1
                else:
                    th_negative += 1
            else:
                if "Symmetry" in part_name:
                    continue
                part_index = shape.get(part_name).part_index
                if part_index in edited_part_index:
                    valid_edit_types = HIGHER_TYPE_HINTS[type_hint]
                    gt_edit = [x[0] for x in self.edit_program if x[1] == 'part' and x[2] == part_index][0]
                    if gt_edit.edit_class in valid_edit_types:
                        th_positive += 1
                    else:
                        th_negative += 1
        if th_positive + th_negative == 0:
            final_th_rate = "NIL"
        else:
            final_th_rate = th_positive / (th_positive + th_negative)
        print(f"Final TH Rate: {final_th_rate}")

        # Relations
        relation_positive = 0
        relation_negative = 0
        for relation, option in relation_to_option.items():
            gt_pred = self.relation_map.get(relation, None)
            if gt_pred:
                if gt_pred == option:
                    relation_positive += 1
                else:
                    relation_negative += 1
        if relation_positive + relation_negative == 0:
            final_relation_rate = "NIL"
        else:
            final_relation_rate = relation_positive / (relation_positive + relation_negative)
        print(f"Final Relation Rate: {final_relation_rate}")

        fully_correct = 1.0
        if final_kp_rate != "NIL":
            fully_correct *= final_kp_rate
        if final_th_rate != "NIL":
            fully_correct *= final_th_rate
        if final_relation_rate != "NIL":
            fully_correct *= final_relation_rate
        
        fully_correct = fully_correct * init_rate
        print(f"Fully Correct Rate: {fully_correct}")

        timing_dir = os.path.join(self.output_dir, "statistics")
        Path(timing_dir).mkdir(parents=True, exist_ok=True)
        timing_info_file = os.path.join(timing_dir, f"new_{self.repetitions}.csv")
        
        info_row = [self.dataset_index, init_rate, final_kp_rate, final_th_rate, final_relation_rate, fully_correct]
        if not os.path.exists(timing_info_file):
            with open(timing_info_file, 'w') as fd:
                fd.write("dataset_id, init_edit, keep_fixed, type_hint, relation, fully_correct \n")
        # check if info_row already exists.
        prexisting_info = []
        with open(timing_info_file, 'r') as fd:
            prexisting_info = fd.readlines()

        prexisting_info = [[y.strip() for y in x.split(",")] for x in prexisting_info[1:]]
        prexisting_info = {x[0]:x[1:] for x in prexisting_info}
        prexisting_info[info_row[0]] = info_row[1:]
        with open(timing_info_file, 'w') as fd:
            fd.write("dataset_id, init_edit, keep_fixed, type_hint, relation, fully_correct \n")
            for cur_info in prexisting_info:
                cur_info_row = [cur_info] + prexisting_info[cur_info]
                cur_info_row = [str(x) for x in cur_info_row]
                cur_info_row = ",".join(cur_info_row) + "\n"
                fd.write(cur_info_row)
        sys.exit(0)

    def load_internal_state(self):
        if os.path.exists(self.internal_save_path):
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
