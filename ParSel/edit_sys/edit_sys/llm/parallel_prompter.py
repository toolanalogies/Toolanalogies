import sqlite3
import multiprocessing as mp
import time
import traceback
from string import Template

from .prompter_v2 import LLMPrompterV2
from ..state import State
from .errors import APIFailure
from .utils import DB_FILE, parallel_make_api_call
from edit_sys.shape_system.edits import HIGHER_TYPE_HINTS
from edit_sys.shape_system.edits import KeepFixed, PartEdit, EditGen
from edit_sys.shape_system.relations import PrimitiveRelation, ReflectionSymmetry
from .base_prompter import update_amount

from edit_sys.shape_system.final_annotation import (shape_to_file_hier,
                                                    get_relation_in_detail_init)
from .prompts.v7 import check_relation_init
from .prompts.v7 import check_relation

class ParallelLLMPrompterV2(LLMPrompterV2):
    

    def api_call_initialize_parallel(self, shape, edit_request, prompter_info):
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
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        outputs = []
        while not queue.empty():
            outputs.append(queue.get())
        end_time = time.time()
        print(f"Time taken for parallel call: {end_time - st}")
        State.api_time += time.time() - start_time
        fixed_parts = [x[0] for x in outputs if x[1]['step'] == "KF"][0]
        primary_edit_gens = [x[0] for x in outputs if x[1]['step'] == "INIT"][0]
        init_type_hinting = [x[0] for x in outputs if x[1]['step'] == "TYPE"][0]

        new_edits = self.process_editgens(shape, fixed_parts, primary_edit_gens)

        # update state.
        update_state(n_rows)
        cur_info_list = []

        return new_edits, init_type_hinting, cur_info_list


    def process_editgens(self, shape, fixed_parts, primary_edit_gens):
        keep_fixed_edits = [KeepFixed(operand=shape.get(part)) for part in fixed_parts]
        primary_edits = []
        for edit_item in primary_edit_gens:
            edit_gen, operand_type, index = edit_item
            if operand_type == "part":
                operand = [x for x in shape.partset if x.part_index == index]
                if len(operand) > 0:
                    operand = operand[0]
                else:
                    continue
            else:
                operand = [x for x in shape.all_relations() if x.relation_index == index][0]
            edit = edit_gen.employ(operand)
            primary_edits.append(edit)
        new_edits = keep_fixed_edits + primary_edits
        return new_edits


    def _get_fixed_parts(self, shape, messages, n_tries=0):
        fixed_parts, content = self._parallel_error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                        parsing_func=self.parse_fixed_parts)
        # extract all there is to extract from the response
        extracted_info = {}
        return fixed_parts, content

    def _get_primary_edits(self, shape, messages, n_tries=0):
        primary_edits, content = self._parallel_error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_initial_instructions)
        extracted_info = {}
        return primary_edits, content
    
    def _get_init_edit_hints(self, shape, messages, n_tries=0):
        edit_type_hints, content = self._parallel_error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_init_edit_hints)
        extracted_info = {}
        return edit_type_hints, content
    
    def _parallel_error_tolerant_call(self, shape, messages, max_tries, n_tries, parsing_func):
        if n_tries >= max_tries:
            print("FAILURE!!")
            raise APIFailure(f"Failure despite {n_tries} tries.")
        content, role, response_valid_finish = parallel_make_api_call(mode=self.mode, client=self.client, model=self.model, messages=messages, temperature=self.temperature, seed=self.seed)
        # response_finish_reason = response.choices[0].finish_reason
        if not response_valid_finish:
            print(content)
            raise Exception("Error in finishing GPT response.")
        try:
            edit_instruction_set = parsing_func(shape, content)
        except Exception as ex:
            error_template =  "There is a error in your code. Please try again.\nHere is the error message:\n $error_message"
            error_message = Template(error_template).substitute(error_message=str(traceback.print_stack()))

            messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": error_message})
            edit_instruction_set, content = self._parallel_error_tolerant_call(shape,
                messages, max_tries=max_tries, n_tries=n_tries + 1, 
                parsing_func=parsing_func)
        return edit_instruction_set, content


class UpfrontParallelLLMPrompterV2(ParallelLLMPrompterV2):
    """ Call the LLM for relations initially itself.
    Issue - Without the local context it might be wrong.
    """

    def api_call_initialize_parallel(self, shape, edit_request, prompter_info):
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

        new_edits = self.process_editgens(shape, fixed_parts, primary_edit_gens)

        all_relation_preds = [x[0] for x in outputs if x[1]['step'] == "check_relation"]
        relation_to_option = {}
        for relation_pred in all_relation_preds:
            relation_to_option[relation_pred[0]] = relation_pred[1]

        self.relation_to_option = relation_to_option
        # update state.
        update_state(n_rows)
        cur_info_list = []

        return new_edits, init_type_hinting, cur_info_list

    def api_call_check_relation(self, shape, edit_request, all_edits, relation, valid_states, resolved_new_edit=None, type_hints=None):
        if not relation.relation_index in self.relation_to_option:
            self.relation_to_option[relation.relation_index] = 1
        if isinstance(relation, PrimitiveRelation):
            relation_index = relation.relation_index
            selected_option = self.relation_to_option[relation_index]
            if not isinstance(relation, ReflectionSymmetry):
                if 2 in valid_states:
                    if selected_option == 1:
                        selected_option = 2
                    
            if not selected_option in valid_states:
                if selected_option == 2:
                    selected_option = 1
                elif selected_option == 1:
                    selected_option = 2
                else:
                    selected_option = valid_states[0]
            cur_info = {"step": "check_relation"}
            cur_info_list = [cur_info]
        else:
            selected_option = min(valid_states)
            cur_info_list = [{"step": "check_relation", "response": ""}]
        return selected_option, cur_info_list

    def create_check_relation_prompt(self, shape, edit_request, relation):

        shape_specification = shape_to_file_hier(shape)
        relation_in_detail = get_relation_in_detail_init(relation, shape)
        if isinstance(relation, ReflectionSymmetry):
            valid_states = [0, 1]
        else:
            valid_states = [0, 1, 2]

        option_tips = check_relation_init.get_option_tips(relation, valid_states)
        options_string = check_relation_init.relation_option_set_to_string(relation, valid_states)
        
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "relation_in_detail": relation_in_detail,
                           "options_string": options_string,
                           "option_tips": option_tips,
                           }
        template = check_relation_init.instructions()
        instructions_filled = Template(template).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages


def get_n_rows():
    # con = sqlite3.connect(DB_FILE)
    # cur = con.cursor()
    # cur.execute("CREATE TABLE IF NOT EXISTS api_calls (call_model TEXT, time_of_call TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER)")
    # cur.execute("SELECT COUNT(*) FROM api_calls")
    # n_rows = cur.fetchall()[0][0]
    # con.close()
    # return n_rows
    return 0

def update_state(n_rows):
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS api_calls (call_model TEXT, time_of_call TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER)")
    # Get the content above the table
    cur.execute(f"SELECT * FROM api_calls WHERE rowid > {n_rows}")
    new_rows = cur.fetchall()
    con.close()
    State.n_api_calls += len(new_rows)
    State.n_prompt_tokens += sum([x[2] for x in new_rows])
    State.n_completion_tokens += sum([x[3] for x in new_rows])
    
def parallel_call_func(prompter_class, prompter_info, shape, edit_request, mode, queue):
    
    prompter = prompter_class(**prompter_info)
    if mode % 3 == 0:
        messages = prompter.create_keep_fixed_prompt(shape, edit_request)
        fixed_parts, content = prompter._get_fixed_parts(shape, messages)
        # keep_fixed_info = prompter.get_keep_fixed_info(shape, keep_fixed_edits, messages, response)
        output = fixed_parts, {"step": "KF"}
    elif mode % 3 == 1:
        
        messages = prompter.create_initialize_prompt(shape, edit_request)
        primary_edits, response = prompter._get_primary_edits(shape, messages)
        if not isinstance(primary_edits, list):
            if isinstance(primary_edits, PartEdit):
                update_amount(primary_edits)
            primary_edits = [primary_edits]
        edit_spec = []
        for edit in primary_edits:
            if isinstance(edit, PartEdit):
                edit_item =  (EditGen(edit.__class__, edit.params), 'part', edit.operand.part_index)
            else:
                edit_item = (EditGen(edit.__class__, edit.params), 'relation', edit.operand.relation_index)
            edit_spec.append(edit_item)
        # primary_edit_info = prompter.get_primary_edit_info(primary_edits, messages, response)
        output = edit_spec, {"step": "INIT"}
    elif mode % 3 == 2:
        messages = prompter.create_init_edit_hint_prompt(shape, edit_request)
        edit_type_hints, response = prompter._get_init_edit_hints(shape, messages)
        # cur_info = {
        #     'step': 'get_edit_hints',
        #     'edit_type_hints': edit_type_hints,
        #     'prompt': messages[1]['content'],
        #     'response': response,
        #     'type_hints': edit_type_hints
        # }
        full_types = {}
        for name, edit_type in edit_type_hints.items():
            if edit_type in ['change_count', "change_delta"]:
                m = shape.get(name)
                if hasattr(m, 'core_relation'):
                    relation = m.core_relation
                else:
                    relation = m.parent.core_relation
                full_types[relation.full_label] = edit_type
                edit_type = "scale"
            full_types[shape.get(name).full_label] = edit_type
        output = full_types, {"step": "TYPE"}
    
    queue.put(output)

def relation_parallel_call(prompter_class, prompter_info, shape, edit_request, relation_index, queue):
    
    prompter = prompter_class(**prompter_info)
    relation = [x for x in shape.all_relations() if x.relation_index == relation_index][0]

    messages = prompter.create_check_relation_prompt(shape, edit_request, relation)
    selected_option, response = prompter._get_check_relation_option(shape, messages, None, [1, 2, 3])
    relation_spec = (relation_index, selected_option)
    output = relation_spec, {"step": "check_relation"}
    queue.put(output)