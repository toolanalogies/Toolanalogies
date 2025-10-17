"""
This system is / was used for the manual annotation of edit programs for each (shape, edit request) pair.
The user plays the role of the LLM and provides answers for various prompts. 
"""
from .base_prompter import LLMPrompterV1 as BaseLLMPrompter, update_amount
from edit_sys.shape_system import *


class HumanPrompter(BaseLLMPrompter):
    
    def api_call_get_edit_hints(self, shape, edit_request, new_edits):
        
        messages = ""
        n_tries = 0
        cur_info = {
        }
        cur_info_list = [cur_info]
        full_types = {}

        return full_types, cur_info_list

    def _get_update(self, shape, messages, n_tries=0):
        raise NotImplementedError
    
    def _get_fixed_parts(self, shape, messages, n_tries=0):
        fixed_parts = [    
            "vertical_side_panel_right",
            "vertical_side_panel_center",
            "bottom_panel",
        ]
        response = ''
        # This should be done later.
        return fixed_parts, response

    def _get_primary_edits(self, shape, messages, n_tries=0):
        # Define the new edits here
        part = shape.get("tabletop")
        part = shape.get("seat_surface")
        # arm_horizontal_bar_left_front_face = shape.get("arm_near_vertical_bar_right").face("up")
        # new_edit = Scale(tabletop, tabletop.center(), "contract", RightUnitVector())
        # part_2 = shape.get("back_frame_vertical_bar_right")
        # seat = shape.get("seat_single_surface")
        # part = shape.get("bar_stretchers/bar_stretcher_right")
        new_edit = Scale(part, part.center(), "expand", RightUnitVector(), FrontUnitVector())
        # new_edit = Scale(part, DownUnitVector())
        # new_edit = Rotate(part, part.edge_center("back", "down"), LeftUnitVector())
        #new_edit = Translate(part.face("left"), RightUnitVector())

        primary_edits = new_edit
        # Scale(shape.get("lamp_head_center_right"), shape.get("lamp_head_center_right").center(), "expand")
        # Scale(shape.get("lamp_head_center"), shape.get("lamp_head_center").center(), "expand")
        # Scale(shape.get("lamp_head_center_left"), shape.get("lamp_head_center_left").center(), "expand")
        # Scale(shape.get("lamp_head_left"), shape.get("lamp_head_left").center(), "expand")
        # operand = shape.get("back_surface_vertical_bars")
        # primary_edits = SymGroupEdit(shape.get("lamp_arms"), change_type="count", extend_from="center")
        # primary_edits = SymGroupEdit(shape.get("lamp_covers"), change_type="count", extend_from="center")
        # primary_edits = SymGroupEdit(shape.get("light_bulbs"), change_type="count", extend_from="center")
        response = ''
        return primary_edits, response

    def _get_check_relation_option(self, shape, messages, relation=None, valid_states=None, n_tries=0):

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
        
        # variable = 1
        response = ''
        return variable, response

    def _get_edit_class(self, shape, messages, unique_option_classes=None, 
                        part_to_edit=None, n_tries=0):
        print(messages[1]['content'])

        print("Selecting Edit Class:")
        print(f"Operand: {part_to_edit}")
        for ind, opt in enumerate(unique_option_classes):
            print(f"{ind+1}. {opt}")
        selected_ind = None
        while(selected_ind is None):
            try:
                selected_ind = int(input(f"select?"))
            except:
                print("something went wrong. Try again")
                selected_ind = None

        response = ''
        return selected_ind, response
    
    def _get_edit_option(self, shape, messages, edit_options, part_to_edit, n_tries=0):
        print(messages[1]['content'])
        print("Selecting Edit:")
        print(f"Operand: {part_to_edit}")
        for ind, opt in enumerate(edit_options):
            print(f"{ind+1}. {opt}")
        
        selected_ind = None
        while(selected_ind is None):
            try:
                selected_ind = int(input(f"select?"))
            except:
                print("something went wrong. Try again")
                selected_ind = None

        response = ''
        return selected_ind, response
    
    def _get_end_algo(self, shape, messages, all_edits, edit_request, n_tries=0):

        print(messages[1]['content'])
        print(f"Edit Request: {edit_request}")
        print("Final Edits:")
        for i, edit in enumerate(all_edits):
            print(f"{i+1}. {edit}")
        algorithm_end = None
        while algorithm_end is None:
            algo_input = input(f"Finish Edit? (y/n)")
            if algo_input == "y":
                algorithm_end = True
            elif algo_input == "n":
                algorithm_end = False
            else:
                print("Invalid input. Try again.")
                algorithm_end = None

        response = ''
        return algorithm_end, response
    
    def _get_new_edit(self, shape, messages, all_edits, edit_request, n_tries=0):

        print(messages[1]['content'])
        new_edit = None
        while (new_edit is None):
            try:
                new_edit_str = input(f"New Edit?")
                # OR SPECIFY IT HERE - if it is called only once.

                new_edit = eval(new_edit_str)
            except:
                print("Invalid input. Try again.")
                new_edit = None

        response = ''
        return new_edit, response

    # add things to the wrapper
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
            messages = self.create_keep_fixed_prompt(shape, edit_request)
            keep_fixed_info['prompt_nocot'] = messages[1]['content']
            cur_info_list.append(keep_fixed_info)
        
        messages = self.create_initialize_prompt(shape, edit_request)
        primary_edits, response = self._get_primary_edits(shape, messages)
        
        if not isinstance(primary_edits, list):
            update_amount(primary_edits)
            primary_edits = [primary_edits]
        primary_edit_info = self.get_primary_edit_info(primary_edits, messages, response)
        messages = self.create_initialize_prompt(shape, edit_request, nocot=True)
        primary_edit_info['prompt_nocot'] = messages[1]['content']

        cur_info_list.append(primary_edit_info)
        edits = keep_fixed_edits + primary_edits

        return edits, cur_info_list

    def api_call_check_relation(self, shape, edit_request, all_edits, relation, valid_states, resolved_new_edit=None, type_hints=None):
        # Check if a relationship should be maintained or not.

        print(f"validating relation {relation.signature()}")

        messages = self.create_check_relation_prompt(shape, edit_request, relation, valid_states, resolved_new_edit)

        selected_option, response = self._get_check_relation_option(shape, messages, relation, valid_states)
        assert selected_option in valid_states, f"Selected option {selected_option} not in valid states {valid_states}"
        cur_info = self.get_check_relation_info(relation, valid_states, selected_option, messages, response)
        messages = self.create_check_relation_prompt(shape, edit_request, relation, valid_states, resolved_new_edit, nocot=True)
        cur_info['prompt_nocot'] = messages[1]['content']
        cur_info_list = [cur_info]
        return selected_option, cur_info_list
    
    def api_call_select_edit(self, shape, edit_request, all_edits, edit_options, part_to_edit, type_hints=None):
        # Check if a relationship should be maintained or not.
        cur_info_list = []
        unique_option_classes, ind_to_unique_class, ind_to_options = self.partition_options(edit_options)
        n_options = len(edit_options)
        if n_options > self.n_options_to_consider:

            messages = self.create_edit_type_prompt(shape, edit_request, part_to_edit,
                                                    unique_option_classes, ind_to_unique_class, ind_to_options)
            selected_ind, response = self._get_edit_class(shape, messages, unique_option_classes, part_to_edit)
            
            cur_info = self.get_edit_type_info(part_to_edit, unique_option_classes, selected_ind, messages, response)
            messages = self.create_edit_type_prompt(shape, edit_request, part_to_edit, unique_option_classes, ind_to_unique_class, ind_to_options, nocot=True)
            cur_info['prompt_nocot'] = messages[1]['content']
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
            messages = self.create_edit_selection_prompt(shape, edit_request, edit_options, part_to_edit, unique_option_classes, nocot=True)
            cur_info['prompt_nocot'] = messages[1]['content']
            cur_info_list.append(cur_info)
            selected_ind = selected_ind - 1
        new_edit_candidate = edit_options[selected_ind]
        new_edit = new_edit_candidate.employ(operand=part_to_edit)

        return new_edit, cur_info_list
    
    def api_call_finish_algo(self, shape, edit_request, all_edits):
        # Check if a relationship should be maintained or not.
        cur_info_list = []
        messages = self.create_finish_algo_prompt(all_edits, shape, edit_request)
        algorithm_end, response = self._get_end_algo(shape, messages, all_edits, edit_request)
        new_messages = self.create_finish_algo_prompt(all_edits, shape, edit_request, nocot=True)
        cur_info = {
            'step': 'finish_algo',
            'algorithm_end': algorithm_end,
            'prompt': messages[1]['content'],
            'prompt_nocot': new_messages[1]['content'],
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
            messages = self.create_new_edit_prompt(all_edits, shape, edit_request, nocot=True)
            cur_info['prompt_nocot'] = messages[1]['content']
            cur_info_list.append(cur_info)
        return algorithm_end, new_edit, cur_info_list