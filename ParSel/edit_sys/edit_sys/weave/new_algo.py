from edit_sys.shape_system import *
from .simple_algo import (infer_relation_state,
                          TWO_STEP_CONTACT_CHECK)
from edit_sys.shape_system.proposal_mechanism import EditProposer
from edit_sys.llm.prompter_v2 import LLMPrompterV2
from edit_sys.llm.parallel_prompter import ParallelLLMPrompterV2
from edit_sys.llm.base_prompter import update_amount
from edit_sys.shape_system import Part

MAX_EDIT_SPEC_TRY = 10

def algo_v4(shape:Part, edit_request:str , prompter:LLMPrompterV2, edit_proposer:EditProposer, edit_mode="GEOM"):

    get_edit = EDIT_MAPPER[edit_mode]
    ### Step 0: Initialize
    if edit_mode == "GEOM":
        add_type_hints = True
    else:
        add_type_hints = False
    # Optional - top-bottom fixer.
    #ground_edit = KeepFixed(shape.get('ground'))
    #all_edits = [ground_edit]
    all_edits = []
    any_breaking = False
    # all_edits = []
    log_info = []
    if isinstance(prompter, ParallelLLMPrompterV2):
        prompter_info = {
            "key": prompter.key,
            "model": prompter.model,
            "temperature": prompter.temperature,
            "max_tries": prompter.max_tries,
            'seed': prompter.seed,
            'mode': prompter.mode
        }
        
        new_edits, init_edit_types, cur_info_list = prompter.api_call_initialize_parallel(shape, edit_request, prompter_info)
    else:
        shape, cur_info_list = prompter.api_call_initial_update(shape, edit_request)
        log_info.extend(cur_info_list)
        new_edits, cur_info_list = prompter.api_call_initialize(shape, edit_request)
        log_info.extend(cur_info_list)

        if add_type_hints:
            init_edit_types, cur_info_list = prompter.api_call_get_init_edit_hints(shape, edit_request, all_edits)
            log_info.extend(cur_info_list)
        else:
            init_edit_types = None

    all_edits.extend(new_edits)
    
    algorithm_end = False
    relation_relations = shape.all_relations(only_active=True)
    relation_relations = [x._relations for x in relation_relations]
    # convert list of set into a single set
    relation_relations = set().union(*relation_relations)

    while not algorithm_end:

        # Step 1: Propagate
        for edit in all_edits:
            edit.operand.state[1] = PART_EDITED

        for part in shape.partset:
            if len(part.sub_parts) > 0:
                edited_children = [child.state[1] == PART_EDITED for child in part.sub_parts]
                if any(edited_children):
                    print(f"Part {part.label} has edited children. Turned Off")
                    shape.deactivate_parent(part)

        shape.clean_up_motion()
        for edit in all_edits:
            edit.propagate()
    
        # Check the relation relations and revert if required.
        if relation_relations:
            for relation in relation_relations:
                if relation.broken():
                    new_edit = relation.resolve()
                    all_edits.extend(new_edit)
                    continue

        for ind, relation in enumerate(shape.all_relations(only_active=True)):
            if relation.broken():
                relation.state[1] = RELATION_BROKEN
            else:
                relation.state[1] = RELATION_STABLE

        potential_relations = [x for x in shape.all_relations(only_active=True) if x.state[1] == RELATION_BROKEN]
        potential_relations = [x for x in potential_relations if x.state[2] != RELATION_REJECTED] # do not consider rejected relations
        unfixable_relations = [x for x in potential_relations if not x.resolvable() and not x.updatable()] # do not consider unfixable relations
        for relation in unfixable_relations:
            print("manually rejecting this one", relation)
            relation.state[2] = RELATION_REJECTED
            relation.state[3] = NOTHING_TO_DO
        broken_relations = [x for x in potential_relations if x.resolvable() or x.updatable()] # do not consider unfixable relations

        # Step 2: Find fixes
        if len(broken_relations) == 0:
            # Step END: see if edit is complete.
            algorithm_end, new_edit, cur_info_list = prompter.api_call_finish_algo(shape, edit_request, all_edits)
            log_info.extend(cur_info_list)
            if not isinstance(new_edit, list):
                new_edit = [new_edit]
            if not algorithm_end:
                all_edits.extend(new_edit)
        else:        
            relations_to_search_for, new_edits = infer_relation_state(shape, edit_request, prompter, all_edits, log_info, broken_relations)

            if len(new_edits) > 0:
                all_edits.extend(new_edits)
                continue
            
            if len(relations_to_search_for) > 0:
                edit_proposer.register_broken_relations(relations_to_search_for)
                part_to_edit = edit_proposer.propose_part_to_edit(shape)
                assert valid_part(part_to_edit), "This should not happen"
                # For this part, we should ask about unchecked relations with already edited parts.
                select_failed = False
                if part_to_edit is None:
                    select_failed = True
                if isinstance(part_to_edit, Part):
                    # Check of extra relations
                    parts_unchecked_relations = [x for x in part_to_edit.all_relations(only_active=True) if x.state[2] == RELATION_UNCHECKED]
                    parts_relation_to_verify = [x for x in parts_unchecked_relations if isinstance(x, FeatureRelation) and x.other_part_edited(part_to_edit)]
                    if len(parts_relation_to_verify) > 0:
                        print("Found relations which may be thrown away without LLM's knowledge")
                        _, new_edits = infer_relation_state(shape, edit_request, prompter, 
                                                                            all_edits, log_info, parts_relation_to_verify)
                        if len(new_edits) > 0:
                            all_edits.extend(new_edits)
                        continue
                if not select_failed:
                    simple_edit, cur_info_list, using_breaking = get_edit(part_to_edit, shape, edit_request, 
                                                             all_edits, prompter, edit_proposer, init_edit_types=init_edit_types)
                    if using_breaking:
                        any_breaking = True
                    log_info.extend(cur_info_list)
                    if simple_edit is None:
                        select_failed = True
                        selected_edits = []
                    else:
                        selected_edits = [simple_edit]
                all_edits.extend(selected_edits)

                # TODO
                if select_failed:
                    relations_to_reject = relations_to_search_for
                    for relation in relations_to_reject:
                        relation.state[2] = RELATION_REJECTED
                        relation.state[3] = NOTHING_TO_DO
                edit_proposer.refresh()
    
    # Finally summarize the broken relations.
    final_state = {}
    for relation in shape.all_relations(only_active=True):
        if relation.state[2] == RELATION_REJECTED:
            final_state[f"r_{relation.relation_index}"] = 0
        else:
            if relation.state[1] == RELATION_BROKEN:
                final_state[f"r_{relation.relation_index}"] = 0
            else:
                final_state[f"r_{relation.relation_index}"] = 1
    for part in shape.partset:
        if part.state[0] == PART_ACTIVE:
            final_state[f'p_{part.part_index}'] = 1
        else:
            final_state[f'p_{part.part_index}'] = 0
                
    return all_edits, log_info, any_breaking, final_state


def get_edit_type(part_to_edit, shape, edit_request, 
                all_edits, prompter:LLMPrompterV2, edit_proposer:EditProposer, max_retries=MAX_EDIT_SPEC_TRY, init_edit_types=None):

    # Predict the entire function solve for it.
    log_info_seq = []
    initial_edit_spec, log_info = prompter.api_call_generate_edit_type(part_to_edit, shape, edit_request, all_edits)
    # log_info_seq.extend(log_info)
    solved_edit, least_breaking = edit_proposer.solve_edit_by_type(part_to_edit, initial_edit_spec, shape)
    using_breaking = False
    if not solved_edit is None:
        edit_successful = True
    elif not least_breaking is None:
        solved_edit = least_breaking[0]
        edit_successful = True
        using_breaking = True
    else:
        solved_edit = KeepFixed(part_to_edit)
    return solved_edit, log_info_seq, using_breaking

def get_edit_type_reversible(part_to_edit, shape, edit_request, 
                all_edits, prompter:LLMPrompterV2, edit_proposer:EditProposer, max_retries=MAX_EDIT_SPEC_TRY, init_edit_types=None):

    # Predict the entire function solve for it.
    log_info_seq = []
    initial_edit_spec, log_info = prompter.api_call_generate_edit_type(part_to_edit, shape, edit_request, all_edits)
    log_info_seq.extend(log_info)
    edit_successful = False
    edit_spec = initial_edit_spec
    count = 0
    failed_specs = []
    breaking_options = []
    while (not edit_successful):
        count  += 1
        solved_edit, least_breaking = edit_proposer.solve_edit_by_type(part_to_edit, edit_spec, shape)
        if not solved_edit is None:
            edit_successful = True
            break
        elif not least_breaking is None:
            all_edits = ['translate', 'rotate', 'scale', 'shear']
            last_ind = all_edits.index(edit_spec)
            if not last_ind == len(all_edits) - 1:
                failed_specs.append(edit_spec)
                breaking_options.append(least_breaking)
                action, log_info = prompter.api_call_get_edit_action(part_to_edit, shape, edit_request, 
                                                                        all_edits, breaking_options)
            else:
                action = "accept"
                log_info = []
            log_info_seq.extend(log_info)
            if action == "accept":
                solved_edit = least_breaking[0]
                edit_successful = True
            elif action == "search_more_complex":
                edit_spec, log_info = prompter.api_call_generate_edit_type(part_to_edit, shape, edit_request, all_edits, last_edit=failed_specs)
                log_info_seq.extend(log_info)
        else:
            break
        if count > max_retries:
            break
    
        
    return solved_edit, log_info_seq


def get_edit_full_spec(part_to_edit, shape, edit_request, 
                all_edits, prompter:LLMPrompterV2, edit_proposer: EditProposer, max_retries=MAX_EDIT_SPEC_TRY, init_edit_types=None):

    # Predict the entire function solve for it.
    log_info_seq = []
    initial_edit_spec, log_info = prompter.api_call_generate_edit_spec(part_to_edit, shape, edit_request, all_edits)
    log_info_seq.extend(log_info)
    edit_successful = False
    edit_spec = initial_edit_spec
    count = 0
    failed_specs = []
    breaking_options = []
    while (not edit_successful):
        count  += 1
        solved_edit, least_breaking = edit_proposer.solve_edit(part_to_edit, edit_spec)
        if not solved_edit is None:
            edit_successful = True
            break
        elif not least_breaking is None:
            failed_specs.append(edit_spec)
            breaking_options.append(least_breaking)
            action, log_info = prompter.api_call_get_edit_action(part_to_edit, shape, edit_request, 
                                                                      all_edits, breaking_options)
            log_info_seq.extend(log_info)
            if action == "accept":
                solved_edit = least_breaking[0]
                edit_successful = True
            elif action == "search_more_complex":
                edit_spec, log_info = prompter.api_call_generate_edit_spec(part_to_edit, shape, edit_request, all_edits, last_edit=failed_specs)
                log_info_seq.extend(log_info)
        else:
            break
        if count > max_retries:
            break
    
        
    return solved_edit, log_info_seq

def get_edit_geom(part_to_edit, shape, edit_request, 
                all_edits, prompter:LLMPrompterV2, edit_proposer:EditProposer, max_retries=MAX_EDIT_SPEC_TRY, init_edit_types=None):

    # Predict the entire function solve for it.
    log_info_seq = []
    initial_edit_type = init_edit_types.get(part_to_edit.full_label, None)
    # log_info_seq.extend(log_info)
    edit_spec = initial_edit_type
    solved_edit, least_breaking = edit_proposer.solve_edit_by_type(part_to_edit, edit_spec, shape)
    using_breaking = False
    if not solved_edit is None:
        edit_successful = True
    elif not least_breaking is None:
        print("IS breaking")
        solved_edit = least_breaking[0]
        edit_successful = True
        using_breaking = True
    else:
        solved_edit = KeepFixed(part_to_edit)
    print("Solved edit is ", solved_edit)
    return solved_edit, log_info_seq, using_breaking

def valid_part(part_to_edit):
    valid = True
    if isinstance(part_to_edit, Part):
        n_edits = len(part_to_edit.primitive.edit_sequence)
    elif isinstance(part_to_edit, PrimitiveRelation):
        n_edits = len(part_to_edit.edit_sequence)
    else:
        n_edits = 0
    if n_edits > 0:
        valid = False
    return valid
    
EDIT_MAPPER = {
    "GEOM": get_edit_geom,
    "V1": get_edit_type,
    "V2": get_edit_type_reversible,
    "V3": get_edit_full_spec,
}

def parallel_call_func(prompter_class, prompter_info, shape, edit_request,  mode):
    
    prompter = prompter_class(**prompter_info)
    if mode == 0:
        messages = prompter.create_keep_fixed_prompt(shape, edit_request)
        fixed_parts, response = prompter._get_fixed_parts(shape, messages)
    
        keep_fixed_edits = [KeepFixed(operand=shape.get(part)) for part in fixed_parts]
        keep_fixed_info = prompter.get_keep_fixed_info(shape, keep_fixed_edits, messages, response)
        output = keep_fixed_edits, keep_fixed_info
    elif mode == 1:
        
        messages = prompter.create_initialize_prompt(shape, edit_request)
        primary_edits, response = prompter._get_primary_edits(shape, messages)
        
        if not isinstance(primary_edits, list):
            update_amount(primary_edits)
            primary_edits = [primary_edits]
        primary_edit_info = prompter.get_primary_edit_info(primary_edits, messages, response)
        output = primary_edits, primary_edit_info
    elif mode == 2:
        
        
        messages = prompter.create_init_edit_hint_prompt(shape, edit_request)
        n_tries = 0

        edit_type_hints, response = prompter._get_init_edit_hints(shape, messages)
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
            full_types[shape.get(name)] = HIGHER_TYPE_HINTS[edit_type]
        output = full_types, cur_info
    return output
        