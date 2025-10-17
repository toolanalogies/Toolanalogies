

from edit_sys.shape_system.constants import (RELATION_UNCHECKED, RELATION_BROKEN, RELATION_REJECTED, 
                                             RELATION_ACTIVE, RELATION_INACTIVE,
                                             RELATION_RETAINED, RELATION_STABLE, RESOLVE_RELATION, 
                                             NOTHING_TO_DO, UPDATE_RELATION)

from edit_sys.shape_system.shape_atoms import PART_EDITED, PART_UNEDITED, PART_ACTIVE, PART_INACTIVE
from edit_sys.shape_system import Part, Primitive, PrimitiveFeature, PrimitiveRelation, FeatureRelation, ReflectionSymmetry
from edit_sys.shape_system.proposal_mechanism import EditProposer
from edit_sys.llm.base_prompter import LLMPrompterV1
from edit_sys.shape_system.edits import KeepFixed

TWO_STEP_CONTACT_CHECK = True

def algo_v3(shape: Part, edit_request: str, prompter: LLMPrompterV1, edit_proposer: EditProposer,
            get_types=False):
     
    # Step 1, add a ground part and a height constraint to it.
    # This should be done with LLM
    ground_edit = KeepFixed(shape.get('ground'))
    all_edits = [ground_edit]
    # all_edits = []
    log_info = []
    # Check sym relations before hand - remove if required.
    shape, cur_info_list = prompter.api_call_initial_update(shape, edit_request)
    log_info.extend(cur_info_list)
    # log_info
    new_edits, cur_info_list = prompter.api_call_initialize(shape, edit_request)
    # all_edits = []
    log_info.extend(cur_info_list)
    all_edits.extend(new_edits)
    print("initial edits are as follows.", all_edits)
    # Add initial relation type hints
    type_hints = {}
    relation_hints = {}
    if get_types:
        sym_relations = [x for x in shape.all_relations(only_active=True) if isinstance(x, PrimitiveRelation)]
        for relation in sym_relations:
            if isinstance(relation, ReflectionSymmetry):
                valid_states = [0, 1]
            else:
                valid_states = [0, 2]
            relation_hint, cur_info_list = prompter.api_call_get_relation_hint(shape, edit_request, all_edits, 
                                                                                relation, valid_states)
            type_hints[relation] = relation_hint
            log_info.extend(cur_info_list)

        # Add type hints
        edit_type_hints, cur_info_list = prompter.api_call_get_edit_hints(shape, edit_request, all_edits)
        # dictionary for type to edit.)
    else:
        type_hints = {}
        edit_type_hints = {}

    log_info.extend(cur_info_list)

    algorithm_end = False
    while not algorithm_end:

        for edit in all_edits:
            edit.operand.state[1] = PART_EDITED
        # For each part, check if it has children, and if the children are edited.
        # If yes then mark activate/deactivate relations accordingly
        for part in shape.partset:
            if len(part.sub_parts) > 0:
                edited_children = [child.state[1] == PART_EDITED for child in part.sub_parts]
                if any(edited_children):
                    print(f"Part {part.label} has edited children.")
                    print("It is turned Off")
                    # update the shape (or just activate deactivate relations accordingly)
                    shape.deactivate_parent(part)

        shape.clean_up_motion()
        # propagate edits:
        for edit in all_edits:
            edit.propagate()

        for ind, relation in enumerate(shape.all_relations(only_active=True)):
            if relation.broken():
                relation.state[1] = RELATION_BROKEN
            else:
                relation.state[1] = RELATION_STABLE
                # relation.state[3] = NOTHING_TO_DO

        potential_relations = [x for x in shape.all_relations(only_active=True) if x.state[1] == RELATION_BROKEN]
        potential_relations = [x for x in potential_relations if x.state[2] != RELATION_REJECTED] # do not consider rejected relations
        unfixable_relations = [x for x in potential_relations if not x.resolvable() and not x.updatable()] # do not consider unfixable relations
        for relation in unfixable_relations:
            print("manually rejecting this one", relation)
            relation.state[2] = RELATION_REJECTED
            relation.state[3] = NOTHING_TO_DO
        broken_relations = [x for x in potential_relations if x.resolvable() or x.updatable()] # do not consider unfixable relations

        # check and offer solutions if required.
        if len(broken_relations) == 0:
            algorithm_end, new_edit, cur_info_list = prompter.api_call_finish_algo(shape, edit_request, all_edits)
            log_info.extend(cur_info_list)
            if not isinstance(new_edit, list):
                new_edit = [new_edit]
            if not algorithm_end:
                all_edits.extend(new_edit)
        else:

            relations_to_search_for, new_edits = infer_relation_state(shape, edit_request, prompter, all_edits, log_info, broken_relations, relation_hints)

            if len(new_edits) > 0:
                all_edits.extend(new_edits)
                continue
            
            if len(relations_to_search_for) > 0:
                # what shapes or relations can be udated?
                # We update based on realtions to search for
                # but return a part which might resolve only a few of the relations.
                edit_proposer.update_edit_candidates(shape, relations_to_search_for, all_edits, edit_type_hints)
                part_to_edit, edit_options = edit_proposer.get_options()
                # For this part, we should ask about unchecked relations with already edited parts.
                if not part_to_edit is None and isinstance(part_to_edit, Part):
                    parts_unchecked_relations = [x for x in part_to_edit.all_relations(only_active=True) if x.state[2] == RELATION_UNCHECKED]
                    parts_relation_to_verify = [x for x in parts_unchecked_relations if isinstance(x, FeatureRelation) and x.other_part_edited(part_to_edit)]
                    if TWO_STEP_CONTACT_CHECK and len(parts_relation_to_verify) > 0:
                        # do sometime
                        edit_proposer.add_back(part_to_edit, edit_options)
                        print("Found relations which may be thrown away without LLM's knowledge")
                        print("Could verify these relations...")

                        dummy_rel_to_search, new_edits = infer_relation_state(shape, edit_request, prompter, 
                                                                                        all_edits, log_info, parts_relation_to_verify,
                                                                                        type_hints)
                        
                        if len(new_edits) > 0:
                            all_edits.extend(new_edits)
                        continue

                select_failed = False
                if len(edit_options) >= 1:
                    # need to select
                    if isinstance(part_to_edit, Part):
                        n_edits = len(part_to_edit.primitive.edit_sequence)
                    elif isinstance(part_to_edit, PrimitiveRelation):
                        n_edits = len(part_to_edit.edit_sequence)
                    else:
                        n_edits = 0
                    if n_edits > 0:
                        print("WHY")
                        raise ValueError("This should not happen")

                    selected_edits, cur_info_list = prompter.api_call_select_edit(shape, edit_request, all_edits, edit_options, part_to_edit, edit_type_hints)
                    log_info.extend(cur_info_list)
                    if selected_edits is None:
                        select_failed = True
                    else:
                        selected_edits = [selected_edits]
                else:
                    select_failed = True
                if select_failed:
                    print("FAILED TO SOLVE")
                    # Now we must drop the broken relations (but what if its due to other )
                    selected_edits = []
                    if part_to_edit is None:
                        # get the part to edit from the broken relation
                        part_to_edit = part_to_edit_from_broken_relations(relations_to_search_for)
                    requested_type = prompter.api_call_desired_edit_type(shape, edit_request, all_edits, part_to_edit)
                    edit_proposer.update_breaking_edit_candidates(part_to_edit, requested_type)
                    part_to_edit, edit_options = edit_proposer.get_breaking_options()
                    selected_batch, cur_info_list = prompter.api_call_select_breaking_edit(shape, edit_request, all_edits, edit_options, part_to_edit)
                    if selected_edits is None:
                        select_failed = True
                        relations_to_reject = []
                    else:
                        select_failed = False
                        relations_to_reject = selected_batch[1]
                        selected_edits = [selected_edits[0]]

                    for relation in relations_to_reject:
                        relation.state[2] = RELATION_REJECTED
                        relation.state[3] = NOTHING_TO_DO
                    log_info.extend(cur_info_list)
                all_edits.extend(selected_edits)
    return all_edits, log_info
    # Keep fixed, and edit seed.
    
    # for the others gather type hints.
    # Type hints - What type of op and use minimal delta one?
    # propagate with system
    # When fails- soft and hard - remove relation prompt.
    # hard failure - no edits feasible.
    # On Failure - either select a different edit, or remove some relations.
    ...
def part_to_edit_from_broken_relations(relations_to_search_for):
    parts_to_edit = []
    for relation in relations_to_search_for:
        parts = relation.parts
        for part in parts:
            if len(part.primitive.edit_sequence) == 0:
                parts_to_edit.append(part)
    # What to do if multiple? 
    # HEURISTICS: select the one with least relations
    sorted_parts = sorted(parts_to_edit, key=lambda x: len(x.all_relations(only_active=True)))
    return sorted_parts[0]

def algo_v2(shape: Part, edit_request: str, prompter: LLMPrompterV1, edit_proposer: EditProposer):
    algorithm_end = False
    all_edits = []
    log_info = []

    while not algorithm_end:
        # Initialize
        if len(all_edits) == 0:
            # ADD decide object structure
            shape, cur_info_list = prompter.api_call_initial_update(shape, edit_request)
            log_info.extend(cur_info_list)
            # log_info
            all_edits, cur_info_list = prompter.api_call_initialize(shape, edit_request)
            # all_edits = []
            log_info.extend(cur_info_list)
            print("initial edits are as follows.", all_edits)
        
        for edit in all_edits:
            edit.operand.state[1] = PART_EDITED
        # For each part, check if it has children, and if the children are edited.
        # If yes then mark activate/deactivate relations accordingly
        for part in shape.partset:
            if len(part.sub_parts) > 0:
                edited_children = [child.state[1] == PART_EDITED for child in part.sub_parts]
                if any(edited_children):
                    print(f"Part {part.label} has edited children.")
                    print("It is turned Off")
                    # update the shape (or just activate deactivate relations accordingly)
                    shape.deactivate_parent(part)

        shape.clean_up_motion()
        # propagate edits:
        for edit in all_edits:
            edit.propagate()

        for ind, relation in enumerate(shape.all_relations(only_active=True)):
            if relation.broken():
                relation.state[1] = RELATION_BROKEN
            else:
                relation.state[1] = RELATION_STABLE
                # relation.state[3] = NOTHING_TO_DO

        potential_relations = [x for x in shape.all_relations(only_active=True) if x.state[1] == RELATION_BROKEN]
        potential_relations = [x for x in potential_relations if x.state[2] != RELATION_REJECTED] # do not consider rejected relations
        unfixable_relations = [x for x in potential_relations if not x.resolvable() and not x.updatable()] # do not consider unfixable relations
        for relation in unfixable_relations:
            print("manually rejecting this one", relation)
            relation.state[2] = RELATION_REJECTED
            relation.state[3] = NOTHING_TO_DO
        broken_relations = [x for x in potential_relations if x.resolvable() or x.updatable()] # do not consider unfixable relations

        # check and offer solutions if required.
        if len(broken_relations) == 0:
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
                # what shapes or relations can be udated?
                # We update based on realtions to search for
                # but return a part which might resolve only a few of the relations.
                edit_proposer.update_edit_candidates(shape, relations_to_search_for, all_edits)
                part_to_edit, edit_options = edit_proposer.get_options()
                # For this part, we should ask about unchecked relations with already edited parts.
                if not part_to_edit is None and isinstance(part_to_edit, Part):
                    parts_unchecked_relations = [x for x in part_to_edit.all_relations(only_active=True) if x.state[2] == RELATION_UNCHECKED]
                    parts_relation_to_verify = [x for x in parts_unchecked_relations if isinstance(x, FeatureRelation) and x.other_part_edited(part_to_edit)]
                    if TWO_STEP_CONTACT_CHECK and len(parts_relation_to_verify) > 0:
                        # do sometime
                        # add it back into the system
                        edit_proposer.add_back(part_to_edit, edit_options)
                        print("Found relations which may be thrown away without LLM's knowledge")
                        print("Could verify these relations...")
                        dummy_rel_to_search, new_edits = infer_relation_state(shape, edit_request, prompter, 
                                                                                        all_edits, log_info, parts_relation_to_verify)
                        
                        if len(new_edits) > 0:
                            all_edits.extend(new_edits)
                            continue
                        continue

                kill_relations = False
                if len(edit_options) > 1:
                    # need to select
                    if isinstance(part_to_edit, Part):
                        n_edits = len(part_to_edit.primitive.edit_sequence)
                    elif isinstance(part_to_edit, PrimitiveRelation):
                        n_edits = len(part_to_edit.edit_sequence)
                    if n_edits > 0:
                        print("WHY")
                        kill_relations = True
                    else:
                        selected_edits, cur_info_list = prompter.api_call_select_edit(shape, edit_request, all_edits, edit_options, part_to_edit)
                        log_info.extend(cur_info_list)
                        selected_edits = [selected_edits]
                elif len(edit_options) == 1:
                    if isinstance(part_to_edit, Part):
                        n_edits = len(part_to_edit.primitive.edit_sequence)
                    elif isinstance(part_to_edit, PrimitiveRelation):
                        n_edits = len(part_to_edit.edit_sequence)
                    if n_edits > 0:
                        print("WHY")
                        kill_relations = True
                    else:
                        selected_edits = edit_options[0].employ(part_to_edit)
                        selected_edits = [selected_edits]
                else:
                    # In this case, just reject the relation with smallest error.
                    # Search Again, but with less relations.
                    # select one of the relations to reject.
                    kill_relations = True

                if kill_relations:
                    print("FAILED TO SOLVE THE FOLLOWING RELATIONS")

                    for relations in relations_to_search_for:
                        print(f"Rejected relation {relations.signature()}")
                        relations.state[2] = RELATION_REJECTED
                        relations.state[3] = NOTHING_TO_DO
                    cur_info = {
                        'step': 'select_edit_failure',
                        'options': [],
                        'selected': None,
                        'failed_relations': [x.relation_index for x in relations_to_search_for]
                    }
                    selected_edits = []
                    log_info.append(cur_info)
                all_edits.extend(selected_edits)
    return all_edits, log_info

def infer_relation_state(shape, edit_request, prompter, all_edits, log_info, broken_relations, relation_hints=None):
    relations_to_resolve = []
    relations_to_update = []
    for relation in broken_relations:
                        # Keep this information for later.
        n_edits = [len(x.primitive.edit_sequence) for x in relation.parts]
        if min(n_edits) > 0:
            print("Manually removing relation", relation)
            relation.state[2] = RELATION_REJECTED
            relation.state[3] = NOTHING_TO_DO

        if relation.state[2] == RELATION_UNCHECKED:
            # TODO: Will this backfire?
            if isinstance(relation, FeatureRelation):
                valid_states = []
            else:
                valid_states = [NOTHING_TO_DO]
            if relation.resolvable():
                valid_states.append(RESOLVE_RELATION)
            if relation.updatable():
                valid_states.append(UPDATE_RELATION)
            if len(valid_states) == 0:
                relation_request = NOTHING_TO_DO
            elif len(valid_states) == 1:
                print("Only one valid state")
                print(f"setting relation {relation.signature()}'state to {valid_states[0]}")
                relation_request = valid_states[0]
            else:
                        # give more information if its a resolvable relation
                resolved_new_edits = None
                if RESOLVE_RELATION in valid_states:
                    if relation.automatically_resolvable():
                        resolved_new_edits = relation.resolve()

                relation_request, cur_info_list = prompter.api_call_check_relation(shape, edit_request, all_edits, 
                                                                                   relation, valid_states, resolved_new_edits)
                log_info.extend(cur_info_list)
                    
            if relation_request == NOTHING_TO_DO:
                relation.state[2] = RELATION_REJECTED
            elif relation_request == RESOLVE_RELATION:
                relation.state[2] = RELATION_RETAINED
                relation.state[3] = RESOLVE_RELATION
                relations_to_resolve.append(relation)
            elif relation_request == UPDATE_RELATION:
                relation.state[2] = RELATION_RETAINED
                relation.state[3] = UPDATE_RELATION
                relations_to_update.append(relation)
                    # Now we have relations to enforce.
        elif relation.state[2] == RELATION_RETAINED:
            # if both have edits - reject
            if relation.state[3] == UPDATE_RELATION and relation.updatable():
                relations_to_update.append(relation)
            elif relation.state[3] == RESOLVE_RELATION and relation.resolvable():
                relations_to_resolve.append(relation)
            else:
                relation.state[2] = RELATION_REJECTED

    relations_to_search_for = []
    new_edits = []
    for relation in relations_to_resolve:
        if relation.automatically_resolvable():
                    # This is technically a hack.
            new_edits.extend(relation.resolve())
            break
        else:
            relations_to_search_for.append(relation)
    if len(new_edits) > 0:
        relations_to_search_for = []
    else:
        for relation in relations_to_update:
            if relation.automatically_updatable():
                new_edits.extend(relation.update())
                break
            else:
                relations_to_search_for.append(relation)
    if len(new_edits) > 0:
        relations_to_search_for = []

    return relations_to_search_for, new_edits
    
def new_algorithm(shape, edit_request, prompter, edit_proposer):
    algorithm_end = False
    all_edits = []
    while not algorithm_end:
        if len(all_edits) == 0:
            all_edits = prompter.api_call_initialize(shape, edit_request)
            print("initial edits are as follows.", all_edits)
        for edit in all_edits:
            edit.operand.state[1] = PART_EDITED
            
        broken_relations = []
        # Next propagate edits.
        shape.clean_up_motion()
        # propagate edits:
        for edit in all_edits:
            edit.propagate()

        # identify broken
        for relation in shape.all_relations(only_active=True):
            if relation.broken():
                relation.state[1] = RELATION_BROKEN
            else:
                relation.state[1] = RELATION_STABLE
                
        broken_relations = [x for x in shape.all_relations(only_active=True) if x.state[1] == RELATION_BROKEN]
        broken_relations = [x for x in broken_relations if x.state[2] != RELATION_REJECTED]

        if len(broken_relations) > 0:
            print(f"The number of broken relations is {len(broken_relations)}")
            # We need to see what we can do.
            relations_to_resolve, new_edits = check_relations(shape, edit_request, prompter, all_edits, broken_relations)
            if len(new_edits) > 0:
                # Restart the process
                all_edits.extend(new_edits)
            else:
                if len(relations_to_resolve) == 0:
                    continue
                else:
                    # For each broken relationship to enforce find solutions.
                    part_set = set()

                    for relation in relations_to_resolve:
                        if isinstance(relation, PrimitiveRelation):
                            for primitive in relation.primitives:
                                # part_to_broken_relations[primitive.part].append(relation)
                                part_set.add(primitive.part)
                        elif isinstance(relation, FeatureRelation):
                            for feature in relation.features():
                                # part_to_broken_relations[feature.primitive.part].append(relation)
                                part_set.add(feature.primitive.part)
                    # Gather edits for all parts.
                    edit_proposer.update_edit_candidates(part_set, all_edits)
                    part, options = edit_proposer.get_options()
                    # should make all the part relations stable
                    print(f"Gathered {len(options)} options.")
                    if len(options) > 1:
                        # For the part's relations, see if they are unspecified.
                        # See if they have a edited part both ways. If yes, then see if they should be retained or not.
                        unspec_relations = [x for x in part.all_relations(only_active=True) if x.state[2] == RELATION_UNCHECKED]
                        relations_to_check = []
                        for relation in unspec_relations:
                            relevant_parts = [cur_part for cur_part in relation.operands if cur_part.label != part.label]
                            edited = [cur_part.state[1] == PART_EDITED for cur_part in relevant_parts]
                            if all(edited):
                                relations_to_check.append(relation)
                        if len(relations_to_check) > 0:
                            relations_to_resolve, new_edits = check_relations(shape, edit_request, prompter, all_edits, broken_relations)
                            if len(new_edits) > 0:
                                # Restart the process
                                all_edits.extend(new_edits)
                                continue
                            else:
                                for relation in relations_to_resolve:
                                    relation.state[1] = RELATION_BROKEN

                        selected_edits = prompter.api_call_select_edit(part, options, all_edits, shape, edit_request)
                    elif len(options) == 1:
                        selected_edits = options[0].employ(operand=part)
                    else:
                        # This is where the backtracking will happen -> Or note relation as not fixable, and move on.
                        raise ValueError("No options found.")
                    
                    for relation in part.all_relations(only_active=True):
                        if relation.state[1] == RELATION_BROKEN:
                            if relation.state[2] == RELATION_RETAINED:
                                relation.state[1] = RELATION_STABLE

                    all_edits.append(selected_edits)
        else:
            # Ask gpt if over. If not add the new one
            # Over anyways if all parts are edited
            unedited_parts = [part for part in shape.partset if part.state[1] == PART_UNEDITED]
            if len(unedited_parts) == 0:
                algorithm_end = True
            else:
                algorithm_end, new_edit = prompter.api_call_finish_algo(all_edits, shape, edit_request)
                if not algorithm_end:
                    all_edits.append(new_edit)
    return all_edits

def check_relations(shape, edit_request, prompter, all_edits, broken_relations):
    relations_to_enforce = []
    for relation in broken_relations:
                # Keep this information for later.
        if relation.state[2] == RELATION_UNCHECKED:
            part_validity = [part.state[1] == PART_UNEDITED for part in relation.operands]
            if any(part_validity):
                retain_relation = prompter.api_call_check_relation(relation, all_edits, shape, edit_request)
            else:
                retain_relation = False
                    # relation.summary = summary
            if retain_relation:
                relation.state[2] = RELATION_RETAINED
            else:
                relation.state[2] = RELATION_REJECTED
        if relation.state[2] == RELATION_RETAINED:
           relations_to_enforce.append(relation)
            # Now we have relations to enforce.
    relations_to_resolve = []
    new_edits = []
    for relation in relations_to_enforce:
        if relation.automatically_resolvable():
                    # This is technically a hack.
            new_edits.extend(relation.resolve())
            relation.state[1] = RELATION_STABLE
            break
        else:
            relations_to_resolve.append(relation)
    return relations_to_resolve, new_edits
        