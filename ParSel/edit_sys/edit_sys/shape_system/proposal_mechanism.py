import itertools
import time
# from .edits import *
from .edits import *
from .geometric_atoms import *
import sympy as sp
import open3d as o3d
import copy
from collections import defaultdict
from .constants import (RELATION_UNCHECKED, RELATION_REJECTED, RELATION_RETAINED,
                        RELATION_BROKEN, RESOLVE_RELATION, UPDATE_RELATION,
                        MOVE_LIMIT)
from .shape_atoms import PART_EDITED, PART_UNEDITED
from .relations import (Relation, FeatureRelation, PrimitiveRelation, 
                        ReflectionSymmetry, TranslationSymmetry, RotationSymmetry, 
                        PointContact, LineContact, FaceContact)
from .utils import evaluate_equals_zero, QUANT_COMPARE_THRESHOLD_MODE_2
from .proposer_utils import (valid_solution, clean_up_motion, 
                             remove_null_edits, merge_finalized_edits, 
                             remove_flips, remove_high_distortion, sort_by_typed_distortion_energy, 
                             get_directions, select_new_edits,
                             COEFF_LIMIT, OPPOSITE_NAME)
MIN_EDIT_CANDIDATES = 1 # 3
RELATION_VALUE = {
    PointContact: 1,
    LineContact: 2,
    FaceContact: 3,
}
class EditProposer:
    def __init__(self, heuristic):
        self.operand_to_edit_candidate = defaultdict(list)
        self.operand_to_tried_edits = defaultdict(list)
        self.broken_relations = []
        self.heuristic  = heuristic
    
    def register_broken_relations(self, relations):
        self.broken_relations.extend(relations)
    def refresh(self):
        self.operand_to_edit_candidate = defaultdict(list)
        self.operand_to_tried_edits = defaultdict(list)
        self.broken_relations = []
    
    def propose_part_to_edit(self, shape):
        # based on the broken relations, 
        # select the part with the most broken relations.
        parts_to_broken = defaultdict(list)
        for relation in self.broken_relations:
            if isinstance(relation, FeatureRelation):
                parts = [x.primitive.part for x in relation.features]
            elif isinstance(relation, PrimitiveRelation):
                parts = [x.part for x in relation.primitives]
            # parts = list(set(parts))
            for part in parts:
                if len(part.primitive.edit_sequence) == 0:
                    parts_to_broken[part].append(relation)

        # TODO Check this.
        for relation in shape.all_relations(only_active=True):
            if relation.state[2] == RELATION_RETAINED:
                if relation.state[3] == UPDATE_RELATION:
                    if isinstance(relation, (TranslationSymmetry, RotationSymmetry)):
                        if len(relation.edit_sequence) == 0:
                            parts_to_broken[relation].append(relation.parent_part)
        max_broken = 0
        selected_part = None
        sorted_keys = sorted(parts_to_broken.keys(), key=lambda x: x.full_label, reverse=True)        
        for part in sorted_keys:
            if isinstance(part, Relation):
                if len(part.edit_sequence) > 0:
                    continue
            else:
                if len(part.primitive.edit_sequence) > 0:
                    continue
            broken_relations = parts_to_broken[part]
            if isinstance(part, Relation):
                max_broken = len(broken_relations)
                selected_part = part
                break
            if len(broken_relations) > max_broken:
                max_broken = len(broken_relations)
                selected_part = part
        return selected_part

    def solve_edit_by_type(self, operand, edit_type, shape, try_extensions=True):
        if isinstance(operand, Relation):
            retained_relations = None
        else:
            retained_relations = self.get_relevant_relations(operand)
            retained_relations = [x for x in retained_relations if isinstance(x, FeatureRelation)]
        if edit_type is None:
            edit_types = HIGHER_TYPE_HINTS[edit_type]
        elif isinstance(edit_type, str):
            edit_types = HIGHER_TYPE_HINTS[edit_type]
        else:
            edit_types = edit_type

        selected_candidates = self.gather_options(operand, retained_relations, None, shape, edit_types, reject_null=False)

        pre_geom_solutions = len(selected_candidates)
        if len(selected_candidates) > 0:
            solved_edit = selected_candidates[0].employ(operand)
            least_breaking = None
        else:
            # Need to solve and record.
            if pre_geom_solutions > 0:
                # THis needs to be handled - its not constraints that are a problem but the fact that the edits are invalid.
                # Ideally, the LLM should try a different definition.
                ...
            print("NO SOLUTION! - detecting breaking solutions")
            # TODO: are just the basic edits enough?
            # Maybe also derive from siblings?
            sibling_edits_to_try = []
            for relation in retained_relations:
                if isinstance(relation, FeatureRelation):
                    parts = [x.primitive.part for x in relation.features]
                elif isinstance(relation, PrimitiveRelation):
                    parts = [x.part for x in relation.primitives]
                other_parts = [x for x in parts if x != operand]
                for other_part in other_parts:
                    sibling_edits = other_part.primitive.edit_sequence
                    sibling_edits_to_try.extend(sibling_edits)
            sibling_edit_gens = []
            for edit in sibling_edits_to_try:
                edit_gen = EditGen(edit.__class__, edit.params)
                sibling_edit_gens.append(edit_gen)
                if isinstance(edit, RestrictedTranslate):
                    opp_dir = OPPOSITE_NAME[edit.params['restrictor_name']]
                    params = {x:y for x, y in edit.params.items() if x != "restrictor_name"}
                    params['origin'] = edit.operand.face_center(opp_dir)
                    edit_gen = EditGen(PartScale1D, params)
                    sibling_edit_gens.append(edit_gen)
            edits_to_try = get_edits_to_try(operand)
            edits_to_try.extend(sibling_edit_gens)
            edits_to_try = type_based_prune(operand, edits_to_try, edit_types)

            if try_extensions:
                new_edits_to_try = relation_based_edit_candidates(operand, shape, edits_to_try)
                edits_to_try.extend(new_edits_to_try)

                sibling_edits_to_try = []
                for relation in retained_relations:
                    if isinstance(relation, FeatureRelation):
                        parts = [x.primitive.part for x in relation.features]
                    elif isinstance(relation, PrimitiveRelation):
                        parts = [x.part for x in relation.primitives]
                    other_parts = [x for x in parts if x != operand]
                    for other_part in other_parts:
                        sibling_edits = other_part.primitive.edit_sequence
                        sibling_edits_to_try.extend(sibling_edits)
                        
                for edit_ind, edit in enumerate(sibling_edits_to_try):
                    new_edits_to_try = get_additional_edit_candidates(operand, edit, edits_to_try)
                    edits_to_try.extend(new_edits_to_try)

                edits_to_try = type_based_prune(operand, edits_to_try, edit_types)

            solutions = solve_edits_breaking(operand, retained_relations, edits_to_try)
            solutions = [x for x in solutions if MAIN_VAR in x[0].amount.free_symbols]
            
            broken_relations = [x[1] for x in solutions]
            broken_equations = [x[2] for x in solutions]
            min_cost = np.min([len(x) for x in broken_relations])
            # Just select least breaking?
            
            selected_candidates = [x[0] for x in solutions if len(x[1]) == min_cost]
            broken_relations = [x[1] for x in solutions if len(x[1]) == min_cost]
            # join the lists in broken_relations
            # selected_candidates, broken_relations = remove_null_edits(selected_candidates, broken_relations)
            selected_candidates, broken_relations = merge_finalized_edits(operand, selected_candidates, broken_relations)
            pre_geom_solutions = len(selected_candidates)
            # selected_candidates, broken_relations = remove_flips(operand, selected_candidates, broken_relations)
            pruned_selected_candidates, pruned_broken_relations = remove_high_distortion(operand, selected_candidates, broken_relations, return_sorted=True)
            if len(pruned_selected_candidates) > 0:
                selected_candidates = pruned_selected_candidates
                broken_relations = pruned_broken_relations

            selected_candidates, broken_relations = sort_by_typed_distortion_energy(operand, selected_candidates, broken_relations, heuristic=self.heuristic)

            # all_broken_relations = list(set(retained_relations))
            # final_list = []
            # for ind, candidate in enumerate(selected_candidates):
            #     cur_broken_rel = broken_relations[ind]
            #     kept_relations = [x for x in all_broken_relations if x not in cur_broken_rel]
            #     cur_value = np.max([RELATION_VALUE[x.__class__] for x in kept_relations])
            #     final_list.append((candidate, cur_broken_rel, cur_value))
            # final_list = sorted(final_list, reverse=True, key=lambda x: x[2])
            # selected_candidates = [x[0] for x in final_list]
            # broken_relations = [x[1] for x in final_list]

            if len(selected_candidates) > 0:
                solved_edit = None
                least_breaking = (selected_candidates[0].employ(operand), broken_relations[0])
            else:
                print("NO SOLUTION! - detecting breaking solutions")
                # if len(selected_candidates)  > 0:
                solved_edit = None
                least_breaking = None
        return solved_edit, least_breaking
    
    def update_edit_candidates(self, shape, relations_to_resolve, previous_edits, edit_type_hints=None):

        start_time = time.time()
        operand_to_relation = defaultdict(list)
        parts_to_consider = []
        for relation in relations_to_resolve:
            if isinstance(relation, PrimitiveRelation):
                pass  # These are never fixed at a primitive level?
                # They are either done by TD definition, or by automatic resolves
            elif isinstance(relation, FeatureRelation):
                # for contact relations -
                parts = [x.primitive.part for x in relation.features]
                n_edits = [len(x.primitive.edit_sequence) for x in parts]
                # min_edits = np.min(n_edits)
                # FOrce single edits
                sel_parts = [x for x, y in zip(parts, n_edits) if y == 0]
                parts_to_consider.extend(sel_parts)

        for part in parts_to_consider:
            # relations which matter -> Active, Retained.
            retained_relations = self.get_relevant_relations(part)

            has_primitive_relations = False
            for relation in retained_relations:
                if isinstance(relation, (TranslationSymmetry, RotationSymmetry)):
                    has_primitive_relations = True
                    break
            # Why? Because for parts which are in a sym. group, we need multiple edits
            # TODO: See if this must be removed.
            if not has_primitive_relations:
                retained_relations = [x for x in retained_relations if isinstance(x, FeatureRelation)]
                # TODO: This is a hack.
                if len(retained_relations) > 0:
                    operand_to_relation[part] = retained_relations

        # Also the relations to be updated
        for relation in shape.all_relations(only_active=True):
            if relation.state[2] == RELATION_RETAINED:
                if relation.state[3] == UPDATE_RELATION:
                    # This should be only for shapes where the parent is active -
                    if isinstance(relation, (TranslationSymmetry, RotationSymmetry)):
                        n_edits = len(relation.edit_sequence)
                        if n_edits == 0:
                            operand_to_relation[relation].append(
                                relation.parent_part)

        for operand, relations_to_keep in operand_to_relation.items():
            if operand in edit_type_hints:
                cur_type_hints = edit_type_hints[operand]
            else:
                cur_type_hints = None
            if operand in self.operand_to_edit_candidate:
                edit_candidates, n_old_rel = self.operand_to_edit_candidate[operand]
                # # TODO: Add new options from new edits
                if len(relations_to_keep) == n_old_rel:
                    pass
                else:
                    edit_candidates = self.prune_options(operand, relations_to_keep, edit_candidates)
                    # Add from the new ones:
                    if len(edit_candidates) < MIN_EDIT_CANDIDATES:
                        print(f"There are only {len(edit_candidates)} options for {operand}")
                        new_edit_candidates = self.gather_new_options(
                            operand, relations_to_keep, previous_edits, edit_candidates, shape, cur_type_hints)
                        edit_candidates.extend(new_edit_candidates)
            else:
                edit_candidates = self.gather_options(operand, relations_to_keep, previous_edits, shape, cur_type_hints)
            self.operand_to_edit_candidate[operand] = (
                edit_candidates, len(relations_to_keep))
        end_time = time.time()

        print("Time for gathering options", end_time - start_time)
    
    def add_back(self, part_to_edit, edit_candidates):
        retained_relations = self.get_relevant_relations(part_to_edit)

        has_primitive_relations = False
        for relation in retained_relations:
            if isinstance(relation, (TranslationSymmetry, RotationSymmetry)):
                has_primitive_relations = True
                break
        # Why? Because for parts which are in a sym. group, we need multiple edits
        # TODO: See if this must be removed.
        if not has_primitive_relations:
            retained_relations = [
                x for x in retained_relations if isinstance(x, FeatureRelation)]
            # TODO: This is a hack.
            if len(retained_relations) > 0:
                retained_relations
            else:
                raise ValueError("This should not happen")
                retained_relations = []

        self.operand_to_edit_candidate[part_to_edit] = (
            edit_candidates, len(retained_relations))

    def get_relevant_relations(self, part):
        relations = part.all_relations(only_active=True)
        retained_relations = [
            x for x in relations if x.state[2] == RELATION_RETAINED]
        # now from these only the ones that have to be resolved - not the ones that must be updated
        retained_relations = [
            x for x in retained_relations if x.state[3] != UPDATE_RELATION]
        # Also only if the other part is edited.
        before_len = len(retained_relations)
        retained_relations = [
            x for x in retained_relations if x.other_part_edited(part)]
        print("length before and after other_part_edit based reject",
              before_len, len(retained_relations))
        return retained_relations

    def get_options(self, minimum_entropy=True):

        # from convert from part to relation
        # How many ways to stabilize a relation?
        # pick the one with fewest options.
        entropy = dict()
        for part, (edit_candidates, n_relations) in self.operand_to_edit_candidate.items():
            n_options = len(edit_candidates)
            if n_options > 0:
                if isinstance(part, Relation):
                    entropy[part] = (n_options, 0)
                else:
                    if len(part.primitive.edit_sequence) == 0:
                        entropy[part] = (n_options, len(
                            part.primitive.edit_sequence))
        n_edits = [x[1] for x in entropy.values()]

        if len(n_edits) == 0:
            return None, []
        else:
            # Now if a part has a potential sym. use the smaller one first.
            min_edits = np.min(n_edits)
            sorted_parts = [x for x, y in entropy.items() if y[1] == min_edits]
            sorted_parts = sorted(sorted_parts, key=lambda x: entropy[x][0])

            part = sorted_parts[0]
            edit_candidates, n_relations = self.operand_to_edit_candidate[part]
            self.operand_to_edit_candidate.pop(part)
            return part, edit_candidates

    def prune_options(self, target_part, retained_relations, edit_candidates):
        # here we will add the n-edit loop
        if isinstance(target_part, Relation):
            all_edit_candidates = edit_candidates
        else:
            edits_to_try = edit_candidates
            all_edit_candidates = try_edits(
                target_part, retained_relations, edits_to_try)
        return all_edit_candidates

    def gather_options(self, operand, retained_relations, previous_edits, shape, type_hints=None, reject_null=False):
        if isinstance(operand, Relation):
            # If translation/rotation return count/delta edit.
            # If contact -> add an empty edit?
            # If reflection -> add an empty edit?
            selected_candidates = operand.gather_update_options()
            # How should it be - Only valid types allowed?
            # or Solve all, but show only valid_types?

        else:

            # here we will add the n-edit loop
            print("gathering options for ", operand)
            solve_edits = solve_edits_v0
            n_edits = len(operand.primitive.edit_sequence)
            self.operand_to_tried_edits[operand].append(f"p_{operand.part_index}")
            if n_edits == 0:
                # First generate based on edits on others - just direct application
                print('Trying the sibling edits themselves')
                sibling_edits_to_try = []
                for relation in retained_relations:
                    if isinstance(relation, FeatureRelation):
                        parts = [x.primitive.part for x in relation.features]
                    elif isinstance(relation, PrimitiveRelation):
                        parts = [x.part for x in relation.primitives]
                    other_parts = [x for x in parts if x != operand]
                    for other_part in other_parts:
                        sibling_edits = other_part.primitive.edit_sequence
                        sibling_edits_to_try.extend(sibling_edits)
                sibling_edit_gens = []
                for edit in sibling_edits_to_try:
                    edit_gen = EditGen(edit.__class__, edit.params)
                    sibling_edit_gens.append(edit_gen)
                edits_to_try = get_edits_to_try(operand)
                edits_to_try.extend(sibling_edit_gens)
                edits_to_try = type_based_prune(operand, edits_to_try, type_hints)

                all_edit_candidates = solve_edits(operand, retained_relations, edits_to_try)
                if len(all_edit_candidates) < MIN_EDIT_CANDIDATES:
                    print('Trying edits from relations')
                    n_relations = len(retained_relations)
                    # Generate edits from existing validated relations
                    new_edits_to_try = relation_based_edit_candidates(operand, shape, edits_to_try)
                    new_edits_to_try = type_based_prune(operand, new_edits_to_try, type_hints)
                    for relation in retained_relations:
                        self.operand_to_tried_edits[operand].append(f"r_{relation.relation_index}")

                    new_edit_candidates = solve_edits(operand, retained_relations, new_edits_to_try)
                    all_edit_candidates.extend(new_edit_candidates)
                    if len(all_edit_candidates) < MIN_EDIT_CANDIDATES:
                        # try objects connected to it
                        print('Trying edits from sibling edits')
                        sibling_edits_to_try = []
                        for relation in retained_relations:
                            if isinstance(relation, FeatureRelation):
                                parts = [x.primitive.part for x in relation.features]
                            elif isinstance(relation, PrimitiveRelation):
                                parts = [x.part for x in relation.primitives]
                            other_parts = [x for x in parts if x != operand]
                            for other_part in other_parts:
                                sibling_edits = other_part.primitive.edit_sequence
                                sibling_edits_to_try.extend(sibling_edits)
                        for edit_ind, edit in enumerate(sibling_edits_to_try):

                            new_edits_to_try = get_additional_edit_candidates(operand, edit, edits_to_try)
                            new_edits_to_try = type_based_prune(operand, new_edits_to_try, type_hints)

                            new_edit_candidates = solve_edits(operand, retained_relations, new_edits_to_try)
                            all_edit_candidates.extend(new_edit_candidates)
                            edits_to_try.extend(new_edits_to_try)
                            self.operand_to_tried_edits[operand].append(f"e_{edit_ind}")
                            if len(all_edit_candidates) > 3:
                                break

            else:
                # TODO: Fix this
                raise ValueError("This should not happen")

            print("Final merging Solutions with new candidates")
            if reject_null:
                selected_candidates = remove_null_edits(all_edit_candidates)
            else:
                selected_candidates = all_edit_candidates

            selected_candidates = merge_finalized_edits(operand, selected_candidates)
            print(f"Null + Merging: Before had {len(all_edit_candidates)} and now have {len(selected_candidates)}")
            # Removing based on geometry
            n_edit_candidates = len(selected_candidates)

            # selected_candidates = remove_flips(operand, selected_candidates)
            selected_candidates = remove_high_distortion(operand, selected_candidates)
            selected_candidates = sort_by_typed_distortion_energy(operand, selected_candidates, heuristic=self.heuristic)

            print(f"Geometric: Before had {n_edit_candidates} and now have {len(selected_candidates)}")

        return selected_candidates

    def gather_new_options(self, operand, retained_relations, previous_edits, previous_candidates, shape, type_hints=None):
        if isinstance(operand, Relation):
            # If translation/rotation return count/delta edit.
            # If contact -> add an empty edit?
            # If reflection -> add an empty edit?

            selected_candidates = operand.gather_update_options()

        else:
            # here we will add the n-edit loop
            print("gathering new options for ", operand)
            solve_edits = solve_edits_v0
            n_edits = len(operand.primitive.edit_sequence)
            edits_to_try = get_edits_to_try(operand)
            edits_to_try = type_based_prune(operand, edits_to_try, type_hints)
            all_edit_candidates = previous_candidates
            # all_edit_candidates = solve_edits(operand, retained_relations, edits_to_try)
            if n_edits == 0:
                if len(all_edit_candidates) < MIN_EDIT_CANDIDATES:
                    print('Trying edits from relations')
                    n_relations = len(retained_relations)
                    # Generate edits from existing validated relations
                    new_edits_to_try = relation_based_edit_candidates(
                        operand, shape, edits_to_try)
                    new_edits_to_try = type_based_prune(operand, new_edits_to_try, type_hints)

                    new_relations = False
                    for relation in retained_relations:
                        relation_key = f"r_{relation.relation_index}"
                        if not relation_key in self.operand_to_tried_edits[operand]:
                            new_relations = True
                            self.operand_to_tried_edits[operand].append(relation_key)
                    if new_relations:
                        real_edits_to_try = new_edits_to_try
                    else:
                        real_edits_to_try = []
                        edits_to_try.extend(new_edits_to_try)

                    new_edit_candidates = solve_edits(operand, retained_relations, real_edits_to_try)
                    all_edit_candidates.extend(new_edit_candidates)
                    edits_to_try.extend(real_edits_to_try)

                    if len(all_edit_candidates) < MIN_EDIT_CANDIDATES:
                        print('Trying edits from sibling edits')
                        sibling_edits_to_try = []
                        for relation in retained_relations:
                            if isinstance(relation, FeatureRelation):
                                parts = [
                                    x.primitive.part for x in relation.features]
                            elif isinstance(relation, PrimitiveRelation):
                                parts = [x.part for x in relation.primitives]
                            other_parts = [x for x in parts if x != operand]
                            for other_part in other_parts:
                                sibling_edits = other_part.primitive.edit_sequence
                                sibling_edits_to_try.extend(sibling_edits)

                        for edit_ind, edit in enumerate(sibling_edits_to_try):
                            edit_key = f"e_{edit_ind}"
                            new_edits_to_try = get_additional_edit_candidates(operand, edit, edits_to_try)
                            new_edits_to_try = type_based_prune(operand, new_edits_to_try, type_hints)

                            if not edit_key in self.operand_to_tried_edits[operand]:
                                new_edit_candidates = solve_edits(
                                    operand, retained_relations, new_edits_to_try)
                                all_edit_candidates.extend(new_edit_candidates)
                                edits_to_try.extend(new_edits_to_try)
                                self.operand_to_tried_edits[operand].append(
                                    edit_key)
                            else:
                                edits_to_try.extend(new_edits_to_try)
                            if len(all_edit_candidates) > 3:
                                break
            else:
                raise ValueError("This should not happen")
                edits_to_try = get_edits_to_try(operand)
                all_edit_candidates = solve_edits(
                    operand, retained_relations, edits_to_try)
                # This is a reset...
                # TODO: Fix this
                # update a part
                # all_edit_candidates = solve_edits_with_prexisting(operand, retained_relations, edits_to_try)

            print("Final Merging Solutions")
            # selected_candidates = remove_null_edits(all_edit_candidates)
            selected_candidates = merge_finalized_edits(operand, selected_candidates)
            print(
                f"Null + Merging: Before had {len(all_edit_candidates)} and now have {len(selected_candidates)}")
            # Removing based on geometry
            n_edit_candidates = len(selected_candidates)

            # selected_candidates = remove_flips(operand, selected_candidates)
            selected_candidates = remove_high_distortion(operand, selected_candidates, return_sorted=True)
            selected_candidates = sort_by_typed_distortion_energy(operand, selected_candidates, heuristic=self.heuristic)
            print(
                f"Geometric: Before had {n_edit_candidates} and now have {len(selected_candidates)}")

        return selected_candidates


def solve_edits_breaking(target_part, retained_relations, edits_to_try):
    
    new_var = sp.Symbol("Y")
    old_var = MAIN_VAR
    st = time.time()
    all_edit_candidates = []
    solution_failures = {}
    solution_map = {}
    for ind, edit_gen in enumerate(edits_to_try):
        if ind % 10 == 0:
            print("ind is ", ind,
                  f"testing a edit of type {edit_gen.edit_class.__name__}")
        # print("trying", edit_gen)
        new_edit = edit_gen.employ(operand=target_part, amount=new_var)
        new_edit.propagate()

        discrepancy_list = []
        ind_to_relation_map = {}
        for relation in retained_relations:
            # assume is instance point contact.
            discrepancy = relation.get_discrepancy()
            if relation.initial_errors is not None:
                for i in range(discrepancy.shape[0]):
                    discrepancy[i,:] += relation.initial_errors[i]
            size = discrepancy.shape[0] * discrepancy.shape[1]
            for cur_ind in range(size):
                ind_to_relation_map[len(discrepancy_list)] = relation
                discrepancy_list.append(discrepancy[cur_ind])
        # Now we must solve and then try it for the rest.
        size = len(discrepancy_list)
        # there are k equations, and some value of X, Y should ensure all are ~ 0.
        # Step 1: See if any of the equations can never be satisfied.
        disc_state = []
        solvable_eq = []
        failure_eq_inds = []
        cur_ind = 0
        eq_to_real_ind_mapper = {}
        for cur_eq in discrepancy_list:
            if (new_var in cur_eq.free_symbols) and (old_var in cur_eq.free_symbols):
                disc_state.append(1)  # can be solved
                eq_to_real_ind_mapper[len(solvable_eq)] = cur_ind
                solvable_eq.append(cur_eq)
            else:
                if evaluate_equals_zero(cur_eq, order=3, mode=2):
                    disc_state.append(0)  # always true
                else:
                    disc_state.append(-1)
                    failure_eq_inds.append(cur_ind)
            cur_ind += 1

        for eq_ind, cur_eq in enumerate(solvable_eq):
            # print("trying to solve", eq_ind)
            solutions = sp.solve(cur_eq, new_var,
                                    exclude=[old_var],
                                    rational=None,
                                    implicit=True,  # TODO Check this.
                                    check=False,
                                    dict=True)
            # print(cur_eq)
            # print(solutions)
            # all_solutions.extend(solutions)
            # if len(solutions) == 0:
            #     solutions = sp.solve(cur_eq, new_var,
            #                         exclude=[old_var],
            #                         rational=None,
            #                         check=False
            #                         )
            for sol_ind, cur_solution in enumerate(solutions):
                solution_key = (ind, eq_ind, sol_ind)
                solution_map[solution_key] = cur_solution
                solution_failures[solution_key] = [x for x in failure_eq_inds]
                for temp_ind, temp_eq in enumerate(solvable_eq):
                    if temp_ind == eq_ind:
                        continue
                    cur_eq = temp_eq.subs(cur_solution)
                    if not evaluate_equals_zero(cur_eq, order=3, mode=2):
                        solution_failures[solution_key].append(eq_to_real_ind_mapper[temp_ind])

        # try x=y
        if len(solvable_eq) > 0 and len(solutions) == 0:
            solution = {new_var: old_var}
            solution_key = (ind, -1, -1)
            solution_map[solution_key] = solution
            solution_failures[solution_key] = [x for x in failure_eq_inds]
            for temp_ind, temp_eq in enumerate(solvable_eq):
                cur_eq = temp_eq.subs(solution)
                if not evaluate_equals_zero(cur_eq, order=3, mode=2):

                    # print("failed at ", cur_solution, "with", cur_eq)
                    solution_failures[solution_key].append(temp_ind)

        clean_up_motion(target_part, new_edit)

    solution_list = []
    for key, solution in solution_map.items():
        cur_solution = solution[new_var]
        if valid_solution(cur_solution):
            edit_ind, eq_ind, sol_ind = key
            edit_gen = edits_to_try[edit_ind]
            edit_candidate = EditGen(edit_gen.edit_class, edit_gen.param_dict, amount=cur_solution)
            broken_eq_inds = solution_failures[key]
            broken_relations = [ind_to_relation_map[x] for x in broken_eq_inds]
            broken_relations = list(set(broken_relations))
            solution_list.append((edit_candidate, broken_relations, broken_eq_inds))
    
    # sort them both in decesnding order of cost.
    solution_list = sorted(solution_list, key=lambda x: len(x[2]))

            
    et = time.time()
    print("Time for gathering options", et - st)
    # Merge the candidates -> when two are similar keep only the one with smaller delta.
    return solution_list

def solve_edits_v0(target_part, retained_relations, edits_to_try,
                   single_solution_mode=True,
                   hasten_search=True,
                   solver_mode="BASE"):
    new_var = sp.Symbol("Y")
    old_var = MAIN_VAR
    st = time.time()
    all_edit_candidates = []
    for ind, edit_gen in enumerate(edits_to_try):
        if ind % 10 == 0:
            print("ind is ", ind,
                  f"testing a edit of type {edit_gen.edit_class.__name__}")
        # print("trying", edit_gen)
        new_edit = edit_gen.employ(operand=target_part, amount=new_var)
        new_edit.propagate()

        discrepancy_list = []
        for relation in retained_relations:
            # assume is instance point contact.
            discrepancy = relation.get_discrepancy()
            if relation.initial_errors is not None:
                for i in range(discrepancy.shape[0]):
                    discrepancy[i,:] += relation.initial_errors[i]
            size = discrepancy.shape[0] * discrepancy.shape[1]
            for cur_ind in range(size):
                discrepancy_list.append(discrepancy[cur_ind])
        # Now we must solve and then try it for the rest.

        current_solution_set = set()
        cur_ind = 0
        size = len(discrepancy_list)
        # there are k equations , and some value of X, Y should ensure all are ~ 0.
        # Step 1: See if any of the equations can never be satisfied.
        disc_state = []
        solvable_eq = []
        breaking_y_eqs = []
        for cur_eq in discrepancy_list:
            if (new_var in cur_eq.free_symbols) and (old_var in cur_eq.free_symbols):
                disc_state.append(1)  # can be solved
                solvable_eq.append(cur_eq)
            else:
                if evaluate_equals_zero(cur_eq, order=3, mode=2):
                    disc_state.append(0)  # always true
                    breaking_y_eqs.append(cur_eq)
                else:
                    disc_state.append(-1)
                    break
        current_solution_set = list()
        eq_types = set(disc_state)
        if not -1 in eq_types:
            # all_solutions = []
            if solver_mode == "BASE":
                for eq_ind, cur_eq in enumerate(solvable_eq):
                    # print("trying to solve", eq_ind)
                    if hasten_search and eq_ind > 0:
                        # if we already have some solutions break
                        # Variant 1.
                        if len(all_edit_candidates) >= MIN_EDIT_CANDIDATES:
                            break

                    # solutions = sp.solve(cur_eq, new_var)
                    solutions = sp.solve(cur_eq, new_var,
                                         exclude=[old_var],
                                         rational=None,
                                         implicit=True,  # TODO Check this.
                                         check=False,
                                         dict=True
                                         )
                    # print(cur_eq)
                    # print(solutions)
                    # all_solutions.extend(solutions)
                    # if len(solutions) == 0:
                    #     solutions = sp.solve(cur_eq, new_var,
                    #                         exclude=[old_var],
                    #                         rational=None,
                    #                         check=False
                    #                         )
                    for cur_solution in solutions:
                        # if not valid_solution(cur_solution[new_var]):
                        #     continue
                        cur_sol_valid = True
                        for temp_ind, temp_eq in enumerate(solvable_eq):
                            if temp_ind == eq_ind:
                                continue
                            cur_eq = temp_eq.subs(cur_solution)
                            if not evaluate_equals_zero(cur_eq, order=3, mode=2):
                                cur_sol_valid = False
                                break
                        if cur_sol_valid:
                            current_solution_set.append(cur_solution)
                            if single_solution_mode:
                                break
                    if single_solution_mode and len(current_solution_set) > 0:
                        break
            else:
                # try:
                #     inequalities = [
                #         x <= QUANT_COMPARE_THRESHOLD_MODE_2 for x in solvable_eq]
                #     sols = sp.solve(inequalities, new_var, exclude=[old_var], rational=None, implicit=True, check=False, dict=True)
                #     sols = sp.simplify(sols)
                #     if not sols:
                #         # Nothing satisfies all the inequalities
                #         continue
                #     for sol in sols.args:
                #         # if has both variables
                #         if old_var in sol.free_symbols and new_var in sol.free_symbols:
                #             eq = sol.args[0] - sol.args[1]
                #             current_solution = sp.solve(eq, new_var, exclude=[
                #                                         old_var], rational=None, implicit=True, check=False, dict=True)
                #             if len(current_solution) > 0:
                #                 current_solution_set.extend(current_solution)
                # except:
                #     pass
                pass
            # try x=y
            if len(current_solution_set) == 0:
                solution = {new_var: old_var}
                cur_sol_valid = True
                for temp_eq in solvable_eq:
                    cur_eq = temp_eq.subs(solution)
                    if not evaluate_equals_zero(cur_eq, order=3, mode=2):
                        # print("failed at ", cur_solution, "with", cur_eq)
                        cur_sol_valid = False
                        break
                if cur_sol_valid:
                    current_solution_set.append(solution)

        if len(current_solution_set) > 0:
            # try it for all discrepancies.
            for solution in current_solution_set:
                cur_solution = solution[new_var]
                valid = True
                for eq in breaking_y_eqs:
                    if not evaluate_equals_zero(eq.subs(solution), order=3, mode=2):
                        valid = False
                        break
                if not valid:
                    continue
                if valid_solution(cur_solution):
                    # if valid_solution(solution):
                    edit_candidate = EditGen(
                        edit_gen.edit_class, edit_gen.param_dict, amount=cur_solution)
                    print("success with", edit_candidate)
                    all_edit_candidates.append(edit_candidate)

        clean_up_motion(target_part, new_edit)

    et = time.time()
    print("Time for gathering options", et - st)
    # Merge the candidates -> when two are similar keep only the one with smaller delta.
    return all_edit_candidates


def type_based_prune(operand, edits_to_try, type_hints):
    all_edits = [x for x in edits_to_try if x.edit_class in type_hints]
    return all_edits



def get_edits_to_try(part, add_face_rotates=False, valid_types=None):
    # important points,
    # Can we find this before hand?
    # TODO: Update the roster based on the previous edits
    # Consider directions and points of interest based on them.

    directions = get_directions(part)
    face_names = {x for x, y in part.name_to_indices.items() if len(y) == 4}
    edge_names = {x for x, y in part.name_to_indices.items() if len(y) == 2}

    translate_edits = []
    for cur_dir in directions:
        translate_edits.append(
            EditGen(PartTranslate, param_dict={'direction': cur_dir}))
    # face translates
    for face_name in face_names:
        for cur_dir in directions:
            translate_edits.append(EditGen(FaceTranslate, param_dict={
                                   'direction': cur_dir, 'restrictor_name': face_name}))
    # Edge translates
    for edge_name in edge_names:
        edge_inds = part.name_to_indices[edge_name]
        points = part.primitive.point_set
        sel_points = [points[x, :] for x in edge_inds]
        edge_vec = (sel_points[1] - sel_points[0]).normalized()
        for cur_dir in directions:
            if sp.Abs(cur_dir.dot(edge_vec)) < 0.9:
                translate_edits.append(EditGen(EdgeTranslate, param_dict={
                                       'direction': cur_dir, 'restrictor_name': edge_name}))

    # Rotation: Only part level
    # origin and direction can be many though.
    rotate_edits = []
    for cur_dir in directions:
        rotate_edits.append(EditGen(PartRotate, param_dict={
                            'rotation_axis': cur_dir, 'rotation_origin': part.center()}))
    for face_name in face_names:
        face_center = part.face_center(face_name)
        face_normal = (face_center - part.center()).normalized()
        for cur_dir in directions:
            if sp.Abs(face_normal.dot(cur_dir)) < 0.9:
                rotate_edits.append(EditGen(PartRotate, param_dict={
                                    'rotation_axis': cur_dir, 'rotation_origin': face_center}))
                if add_face_rotates:
                    rotate_edits.append(EditGen(FaceRotate, param_dict={
                                        'rotation_axis': cur_dir, 'rotation_origin': face_center, 'restrictor_name': face_name}))
    for edge_name in edge_names:
        edge_center = part.edge_center(edge_name[0], edge_name[1])
        for cur_dir in directions:
            rotate_edits.append(EditGen(PartRotate, param_dict={
                                    'rotation_axis': cur_dir, 'rotation_origin': edge_center}))
            
    # Scaling1D: Only part level
    scale_1d_edits = []
    for cur_dir in directions:
        scale_1d_edits.append(EditGen(PartScale1D, param_dict={
                              'direction': cur_dir, 'origin': part.center()}))
    for face_name in face_names:
        face_center = part.face_center(face_name)
        face_normal = (face_center - part.center()).normalized()
        for cur_dir in directions:
            if sp.Abs(face_normal.dot(cur_dir)) < 0.9:
                scale_1d_edits.append(EditGen(FaceScale1D, param_dict={
                                      'direction': cur_dir, 'origin': part.center(), 'restrictor_name': face_name}))
    # Scaling2D: Part Level, and Face Level
    scale_2d_edits = []
    for cur_dir in directions:
        scale_2d_edits.append(EditGen(PartScale2D, param_dict={
                              'plane_normal': cur_dir, 'origin': part.center()}))
    for face_name in face_names:
        face_center = part.face_center(face_name)
        face_normal = (face_center - part.center()).normalized()
        for cur_dir in directions:
            if sp.Abs(face_normal.dot(cur_dir)) > 0.9:
                scale_2d_edits.append(EditGen(FaceScale2D, param_dict={
                                      'plane_normal': cur_dir, 'origin': part.center(), 'restrictor_name': face_name}))

    # Scale 3D:
    scale_3d_edits = [
        EditGen(PartScale3D, param_dict={'origin': part.center()})
    ]
    for face_name in face_names:
        face_center = part.face_center(face_name)
        scale_3d_edits.append(
            EditGen(PartScale3D, param_dict={'origin': face_center}))

    # Shear is covered in the part level edits.. Maybe the 2D shears?
    shear_edits = []
    for dir_1 in directions:
        for dir_2 in directions:
            cr = dir_1.cross(dir_2).norm()
            if not evaluate_equals_zero(cr):
                shear_edits.append(EditGen(PartShear, param_dict={
                                   'shear_direction': dir_1, 'shear_plane_normal': dir_2, 'shear_plane_origin': part.center()}))

    all_edits = translate_edits + rotate_edits + scale_1d_edits + \
        scale_2d_edits + scale_3d_edits + shear_edits
    
    if valid_types is not None:
        all_edits = [x for x in all_edits if x.edit_class in valid_types]
    return all_edits


def relation_based_edit_candidates(part, shape, old_edits_to_try):

    relations = shape.all_relations(only_active=True)
    retained_relations = [
        x for x in relations if x.state[2] == RELATION_RETAINED]
    # now from these only the ones that have to be resolved - not the ones that must be updated
    retained_relations = [
        x for x in retained_relations if x.state[3] != UPDATE_RELATION]
    all_new_edits = []
    face_names = {x for x, y in part.name_to_indices.items() if len(y) == 4}
    edge_names = {x for x, y in part.name_to_indices.items() if len(y) == 2}

    for relation in retained_relations:
        if isinstance(relation, FeatureRelation):
            # we have points. # what if its moving?
            ...
        elif isinstance(relation, ReflectionSymmetry):
            point = relation.plane.point
            direction = relation.plane.normal
            # add translation along dir
            all_new_edits.append(
                EditGen(PartTranslate, param_dict={'direction': direction}))
            for face_name in face_names:
                all_new_edits.append(EditGen(FaceTranslate, param_dict={
                                     'direction': direction, 'restrictor_name': face_name}))
            for edge_name in edge_names:
                edge_inds = part.name_to_indices[edge_name]
                points = part.primitive.point_set
                sel_points = [points[x, :] for x in edge_inds]
                edge_vec = (sel_points[1] - sel_points[0]).normalized()
                if sp.Abs(direction.dot(edge_vec)) < 0.9:
                    all_new_edits.append(EditGen(EdgeTranslate, param_dict={
                                         'direction': direction, 'restrictor_name': edge_name}))
            all_new_edits.append(EditGen(PartScale1D, param_dict={
                                 'direction': direction, 'origin': point}))
            for face_name in face_names:
                face_center = part.face_center(face_name)
                face_normal = (face_center - part.center()).normalized()
                if sp.Abs(face_normal.dot(direction)) < 0.9:
                    all_new_edits.append(EditGen(FaceScale1D, param_dict={
                                         'direction': direction, 'origin': point, 'restrictor_name': face_name}))

            all_new_edits.append(EditGen(PartScale2D, param_dict={
                                 'plane_normal': direction, 'origin': point}))
            for face_name in face_names:
                face_center = part.face_center(face_name)
                face_normal = (face_center - part.center()).normalized()
                if sp.Abs(face_normal.dot(direction)) > 0.9:
                    all_new_edits.append(EditGen(FaceScale2D, param_dict={
                                         'plane_normal': direction, 'origin': point, 'restrictor_name': face_name}))
            all_new_edits.append(
                EditGen(PartScale3D, param_dict={'origin': point}))

            # what
            # if the direction is new, then it makes sense to try face scales as well.
        elif isinstance(relation, TranslationSymmetry):
            direction = relation.delta.normalized()
        elif isinstance(relation, RotationSymmetry):
            point = relation.axis.point
            direction = relation.axis.direction

    return all_new_edits


def get_additional_edit_candidates(part, edit):
    all_new_edits = []

    face_names = {x for x, y in part.name_to_indices.items() if len(y) == 4}
    edge_names = {x for x, y in part.name_to_indices.items() if len(y) == 2}

    if isinstance(edit, PartEdit):
        edit_part = edit.operand
        try_all = False
        if isinstance(edit, PartTranslate):
            # Specific to the type of edit

            new_ed_gen = EditGen(PartTranslate, param_dict={
                                 'direction': edit.direction})
            all_new_edits.append(new_ed_gen)
            for face_name in face_names:
                new_ed_gen = EditGen(FaceTranslate, param_dict={
                                     'direction': edit.direction, 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)
            for edge_name in edge_names:
                new_ed_gen = EditGen(EdgeTranslate, param_dict={
                                     'direction': edit.direction, 'restrictor_name': edge_name})
                all_new_edits.append(new_ed_gen)
        elif isinstance(edit, FaceTranslate):
            # Specific to the type of edit
            opposite_name = OPPOSITE_NAME[edit.restrictor_name]
            center = edit_part.face_center(opposite_name)
            new_ed_gen = EditGen(PartScale1D, param_dict={
                                 'direction': edit.direction, 'origin': center})
            all_new_edits.append(new_ed_gen)
            for face in face_names:
                new_ed_gen = EditGen(FaceScale1D, param_dict={
                                     'direction': edit.direction, 'origin': center, "restrictor_name": face})
                all_new_edits.append(new_ed_gen)

        elif isinstance(edit, EdgeTranslate):
            opposite_name = (OPPOSITE_NAME[x] for x in edit.restrictor_name)
            center = edit_part.edge_center(*opposite_name)
            new_ed_gen = EditGen(PartScale1D, param_dict={
                                 'direction': edit.direction, 'origin': center})
            all_new_edits.append(new_ed_gen)
            for face in face_names:
                new_ed_gen = EditGen(FaceScale1D, param_dict={
                                     'direction': edit.direction, 'origin': center, "restrictor_name": face})
                all_new_edits.append(new_ed_gen)

        elif isinstance(edit, PartRotate):
            # Specific to the type of edit
            new_ed_gen = EditGen(PartRotate, param_dict={
                                 'rotation_axis': edit.rotation_axis, 'rotation_origin': edit.rotation_origin})
            all_new_edits.append(new_ed_gen)
            for face_name in face_names:
                new_ed_gen = EditGen(FaceRotate, param_dict={
                                     'rotation_axis': edit.rotation_axis, 'rotation_origin': edit.rotation_origin, 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)
        elif isinstance(edit, PartScale1D):
            # Specific to the type of edit
            new_ed_gen = EditGen(PartTranslate, param_dict={
                                 'direction': edit.direction})
            all_new_edits.append(new_ed_gen)
            for face_name in face_names:
                new_ed_gen = EditGen(FaceTranslate, param_dict={
                                     'direction': edit.direction, 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)

            new_ed_gen = EditGen(PartScale1D, param_dict={
                                 'direction': edit.direction, 'origin': edit.origin})
            all_new_edits.append(new_ed_gen)
            for face_name in face_names:
                new_ed_gen = EditGen(FaceScale1D, param_dict={
                                     'direction': edit.direction, 'origin': edit.origin, 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)
        elif isinstance(edit, FaceScale1D):
            opposite_face = OPPOSITE_NAME[edit.restrictor_name]
            center = edit_part.face_center(opposite_face)
            new_ed_gen = EditGen(PartScale1D, param_dict={
                                 'direction': edit.direction, 'origin': center})
            all_new_edits.append(new_ed_gen)

            new_ed_gen = EditGen(PartTranslate, param_dict={
                                 'direction': edit.direction})
            all_new_edits.append(new_ed_gen)
            for face_name in face_names:
                new_ed_gen = EditGen(FaceTranslate, param_dict={
                                     'direction': edit.direction, 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)

            for face_name in face_names:
                new_ed_gen = EditGen(FaceScale1D, param_dict={
                                     'direction': edit.direction, 'origin': center, 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)
        elif isinstance(edit, PartScale2D):
            # Specific to the type of edit
            joiner = part.primitive.center() - edit.operand.center()
            joiner = joiner - joiner.dot(edit.plane_normal) * edit.plane_normal
            direction = joiner.normalized()
            new_ed_gen = EditGen(PartTranslate, param_dict={
                                 'direction': direction})
            all_new_edits.append(new_ed_gen)
            for face_name in face_names:
                new_ed_gen = EditGen(FaceTranslate, param_dict={
                                     'direction': direction, 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)
                new_ed_gen = EditGen(FaceScale1D, param_dict={
                                     'direction': direction, 'restrictor_name': face_name, 'origin': edit.origin})
                all_new_edits.append(new_ed_gen)
                new_ed_gen = EditGen(FaceScale2D, param_dict={
                                     'plane_normal': direction, 'restrictor_name': face_name, 'origin': edit.origin})
                all_new_edits.append(new_ed_gen)

            new_ed_gen = EditGen(PartScale2D, param_dict={
                                 'plane_normal': edit.plane_normal, 'origin': edit.origin})
            all_new_edits.append(new_ed_gen)
            for face_name in face_names:
                new_ed_gen = EditGen(FaceScale2D, param_dict={
                                     'plane_normal': edit.plane_normal, 'origin': edit.origin, 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)
        elif isinstance(edit, FaceScale2D):
            opposite_face = OPPOSITE_NAME[edit.restrictor_name]
            center = edit_part.face_center(opposite_face)
            new_ed_gen = EditGen(PartScale2D, param_dict={
                                 'plane_normal': edit.plane_normal, 'origin': center})
            all_new_edits.append(new_ed_gen)
            for face_name in face_names:
                new_ed_gen = EditGen(FaceScale2D, param_dict={
                                     'plane_normal': edit.plane_normal, 'origin': center, 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)
        elif isinstance(edit, PartScale3D):

            new_ed_gen = EditGen(PartTranslate, param_dict={
                                 'direction': edit.origin - part.center()})
            all_new_edits.append(new_ed_gen)
            for face_name in face_names:
                new_ed_gen = EditGen(FaceTranslate, param_dict={
                                     'direction': edit.origin - part.center(), 'restrictor_name': face_name})
                all_new_edits.append(new_ed_gen)
            try_all = True
        elif isinstance(edit, KeepFixed):

            new_edits_to_try = get_edits_to_try(
                edit_part, add_face_rotates=False)
            for ed_gen in new_edits_to_try:
                all_new_edits.append(ed_gen)
        else:
            try_all = True
        if try_all:

            new_edits_to_try = get_edits_to_try(
                edit_part, add_face_rotates=False)
            for ed_gen in new_edits_to_try:
                # if issubclass(ed_gen.edit_class, RestrictedEdit):
                #     restrictor_name = ed_gen.param_dict['restrictor_name']
                #     if restrictor_name in face_names:
                #         for face_name in face_names:
                #             params = {x: y for x,
                #                       y in ed_gen.param_dict.items()}
                #             params['restrictor_name'] = face_name
                #             all_new_edits.append(
                #                 EditGen(ed_gen.edit_class, param_dict=params))
                #     elif restrictor_name in edge_names:
                #         for edge_name in edge_names:
                #             params = {x: y for x,
                #                       y in ed_gen.param_dict.items()}
                #             params['restrictor_name'] = edge_name
                #             all_new_edits.append(
                #                 EditGen(ed_gen.edit_class, param_dict=params))
                # else:
                    all_new_edits.append(ed_gen)
    # all new edits should have face translate.
    # then check if they are being rejected.
    return all_new_edits

