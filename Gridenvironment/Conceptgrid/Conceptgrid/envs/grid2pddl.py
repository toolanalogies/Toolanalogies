from turtle import pos, width
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from Conceptgrid.envs.Concept_classes2 import GridEnv2

import itertools
import os
from typing import Tuple

class PDDLWriter():

    domain_template = """(define
        (domain {problem_name})
        (:requirements :typing :strips :equality :conditional-effects :negative-preconditions :quantified-preconditions :disjunctive-preconditions)

        (:types
            objects
            walls
            textiles
            position
            direction
            textile_id
            textile_color
            textile_termination
            object_id
            object_color
            object_shape
            
        )

        (:constants
            up down left right - direction
            ! ? - textile_id
            orange white - textile_color
            success death - textile_termination
            A B G R W - object_id
            red blue green brown black - object_color
            square triangle round - object_shape
        )

        (:predicates
            (at ?obj - objects ?pos - position)
            (at_wall ?wll - walls ?pos - position)
            (at_txtl ?txtl - textiles ?pos - position)
            (connected ?from - position ?to - position ?dir - direction)
            (is_movable ?obj - objects)
            (is_controllable ?obj - objects)
            (has_txt_term ?txtl - textiles ?txtl_term - textile_termination)
            (has_txt_color ?txtl - textiles ?txtl_color - textile_color)
            (has_obj_color ?obj - objects ?obj_color - object_color)
            (has_wall_color ?wll - walls ?obj_color - object_color)
            (should_move ?obj - objects ?dir - direction)
            (stop_agent ?dir - direction)
            (agent_dead)
            (goal_true)
            (is_empty_obj ?pos - position)
            (is_empty_textile ?pos - position)
        )

    ; MOVE AGENT IN FREE SPACE
        (:action move_agent
            :parameters (?agent - objects ?dir - direction ?from - position ?to - position)
            :precondition (and
                (is_controllable ?agent)
                (at ?agent ?from)
                (connected ?from ?to ?dir)
                (not (stop_agent ?dir))
                (is_empty_obj ?to)
                (is_empty_textile ?to)
                (not (agent_dead))
            )
            :effect (and
                (not (at ?agent ?from))
                (at ?agent ?to)
                (is_empty_obj ?from)
                (not (is_empty_obj ?to))
                (forall (?dir2 - direction)
                    (not (stop_agent ?dir2))
                )
            )
        )

        ; MOVE AGENT TO THE LOC OTHER OBJECT

        (:action move_agent_to_movable
            :parameters (?agent - objects ?pushed_into - objects ?dir - direction ?from - position ?to - position)
            :precondition (and
                (is_controllable ?agent)
                (at ?agent ?from)
                (connected ?from ?to ?dir)
                (at ?pushed_into ?to)
                (not (stop_agent ?dir))
                (is_movable ?pushed_into)
                (not (should_move ?pushed_into ?dir))
                (is_empty_textile ?to)
                (not (agent_dead))
            )
            :effect (and
                (should_move ?pushed_into ?dir)
                (forall (?dir2 - direction)
                    (when (not (= ?dir ?dir2))
                        (stop_agent ?dir2)
                    )
                )
            )
        )

    ; FOR ALL SHOULD MOVE TRUE
        (:action move_object
            :parameters (?being_pushed - objects ?dir - direction ?from - position ?to - position)
            :precondition (and
                (not (is_controllable ?being_pushed))
                (at ?being_pushed ?from)
                (connected ?from ?to ?dir)
                (should_move ?being_pushed ?dir)
                (is_empty_obj ?to)
                (not (has_obj_color ?being_pushed red))
                (not (agent_dead))
            )
            :effect (and
                (not (at ?being_pushed ?from))
                (at ?being_pushed ?to)
                (is_empty_obj ?from)
                (not (is_empty_obj ?to))
                (not (should_move ?being_pushed ?dir))
            )
        )

    ; TWO OBJECTS NOT MOVABLE FOR ALL SHOULD MOVE FALSE
        (:action move_object_to_not_movable
            :parameters (?being_pushed - objects ?pushed_into - objects ?dir - direction ?from - position ?to - position)
            :precondition (and
                (not (is_controllable ?being_pushed))
                (at ?being_pushed ?from)
                (connected ?from ?to ?dir)
                (should_move ?being_pushed ?dir)
                (at ?pushed_into ?to)
                (not (is_movable ?pushed_into))
                (not (agent_dead))            
            )
            :effect (and
                        (forall (?obj - objects)
                            (not (should_move ?obj ?dir))
                        )
                        (forall (?dir2 - direction)
                            (when (not (= ?dir ?dir2))
                                (not (stop_agent ?dir2))
                            )
                        )  
                        (stop_agent dir)                 
            )
        )

    ; TWO OBJECTS NOT MOVABLE FOR ALL SHOULD MOVE FALSE
        (:action move_object_to_wall
            :parameters (?being_pushed - objects ?dir - direction ?from - position ?to - position)
            :precondition (and
                (not (is_controllable ?being_pushed))
                (at ?being_pushed ?from)
                (connected ?from ?to ?dir)
                (should_move ?being_pushed ?dir)
                (exists (?wll - walls)
                        (at_wall ?wll ?to))
                (not (agent_dead))            
            )
            :effect (and
                        (forall (?obj - objects)
                            (not (should_move ?obj ?dir))
                        )
                        (forall (?dir2 - direction)
                            (when (not (= ?dir ?dir2))
                                (not (stop_agent ?dir2))
                            )
                        )  
                        (stop_agent dir)                 
            )
        )

    ; SETS OBJECTS TO MUST MOVE IF MOVABLE

        (:action move_object_to_movable
            :parameters (?being_pushed - objects ?pushed_into - objects ?dir - direction ?from - position ?to - position)
            :precondition (and
                (not (is_controllable ?being_pushed))
                (at ?being_pushed ?from)
                (connected ?from ?to ?dir)
                (should_move ?being_pushed ?dir)
                (at ?pushed_into ?to)
                (is_movable ?pushed_into)
                (not (should_move ?pushed_into ?dir))
                (not (agent_dead))
            )

            :effect (should_move ?pushed_into ?dir)
        )

    ; PUTTING RED_BALL ONTO GOAL LOCATION

        (:action move_red_to_goal
            :parameters (?being_pushed - objects ?dir - direction ?from - position ?to - position)
            :precondition (and
                (not (is_controllable ?being_pushed))
                (at ?being_pushed ?from)
                (has_obj_color ?being_pushed red)
                (connected ?from ?to ?dir)
                (should_move ?being_pushed ?dir)
                (is_empty_obj ?to)
                (exists (?txtl - textiles)
                    (and
                        (at_txtl ?txtl ?to)
                        (has_txt_term ?txtl success)
                    )
                )
                (not (agent_dead))
            )
            :effect (and
                (not (at ?being_pushed ?from))
                (at ?being_pushed ?to)
                (is_empty_obj ?from)
                (not (is_empty_obj ?to))
                (not (should_move ?being_pushed ?dir))
                (agent_dead)
            )
        )

    ; PUTTING RED_BALL ONTO ANY LOCATION

        (:action move_red_to_any
            :parameters (?being_pushed - objects ?dir - direction ?from - position ?to - position)
            :precondition (and
                (not (is_controllable ?being_pushed))
                (at ?being_pushed ?from)
                (has_obj_color ?being_pushed red)
                (connected ?from ?to ?dir)
                (should_move ?being_pushed ?dir)
                (is_empty_obj ?to)
                (not
                    (exists (?txtl - textiles)
                        (and
                            (at_txtl ?txtl ?to)
                            (has_txt_term ?txtl success)
                        )
                    )
                )
                (not (agent_dead))
            )
            :effect (and
                (not (at ?being_pushed ?from))
                (at ?being_pushed ?to)
                (is_empty_obj ?from)
                (not (is_empty_obj ?to))
                (not (should_move ?being_pushed ?dir))
            )
        )

    ; GOAL PREDICATE

        (:action goal_action
            :parameters (?obj - objects ?txtl - textiles ?pos - position)
            :precondition (and
                (not (goal_true))
                (at ?obj ?pos)
                (at_txtl ?txtl ?pos)
                (has_obj_color ?obj green)
                (has_txt_term ?txtl success)
                (not (agent_dead))
                (not
                    (exists (?dir - direction)
                        (stop_agent ?dir)
                    )
                )
            )

            :effect ( goal_true
            )
        )
    )"""

    problem_template = """(define
        (problem {problem_name})
        (:domain {problem_name})

        (:objects
    {obj_and_positions_decl}
        )

        (:init
    {object_predicates}
    {textile_predicates}
            ; Define the grid of positions. (0, 0) is the top-left.
    {position_connections}
        )

        (:goal
            (goal_true)
        )
    )"""

    def grid2pddl(self, Grid):

        height = Grid.max_height
        width = Grid.max_width
        all_poses = []
        obj_and_positions_decl = "\n".join(
            (" " * 8 + " ".join([f"pos{x}-{y}" for x in range(width)]) + " - position")
            for y in range(height)
        )
        for x in range(width):
            for y in range(height):
                all_poses.append((x,y))

        position_connections = ""
        for y in range(height):
            for x in range(width):
                if x > 0:
                    position_connections += (
                        f"        (connected pos{x}-{y} pos{x-1}-{y} left)\n"
                    )
                if x + 1 < width:
                    position_connections += (
                        f"        (connected pos{x}-{y} pos{x+1}-{y} right)\n"
                    )
                if y > 0:
                    position_connections += (
                        f"        (connected pos{x}-{y} pos{x}-{y-1} up)\n"
                    )
                if y + 1 < height:
                    position_connections += (
                        f"        (connected pos{x}-{y} pos{x}-{y+1} down)\n"
                    )
        object_names = ""
        wall_names = ""
        textile_names = ""
        object_predicates = ""
        textile_predicates = ""
        occupied_obj = []
        occupied_txt = []
        for key, object_i in Grid.state.objects.items():
            
            if object_i.id == 'W':
                wall_names += f"{key} "
                for ftr in Grid.object_features_key:
                    ftr_val = getattr(object_i, ftr)
                    if ftr == 'x':                
                        obj_x = ftr_val                
                    elif ftr == 'y':
                        obj_y = ftr_val
                        object_predicates += (
                                f"        (at_wall {key} pos{obj_x}-{obj_y})\n"
                            )
                        occupied_obj.append((obj_x, obj_y))                
                    elif ftr == 'color':
                        object_predicates += (
                                f"        (has_wall_color {key} {ftr_val})\n"
                            )
                    else:
                        continue
            else:
                object_names += f"{key} "
                for ftr in Grid.object_features_key:
                    ftr_val = getattr(object_i, ftr)
                    if ftr == 'x':                
                        obj_x = ftr_val                
                    elif ftr == 'y':
                        obj_y = ftr_val
                        object_predicates += (
                                f"        (at {key} pos{obj_x}-{obj_y})\n"
                            )
                        occupied_obj.append((obj_x, obj_y))                
                    elif ftr == 'controllable':
                        if ftr_val == True:
                            object_predicates += (
                                f"        (is_controllable {key})\n"
                            )
                    elif ftr == 'movable':
                        if ftr_val == True:
                            object_predicates += (
                                f"        (is_movable {key})\n"
                            )
                    elif ftr == 'color':
                        object_predicates += (
                                f"        (has_obj_color {key} {ftr_val})\n"
                            )
                    else:
                        continue
                
        for key, txtl_i in Grid.state.textiles.items():
            if key == 'goal':
                textile_names += f"{key}ss "
            else:
                textile_names += f"{key} "
            for ftr in Grid.textile_features_key:
                ftr_val = getattr(txtl_i, ftr)
                if ftr == 'x':
                    txtl_x = ftr_val
                elif ftr == 'y':
                    txtl_y = ftr_val
                    if key == 'goal':
                        textile_predicates += (
                                f"        (at_txtl {key}ss pos{txtl_x}-{txtl_y})\n"
                            )
                    else:
                        textile_predicates += (
                                f"        (at_txtl {key} pos{txtl_x}-{txtl_y})\n"
                            )
                elif ftr == 'textile_termination':
                    if ftr_val == 'goal':
                        textile_predicates += (
                            f"        (has_txt_term {key}ss success)\n"
                        )
                    else:
                        textile_predicates += (
                            f"        (has_txt_term {key} {ftr_val})\n"
                        )
                        occupied_txt.append((txtl_x, txtl_y))
                elif ftr == 'color':
                    if ftr_val == 'white':
                        textile_predicates += (
                            f"        (has_txt_color {key}ss {ftr_val})\n"
                        )
                    else:
                        textile_predicates += (
                            f"        (has_txt_color {key} {ftr_val})\n"
                        )
                    
                else:
                    continue
                
                
        empty_obj = list(set(all_poses) - set(occupied_obj))
        empty_txt = list(set(all_poses) - set(occupied_txt))
        for p_tuple in empty_obj:
            x, y = p_tuple
            object_predicates += (
                                f"        (is_empty_obj pos{x}-{y})\n"
                            )
        for p_tuple in empty_txt:
            x, y = p_tuple
            textile_predicates += (
                                f"        (is_empty_textile pos{x}-{y})\n"
                            )

        obj_and_positions_decl += "\n" + " " * 8 + object_names + "- objects"
        obj_and_positions_decl += "\n" + " " * 8 + textile_names + "- textiles"
        
        print(object_names)
        print(obj_and_positions_decl)
        print(position_connections)
        print(object_predicates)
        print(textile_predicates)
        problem_name = 'ToolGrid'
        return (
            self.domain_template.format(
                problem_name=problem_name
            ),
            self.problem_template.format(
                problem_name=problem_name,
                obj_and_positions_decl=obj_and_positions_decl,
                position_connections=position_connections,
                object_predicates=object_predicates,
                textile_predicates=textile_predicates,
            ),
        )

if __name__ == "__main__":
    #print(domain_template)
    new_Grid = GridEnv2(state_mode = 'obj', randomize = False, tool_usage = True, change_object_size= False)
    plt.imsave('image.png',new_Grid.image_renderer())
    pddl_write = PDDLWriter()
    pddl_domain, pddl_problem = pddl_write.grid2pddl(new_Grid)
    with open('path to domain pddl', "w") as domain_file:
            domain_file.write(pddl_domain)

    with open('path to problem pddl', "w") as problem_file:
            problem_file.write(pddl_problem)
