import os
import time
import _pickle as cPickle
import sympy as sp
import open3d.visualization.gui as gui

from .base import update_switch_board, load_edits, get_amount_str

# just create some edits and call update procedural panel
from edit_sys.llm.common import KEY, MODEL, TEMPERATURE, SEED, MODE
import edit_sys.shape_system.parallel_prop as parallel_prop
from edit_sys.weave.new_algo import algo_v4
import edit_sys.llm as prompters #import HumanPrompter, LLMPrompterV1, PureGeometryPrompter
from edit_sys.data_loader.partnet_shape import get_obj
from edit_sys.shape_system.edits import MAIN_VAR
from edit_sys.shape_system.constants import PART_ACTIVE, PART_INACTIVE

from .base import EditSysApp

class OnlineEditSysApp(EditSysApp):

    def __init__(self, *args, **kwargs):
        self.item = kwargs.pop("item")
        self.repetitions = kwargs.pop("repetitions")
        super().__init__(*args, **kwargs)

        self._obj_panel.get_children()[0].set_is_open(False)
        # self._settings_panel.get_children()[0].set_is_open(False)

    def _append_edits_from_pickle(self, program_file, link_to_single=False):
        shape = self.symbolic_data[0]
        edit_gens = cPickle.load(open(program_file, "rb"))

        if not hasattr(self, "edits"):
            self.edits = {}

        def _append_one(program_item):
            FREE_VAR = sp.Symbol(f"f_{len(self.edits)}")
            for cur_edit in program_item:
                cur_edit[0].amount = cur_edit[0].amount.subs({MAIN_VAR: FREE_VAR})
            edits, _ = load_edits(shape, program_item)
            self.edits[f"Edit_{len(self.edits)}"] = (edits, FREE_VAR)

        if isinstance(edit_gens, dict):
            for _, program_item in edit_gens.items():
                _append_one(program_item)
        else:
            # single program
            _append_one(edit_gens)

        # finalize: propagate + refresh UI, just like current code path
        shape.clean_up_motion()
        
        for name, edit_item in self.edits.items():
            edit_list, sym = edit_item
            for edit in edit_list:
                edit.propagate()
        
        for part in shape.partset:
            if len(part.partset) == 0:
                continue
            else:
                # check activity
                core_rel = part.core_relation
                if len(core_rel.edit_sequence) > 0:
                    part.state[0] = PART_ACTIVE
                    for subpart in part.partset:
                        subpart.state[0] = PART_INACTIVE
                else:
                    if len(part.primitive.edit_sequence) > 0:
                        part.state[0] = PART_ACTIVE
                        for subpart in part.partset:
                            subpart.state[0] = PART_INACTIVE
                    else:
                        if any([len(subpart.primitive.edit_sequence) > 0 for subpart in part.partset]):
                            part.state[0] = PART_INACTIVE
                            for subpart in part.partset:
                                subpart.state[0] = PART_ACTIVE
                    
        for part in shape.partset:
            part_edited = len(part.primitive.edit_sequence) > 0
            if part_edited:
                part.state[0] = PART_ACTIVE
            if len(part.sub_parts) > 0:
                edited_children = [len(child.primitive.edit_sequence) > 0 for child in part.sub_parts]
                if any(edited_children):
                    print(f"Part {part.label} has edited children. Turned Off")
                    shape.deactivate_parent(part)

        self.update_from_switch_board()
        self.update_procedural_panel()
        self._edit_prebake()


    def get_edit(self, link_to_single=False, program_file=None):

        if hasattr(self, "symbol_to_parts"):
            for symbol, caller in self.symbol_value_dict.items():
                self.symbol_value_dict[symbol] = 0
                self._edit_execute(symbol)
        
        if program_file is not None:
            # Load from pickle instead of LLM

            self._append_edits_from_pickle(program_file, link_to_single=link_to_single)
            return
            


        if not hasattr(self, 'edits'):
            self.edits = {}
            self.all_programs = {}
            self.prompter_class = getattr(prompters, self.method_marker)
            self.prompter = self.prompter_class(MODE, KEY, MODEL, TEMPERATURE, SEED)
            self.prompter.repetitions = self.repetitions
            self.edit_proposer = parallel_prop.ParallelEditProposer()
        
        # Generate the new edit
        chat_box = self._chat_panel.get_children()[0].get_children()[3].get_children()[0]
        new_edit_request = chat_box.text_value
        processed_data, symbolic_data = get_obj(self.selected_obj, redo_search=False, data_dir=self.data_dir, mode="new",
                                                add_ground=True)
        shape = symbolic_data[0]
        shape.clean_up_motion()
        # Run the algorithm
        if 'category' in self.item.keys():
            shape.label = self.item['category']
        start_time = time.time()
        all_edits, log_info, any_breaking, state_info = algo_v4(shape, new_edit_request, self.prompter, self.edit_proposer)
        end_time = time.time()
        n_edits = len(self.edits)
        FREE_VAR = sp.Symbol(f"f_{n_edits}")
        for edit in all_edits:
            edit.amount = edit.amount.subs({MAIN_VAR: FREE_VAR})

        edit_gens = [x.save_format() for x in all_edits]
        # self.edits = call_edit_sys(shape)
        shape = self.symbolic_data[0]
        self.switch_board = update_switch_board(self.name_to_geom, self.label_dict,
                                                self.part_dict, shape)
        self.update_from_switch_board()


        edits, edited_parts = load_edits(shape, edit_gens)
        
        self.edits[f"Edit_{n_edits}"] = (edits, FREE_VAR)
        
        
        shape.clean_up_motion()
        for name, edit_item in self.edits.items():
            edit_list, sym = edit_item
            for edit in edit_list:
                edit.propagate()
        
        for part in shape.partset:
            if len(part.partset) == 0:
                continue
            else:
                # check activity
                core_rel = part.core_relation
                if len(core_rel.edit_sequence) > 0:
                    part.state[0] = PART_ACTIVE
                    for subpart in part.partset:
                        subpart.state[0] = PART_INACTIVE
                else:
                    if len(part.primitive.edit_sequence) > 0:
                        part.state[0] = PART_ACTIVE
                        for subpart in part.partset:
                            subpart.state[0] = PART_INACTIVE
                    else:
                        if any([len(subpart.primitive.edit_sequence) > 0 for subpart in part.partset]):
                            part.state[0] = PART_INACTIVE
                            for subpart in part.partset:
                                subpart.state[0] = PART_ACTIVE
                    
        for part in shape.partset:
            part_edited = len(part.primitive.edit_sequence) > 0
            if part_edited:
                part.state[0] = PART_ACTIVE
            if len(part.sub_parts) > 0:
                edited_children = [len(child.primitive.edit_sequence) > 0 for child in part.sub_parts]
                if any(edited_children):
                    print(f"Part {part.label} has edited children. Turned Off")
                    shape.deactivate_parent(part)
        
        self.update_from_switch_board()
        self.update_procedural_panel()
        self._edit_prebake()
        for ind, edit in enumerate(edits):
            print(ind, edit, get_amount_str(edit.amount))
        chat_box.text_value = ""

    def update_procedural_panel(self):
        self.symbol_value_dict = {}
        if self.edits is None:
            return
        for edit_name, edit_item in self.edits.items():
            edit_list, symbol = edit_item
            self.symbol_value_dict[symbol] = 0.0

        proc_panel = gui.Vert()
        for edit_name, edit_item in self.edits.items():
            edit_list, symbol = edit_item
            title = gui.Horiz()
            title.add_stretch()
            title.add_child(gui.Label(edit_name))
            title.add_stretch()
            proc_panel.add_child(title)
            # symbolic slider.
            sym_value_container = gui.Horiz()
            sym_value_container.add_stretch()
            sym_value = gui.Label(" 0.00")
            # sym_value_container.add_child(sym_value)
            sym_value_container.add_stretch()
            proc_panel.add_child(sym_value_container)
            # now represent mesh in a different way
            # slider for the symbolic value
            symbolic_slider = gui.Slider(gui.Slider.DOUBLE)
            symbolic_slider.set_limits(-0.5, 0.5)
            edit_callback = self.get_edit_callback(symbol)

            self.edit_callbacks[symbol] = edit_callback
            symbolic_slider.set_on_value_changed(edit_callback)
            proc_panel.add_child(symbolic_slider)

            btn_holder = gui.Horiz()
            del_btn = gui.Button("Remove")
            del_btn.set_on_clicked(self.remove_edit(edit_name, symbol))
            btn_holder.add_stretch()
            btn_holder.add_child(del_btn)
            btn_holder.add_stretch()
            proc_panel.add_child(btn_holder)

        save_btn = gui.Button("Save")
        save_btn.set_on_clicked(self.save_model)
        btn_holder = gui.Horiz()
        btn_holder.add_stretch()
        btn_holder.add_child(save_btn)
        btn_holder.add_stretch()

        proc_panel.add_child(btn_holder)
        # procedural slider
        self.proc_inner.set_widget(proc_panel)

    def remove_edit(self, edit_name, symbol):
        def remove_edit():
            for symbol, caller in self.symbol_value_dict.items():
                self.symbol_value_dict[symbol] = 0
                self._edit_execute(symbol)
            self.edits.pop(edit_name)
            self.symbol_value_dict.pop(symbol)

            self.update_from_switch_board()
            self.update_procedural_panel()
            self._edit_prebake()


            
        return remove_edit

    # start the process, and wait for the solver to give the output.
