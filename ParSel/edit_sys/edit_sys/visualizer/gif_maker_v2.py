import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from .settings import Settings
from .gif_maker import GIFCreator
import numpy as np
import sympy as sp
from .base import load_edits
import _pickle as cPickle
from edit_sys.shape_system.constants import PART_ACTIVE
from edit_sys.shape_system.edits import MAIN_VAR
from edit_sys.shape_system.shape_atoms import Part

import os
from pathlib import Path
import base64
from .base import MESH_ITEM, LABEL_ITEM, PC_ITEM, OBB_ITEM
from .utils import create_geom_map, create_switch_board, update_switch_board, update_model
from edit_sys.data_loader.partnet_shape import get_obj

RES = 512

class GifCreatorV2(GIFCreator):

    def __init__(self,
                 edit_request,
                 dataset_index,
                 shape_id,
                 selected_obj,
                 method_marker,
                 data_dir,
                 output_dir,
                 redo_search=False,
                 width=RES,
                 height=RES,):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.method_marker = method_marker
        self.shape_id = shape_id
        self.edit_request = edit_request
        self.dataset_index = dataset_index

        processed_data, symbolic_data = get_obj(selected_obj, redo_search=redo_search,
                                                data_dir=self.data_dir, mode="new", add_ground=True)
        shape = symbolic_data[0]
        ground = shape.get('ground')
        # get all the parts in contact with ground
        self.ground_parts = [x.features[0].primitive.part for x in ground.all_relations()]


        processed_data, symbolic_data = get_obj(selected_obj, redo_search=redo_search,
                                                data_dir=self.data_dir, mode="new", add_ground=False)

        self.part_dict, self.relation_dict, self.intersection_dict, add_color = processed_data
        
        self.name_to_geom, self.label_dict = create_geom_map(*processed_data)
        self.switch_board = create_switch_board(self.name_to_geom, self.label_dict)

        self.switch_board = update_switch_board(self.name_to_geom, self.label_dict,
                                                self.part_dict, symbolic_data[0])


        self.part_dict, self.relation_dict, self.intersection_dict, add_color = processed_data
        self.symbolic_data = symbolic_data
        self.shape = symbolic_data[0]
        self.switch_board = update_switch_board(self.name_to_geom, self.label_dict,
                                                self.part_dict, self.shape)
        
        self._scene = rendering.OffscreenRenderer(width, height)
        self.settings = Settings()
        self._apply_settings()
        # Temporary
        self.add_ground()
        self.reposition()
        self.update_from_switch_board()
        self._scene.scene.scene.set_sun_light([-0.707, 0.0,-0.707], [1.0, 1.0, 1.0], 75000)
        self._scene.scene.scene.set_sun_light([0.707, 0.0,0.707], [1.0, 1.0, 1.0], 750000)
        self._scene.scene.scene.enable_sun_light(True)
        self._scene.scene.set_lighting(self._scene.scene.LightingProfile.DARK_SHADOWS, (-0.577, -0.577, -0.077))
        # Step 1 -> get the renderer to the same quality as base.
        start_gen = True
        if start_gen:
            self.get_edit(link_to_single=True)

        self.initialize_gifer()

        if start_gen:
            
            new_mesh = o3d.geometry.TriangleMesh()
            for name, geom in self.name_to_geom.items():
                item_type = name.split("_")[-1]
                if item_type in ['mesh']:
                    for mesh_info in geom.meshes:
                        new_mesh += mesh_info.mesh
            # Set the max range and export that too
            cur_value = self.range[-1]
            for symbol, caller in self.symbol_value_dict.items():
                self.symbol_value_dict[symbol] = cur_value
                self._edit_execute(symbol)

            for name, geom in self.name_to_geom.items():
                item_type = name.split("_")[-1]
                if item_type in ['mesh']:
                    for mesh_info in geom.meshes:
                        new_mesh += mesh_info.mesh

            cur_value = self.range[0]
            for symbol, caller in self.symbol_value_dict.items():
                self.symbol_value_dict[symbol] = cur_value
                self._edit_execute(symbol)

            # get the center
            self.scene_aabb = new_mesh.get_axis_aligned_bounding_box()
            self.mesh_center = self.scene_aabb.get_center()
            # get AABB
            delta = self.scene_aabb.get_half_extent() * 2
            self.camera_y = self.scene_aabb.get_half_extent()[1] * 1.5
            self.radius = (delta[0] ** 2 + delta[2] ** 2)**0.5
            self.radius = max(self.radius, self.camera_y)
            theta = 0
            camera_x = self.radius * np.sin(theta)
            camera_z = self.radius * np.cos(theta)
            camera_location = self.scene_aabb.get_center() + np.array([camera_x, self.camera_y, camera_z])
            self._scene.scene.camera.look_at(self.mesh_center, camera_location, [0, 1, 0])


            gif_save_dir = os.path.join(self.output_dir, "gifs", self.method_marker, f"{self.dataset_index}")
            Path(gif_save_dir).mkdir(parents=True, exist_ok=True)
            self.gif_save_dir = gif_save_dir

            self.start_creation()

    def start_creation(self):
        iteration = 0
        while(self.old_counter < self.max_count):
            cur_value = self.sym_values[self.old_counter]
            for symbol, caller in self.symbol_value_dict.items():
                self.symbol_value_dict[symbol] = cur_value
                self._edit_execute(symbol)
            theta = self.theta_values[self.old_counter]
            camera_x = self.radius * np.sin(theta)
            camera_z = self.radius * np.cos(theta)
            camera_location = np.array([camera_x, self.camera_y, camera_z])
            self._scene.scene.camera.look_at(self.mesh_center, camera_location, [0, 1, 0])
            file_name = os.path.join(self.gif_save_dir, f"{self.old_counter:04d}.png")
            # self.export_image(file_name, self._scene.frame.width, self._scene.frame.height)
            img = self._scene.render_to_image()
            quality = 9  # png
            o3d.io.write_image(file_name, img, quality)
            self.old_counter += 1
            print(f"saving image {self.old_counter} out of {self.max_count}")
        print("Done!")
        self.html_stuff()
        out_name = os.path.join(self.gif_save_dir, f"{self.dataset_index}.mp4")
        os.system(f"ffmpeg -framerate 30 -i {self.gif_save_dir}/%04d.png -vf scale=512:-1 -vcodec libx264 -crf 25 -pix_fmt yuv420p {out_name} -y")

        os.system(f"rm {self.gif_save_dir}/*.png")
        # clean up ./tmp as well
        os.system(f"rm ./tmp/*")

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])


    def update_from_switch_board(self):

        for name, state in self.switch_board.items():
            if name in self.name_to_geom:
                item_type = MESH_ITEM
            elif name in self.label_dict:
                item_type = LABEL_ITEM
            else:
                item_type = PC_ITEM
            self._toggle_switch(name, not state, item_type)
            self._toggle_switch(name, state, item_type)

    def _toggle_switch(self, name, state, item_type):
        old_state = self.switch_board[name]
        if old_state == state:
            return
        self.switch_board[name] = state
        # load the object
        if item_type == MESH_ITEM:
            if state:
                try:
                    self._scene.scene.add_model(name, self.name_to_geom[name])
                except:
                    self._scene.scene.add_geometry(name, self.name_to_geom[name], self.settings.material)
            else:
                self._scene.scene.remove_geometry(name)
                # self._scene.scene.remove_model(name)

    def update_procedural_panel(self):
        self.symbol_value_dict = {}
        if self.edits is None:
            return
        proc_panel = gui.Vert()
        for edit_name, edit_item in self.edits.items():
            edit_list, symbol = edit_item
            self.symbol_value_dict[symbol] = 0.0
            