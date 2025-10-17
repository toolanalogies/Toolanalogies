import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from .settings import Settings
import numpy as np
import base64
from .base import MESH_ITEM, LABEL_ITEM, PC_ITEM, OBB_ITEM
from .utils import create_geom_map, create_switch_board, update_switch_board, update_model

RES = 512

class ShapeRenderer:
    def __init__(self, processed_data, symbolic_data, save_name):

        self.part_dict, self.relation_dict, self.intersection_dict, add_color = processed_data
        self.save_name = save_name
        self.name_to_geom, self.label_dict = create_geom_map(*processed_data)
        self.switch_board = create_switch_board(self.name_to_geom, self.label_dict)

        self.shape = symbolic_data[0]
        self.switch_board = update_switch_board(self.name_to_geom, self.label_dict,
                                                self.part_dict, self.shape)
        self._scene = rendering.OffscreenRenderer(RES, RES)
        self.settings = Settings()
        self._apply_settings()
        # Temporary
        self.add_ground()
        # Move mesh above:
        self.reposition()
        self.update_from_switch_board()
        self._scene.scene.scene.set_sun_light([-0.707, 0.0,-0.707], [1.0, 1.0, 1.0], 75000)
        self._scene.scene.scene.set_sun_light([0.707, 0.0,0.707], [1.0, 1.0, 1.0], 750000)
        self._scene.scene.scene.enable_sun_light(True)
        # self._scene.enable_light_shadow(True)
        self._scene.scene.set_lighting(self._scene.scene.LightingProfile.DARK_SHADOWS, (-0.577, -0.577, -0.077))
        # Step 1 -> get the renderer to the same quality as base.

    def generate_image(self):
        img = self._scene.render_to_image()
        o3d.io.write_image(self.save_name, img, 9)
        en_img = encode_image(self.save_name)
        return en_img
    def reposition(self):

        mega_mesh = o3d.geometry.TriangleMesh()
        for part in self.shape.partset:
            if part.label == "ground":
                continue
            mesh_name = f"{part.part_index}_mesh"
            mesh = self.name_to_geom[mesh_name]
            for mesh_info in mesh.meshes:
                mega_mesh += mesh_info.mesh
        aabb = mega_mesh.get_axis_aligned_bounding_box()
        min_y = aabb.get_min_bound()[1]
        for part in self.shape.partset:
            if part.label == "ground":
                continue
            mesh_name = f"{part.part_index}_mesh"
            mesh = self.name_to_geom[mesh_name]
            for mesh_info in mesh.meshes:
                mesh_info.mesh.translate((0, -min_y, 0))
        # set camera according to the aabb
        # get extent
        center = aabb.get_center()
        extent = aabb.get_extent()
        extent[0] *= 1.5
        extent[1] *= 1.0
        extent[2] *= 1.5
        center[1] -= min_y
        self._scene.scene.camera.look_at(center, extent, [0, 1, 0])

            

    def add_ground(self):

        ground_plane = o3d.geometry.TriangleMesh.create_box(
            50.0, 0.1, 50.0, create_uv_map=True, map_texture_to_each_face=True)
        ground_plane.compute_triangle_normals()
        rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi, 0, 0))
        ground_plane.rotate(rotate_180)
        ground_plane.translate((-25.0, -0.1, -25.0))
        ground_plane.paint_uniform_color((0.25, 0.5, 0.25))
        self.ground_plane = o3d.t.geometry.TriangleMesh.from_legacy(ground_plane)
        self._scene.scene.add_geometry("ground_plane", self.ground_plane, self.settings._materials['defaultLit'])




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



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')