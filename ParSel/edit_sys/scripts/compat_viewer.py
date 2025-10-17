# Has to load the file, and the two models.

import csv
import os
from pathlib import Path
import _pickle as cPickle
import open3d as o3d
import numpy as np
from collections import defaultdict

SHAPETALK_FILE = "/media/aditya/DATA/data/shapetalk/language/shapetalk_preprocessed_public_version_0.csv"
# all partnet ids
PARTNET_FILE = "/media/aditya/OS/data/partnet/metadata/class_wise_dict.pkl"

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from edit_sys.data_loader.io import load_parts
from edit_sys.data_loader.io import add_mesh_only, add_obb_and_mesh
import sqlite3
from scripts.compat_to_ours import (COMPAT_META_DIR, COMPAT_ZIP_PATH, 
                                    get_config, StylizedShapeLoader)
from scripts.local_config import DATA_DIR, DATA_MODE, BASE_DIR, METADATA_FILE

W = 1024
H = 1024
DATABASE_FILE = "compat.db"

class ShapeTalkApp:

    def __init__(self, data_dir, start_index=None):

        self.window = gui.Application.instance.create_window("ShapeTalk Data", W, H)
        w = self.window

        # member variables
        self.data_dir = data_dir
        self.sql_conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.sql_conn.cursor()
        # create table if not exists
        # Keep 
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS COMPAT (index_id INTEGER PRIMARY KEY, source_anno_id TEXT, valid INTEGER)")
        # If table exits, load the size of the table
        if start_index is None:
            count = self.cursor.execute(f"SELECT COUNT(*) FROM COMPAT")
            count = count.fetchone()[0]
            self.index = count
        else:
            self.index = start_index

        self.show_mesh = True
        self.show_bbox = False
        self.show_labels = True
        self.show_axis = False

        self.em = w.theme.font_size
        em = self.em

        # 3D Widget
        _widget3d = gui.SceneWidget()
        _widget3d.scene = rendering.Open3DScene(w.renderer)
        _widget3d.scene.show_axes(True)
        
        ground_plane = o3d.geometry.TriangleMesh.create_box(
            50.0, 0.1, 50.0, create_uv_map=True, map_texture_to_each_face=True)
        ground_plane.compute_triangle_normals()
        rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi, 0, 0))
        ground_plane.rotate(rotate_180)
        ground_plane.translate((-25.0, -0.1, -25.0))
        ground_plane.paint_uniform_color((1, 1, 1))
        self.ground_plane = o3d.t.geometry.TriangleMesh.from_legacy(ground_plane)

        # ground_plane = o3d.cuda.pybind.visualization.rendering.Scene.GroundPlane(2)
        # _widget3d.scene.show_ground_plane(True, ground_plane)

        # _widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        # create a frame that encapsulates the Scenewidget
        # _widget3d.frame = gui.Rect(0, H//3, W, (H * 2)//3)
        _widget3d.frame = gui.Rect(0, 0, W, (5 * H)// 6)
        _widget3d.scene.set_background([200, 0, 0, 200]) # not working?!
        self._widget3d = _widget3d

        # gui layout
        gui_layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        # create frame that encapsulates the gui
        # gui_layout.frame = gui.Rect(w.content_rect.x, w.content_rect.y, w.content_rect.width, w.content_rect.height//3)
        gui_layout.frame = gui.Rect(w.content_rect.x, (5 * w.content_rect.height) // 6, w.content_rect.width, w.content_rect.height//6)
        # File-chooser widget

        self.utterance_display = gui.WidgetProxy()
        gui_layout.add_child(self.utterance_display)

        high_data_btn = gui.Button("High data")
        high_data_btn.horizontal_padding_em = 0.5
        high_data_btn.vertical_padding_em = 0
        high_data_btn.set_on_clicked(self.accept_data_high)

        med_data_btn = gui.Button(f"Medium data")
        med_data_btn.horizontal_padding_em = 0.5
        med_data_btn.vertical_padding_em = 0
        med_data_btn.set_on_clicked(self.accept_data_medium)
        

        reject_data_btn = gui.Button(f"Reject data")
        reject_data_btn.horizontal_padding_em = 0.5
        reject_data_btn.vertical_padding_em = 0
        reject_data_btn.set_on_clicked(self.reject_data)

        # create a
        model_load_layout = gui.Horiz()
        model_load_layout.add_stretch()
        model_load_layout.add_fixed(0.25 * em)
        model_load_layout.add_child(high_data_btn)
        model_load_layout.add_fixed(0.25 * em)
        model_load_layout.add_child(med_data_btn)
        model_load_layout.add_fixed(0.25 * em)
        model_load_layout.add_child(reject_data_btn)

        gui_layout.add_child(model_load_layout)

        collapse = gui.CollapsableVert("Visualization Settings", 0.5 * em,
                                       gui.Margins(em, 0, 0, 0))
        vis_settings = gui.Horiz()
        cb_1 = gui.Checkbox("Mesh")
        cb_1.set_on_checked(self._toggle_mesh)  # set the callback function
        cb_1.checked = self.show_mesh
        cb_2 = gui.Checkbox("BBox")
        cb_2.set_on_checked(self._toggle_bbox)  # set the callback function
        cb_2.checked = self.show_bbox
        cb_3 = gui.Checkbox("Labels")
        cb_3.set_on_checked(self._toggle_labels)  # set the callback function
        cb_3.checked = self.show_labels
        cb_4 = gui.Checkbox("Axis")
        cb_4.set_on_checked(self._toggle_axis)  # set the callback function
        cb_4.checked = self.show_axis

        vis_settings.add_child(cb_1)
        vis_settings.add_child(cb_2)
        vis_settings.add_child(cb_3)
        vis_settings.add_child(cb_4)
        collapse.add_child(vis_settings)
        gui_layout.add_child(collapse)

        self.relations = gui.WidgetProxy()
        gui_layout.add_child(self.relations)
        
        w.add_child(_widget3d)
        w.add_child(gui_layout)

        # load data

        config = get_config(
            zip_path=COMPAT_ZIP_PATH,
            meta_dir=COMPAT_META_DIR,
            split="train",
            semantic_level="fine"
        )
        if DATA_MODE == "PARTNET":
            info = cPickle.load(open(METADATA_FILE, "rb"))
            # shuffle
            info = info['Chair']
            np.random.shuffle(info)
            self.info = info

            # real_data = []
            # for cur_info in self.info:
            #     anno_id = str(cur_info['anno_id'])
            #     parts, target_category = load_parts(anno_id, self.data_dir)
            #     bulbs = [x for x in parts if 'bulb' in x['label']]
            #     if len(bulbs) > 1:
            #         real_data.append(cur_info)
            # self.info = real_data
            self.add_color = True
        else:
            self.data_loader = StylizedShapeLoader(**config)
            self.add_color = False

        self.load_data()

    def accept_data_high(self):
        # Add data to the database
        item = self.dataset[self.index]
        self.enter_to_db(item, 2)
        self.index += 1
        self.load_data()

    def accept_data_medium(self):
        # Add data to the database
        self.enter_to_db(self.index, 1)
        self.index += 1
        self.load_data()

    def reject_data(self):
        self.enter_to_db(self.index, 0)
        self.index += 1
        self.load_data()
    
    def enter_to_db(self, item, validity):
        # return None
        # overwrite if exists
        self.cursor.execute(f"INSERT OR REPLACE INTO COMPAT (index_id, valid) VALUES (?, ?)", (item, validity))
        self.sql_conn.commit()

    def load_data(self):
        if DATA_MODE == "PARTNET":
            cur_info = self.info[self.index]
            anno_id = str(cur_info['anno_id'])
            print('anno_id', anno_id)
            self.parts, self.name_3ds = self.load_models(anno_id)

        else:
            index = self.index
            save_dir = os.path.join(self.data_dir, f"{index}")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            info_file = os.path.join(save_dir, "result_after_merging.json")
            if not os.path.exists(info_file):
                shape_id, style_id, shape_label, obj = self.data_loader.__getitem__(index)
                self.data_loader.save_to_our_form(index, obj, shape_label, save_dir, allow_split=False)
            str_index = str(index)
            self.parts, self.name_3ds = self.load_models(str_index)
            print(self.index)

    def load_models(self, source_anno_id, plant_to_ground=False):

        # def load_part_to_scene(self, parts, scene):
        # Clean up scene
        self._widget3d.scene.clear_geometry()
        if hasattr(self, "name_3ds"):
            self.remove_labels(self.name_3ds)
        # self._widget3d.scene.add_geometry("ground_plane", self.ground_plane, rendering.MaterialRecord())
        all_meshes = []
        all_parts = []
        all_names = []
        shape_count = 0
        for ind, part_id in enumerate([source_anno_id]):
            
            parts, target_category = load_parts(part_id, self.data_dir)
            
            for part in parts:
                part['label'] = f"{part['label']}_{shape_count}"
            mesh_dir = os.path.join(self.data_dir, str(part_id), "objs")
            # if DATA_MODE == "PARTNET":
            #     parts = add_obb_and_mesh(mesh_dir, parts)
            # else:
            parts = add_mesh_only(mesh_dir, parts)
            
            colors = np.random.uniform(0, 1, (len(parts), 3))
            if self.add_color:
                for i, part in enumerate(parts):
                    # obb = obb_to_mesh(part['obb'], half=False, return_mat=False)
                    # part['obb_mesh'] = obb
                    mesh = part['mesh']
                    for mesh in mesh.meshes:
                        mesh.mesh.paint_uniform_color(colors[i])
                    # part['obb_mesh'].paint_uniform_color(colors[i])

            meshes = [p['mesh'] for p in parts]

            if plant_to_ground:
                shape_mesh = o3d.geometry.TriangleMesh()
                for mesh in meshes:
                    shape_mesh += mesh
                bounds = shape_mesh.get_axis_aligned_bounding_box()
                min_y = bounds.get_min_bound()[1]
                for i, part in enumerate(parts):
                    part['mesh'].translate((0, -min_y, 0))
                    # part['obb'].translate((0, -min_y, 0))
                    # part['obb_mesh'].translate((0, -min_y, 0))

            if self.show_mesh:
                self.add_mesh(parts)
            # if self.show_bbox:
            #     self.add_bbox(parts)
            if self.show_labels:
                name_3ds = self.add_labels(parts)
            else:
                self.remove_labels(self.name_3ds)
                name_3ds = []
            if self.show_axis:
                self.add_axis()
            all_meshes.extend(meshes)
            all_parts.extend(parts)
            all_names.extend(name_3ds)
            shape_count += 1

        mesh = o3d.geometry.TriangleMesh()
        for cur_mesh in all_meshes:
            for mini_mesh in cur_mesh.meshes:
                mesh += mini_mesh.mesh
        bounds = mesh.get_axis_aligned_bounding_box()
        self._widget3d.setup_camera(60, bounds, bounds.get_center())
        return all_parts, all_names

    def remove_mesh(self, parts):
        for ind, part in enumerate(parts):
            label = part['label']
            name = f"{label}_{ind}"
            self._widget3d.scene.remove_geometry(name)
    
    def remove_bbox(self, parts):
        ...
        # for ind, part in enumerate(parts):
        #     label = part['label']
        #     name = f"{label}_{ind}_obb"
        #     self._widget3d.scene.remove_geometry(name)
    
    def remove_labels(self, name_3ds):
        for ind, part in enumerate(name_3ds):
            self._widget3d.remove_3d_label(part)
    
    def add_mesh(self, parts):
        for ind, part in enumerate(parts):
            label = part['label']
            name = f"{label}_{ind}"
            self._widget3d.scene.add_model(name, part['mesh'])
    
    def add_bbox(self, parts):
        ...
        # for ind, part in enumerate(parts):
        #     label = part['label']
        #     name = f"{label}_{ind}_obb"
        #     self._widget3d.scene.add_geometry(name, part['obb_mesh'], rendering.MaterialRecord())

    def add_labels(self, parts):
        name_3ds = []
        for ind, part in enumerate(parts):
            label = part['label']
            name = f"{label}"
            center = np.asarray(part['mesh'].meshes[0].mesh.vertices).mean(axis=0)
            coordinate = center.tolist()
            name_3d = self._widget3d.add_3d_label(coordinate, name)
            name_3ds.append(name_3d)
        return name_3ds


    def add_axis(self):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        self._widget3d.scene.add_geometry("axis", mesh_frame, rendering.MaterialRecord())
    
    def _toggle_mesh(self, is_checked):
        if is_checked:
            # was unchecked, now checked
            self.show_mesh = True
            self.add_mesh(self.parts)
        else:
            # was checked, now unchecked
            self.show_mesh = False
            self.remove_mesh(self.parts)

    def _toggle_bbox(self, is_checked):
        if is_checked:
            # was unchecked, now checked
            self.show_bbox = True
            self.add_bbox(self.parts, self.obbs)
        else:
            # was checked, now unchecked
            self.show_bbox = False
            self.remove_bbox(self.obbs)

    def _toggle_labels(self, is_checked):
        if is_checked:
            # was unchecked, now checked
            self.show_labels = True
            self.name_3ds = self.add_labels(self.parts)
        else:
            # was checked, now unchecked
            self.show_labels = False
            self.remove_labels(self.name_3ds)

    def _toggle_axis(self, is_checked):
        if is_checked:
            # was unchecked, now checked
            self.show_axis = True
            self.add_axis()
        else:
            # was checked, now unchecked
            self.show_axis = False
            self._widget3d.scene.remove_geometry("axis")

if __name__ == "__main__":
    # target_category = "Vase"
    # now load the parts for each
    data_dir = DATA_DIR
    gui.Application.instance.initialize()
    w = ShapeTalkApp(data_dir, start_index=1271)
    gui.Application.instance.run()