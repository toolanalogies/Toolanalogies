# Has to load the file, and the two models.

import csv
import os
import pickle
import open3d as o3d
import numpy as np
from collections import defaultdict

SHAPETALK_FILE = "/media/aditya/DATA/data/shapetalk/language/shapetalk_preprocessed_public_version_0.csv"
# all partnet ids
PARTNET_FILE = "/media/aditya/OS/data/partnet/metadata/class_wise_dict.pkl"

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from edit_sys.data_loader.partnet import load_parts, add_obb_and_mesh, obb_to_mesh
import sqlite3

W = 1024
H = 512
DATABASE_FILE = "shapetalk_selection.db"

class ShapeTalkApp:

    def __init__(self, target_category, dataset, data_dir, start_index=None):

        self.window = gui.Application.instance.create_window("ShapeTalk Data", W, H)
        w = self.window

        # member variables
        # set start index
        self.dataset = dataset
        if target_category == "Table":
            target_category = f"{target_category}_an"
        self.target_category = target_category
        self.data_dir = data_dir
        self.sql_conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.sql_conn.cursor()
        # create table if not exists
        # Keep 
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {target_category} (index_id INTEGER PRIMARY KEY, source_anno_id TEXT, target_anno_id TEXT, utterance TEXT, valid INTEGER)")
        # If table exits, load the size of the table
        if start_index is None:
            count = self.cursor.execute(f"SELECT COUNT(*) FROM {target_category}")
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
        self.load_data()

    def accept_data_high(self):
        # Add data to the database
        item = self.dataset[self.index]
        self.enter_to_db(item, 2)
        self.index += 1
        self.load_data()

    def accept_data_medium(self):
        # Add data to the database
        item = self.dataset[self.index]
        self.enter_to_db(item, 1)
        self.index += 1
        self.load_data()

    def reject_data(self):
        item = self.dataset[self.index]
        self.enter_to_db(item, 0)
        self.index += 1
        self.load_data()
    
    def enter_to_db(self, item, validity):
        return None
        index = item['index']
        source_anno_id = item['source_anno_id']
        target_anno_id = item['target_anno_id']
        utterance = item['utterance']
        # overwrite if exists
        self.cursor.execute(f"INSERT OR REPLACE INTO {self.target_category} (index_id, source_anno_id, target_anno_id, utterance, valid) VALUES (?, ?, ?, ?, ?)", (index, source_anno_id, target_anno_id, utterance, validity))
        self.sql_conn.commit()

    def load_data(self):
        index = self.index
        print("index", index)
        item = self.dataset[index]
        utterance = item['utterance']
        source_anno_id = item['source_anno_id']
        target_anno_id = item['target_anno_id']
        self.load_utterance(utterance)
        self.parts, self.name_3ds = self.load_models(source_anno_id, target_anno_id)

    def load_utterance(self, utterance):
        em = self.em * 2
        main = gui.Horiz()
        main.add_child(gui.Label(utterance))
        self.utterance_display.set_widget(main)

    def load_models(self, source_anno_id, target_anno_id, plant_to_ground=True):

        # def load_part_to_scene(self, parts, scene):
        # Clean up scene
        self._widget3d.scene.clear_geometry()
        if hasattr(self, "name_3ds"):
            self.remove_labels(self.name_3ds)
        self._widget3d.scene.add_geometry("ground_plane", self.ground_plane, rendering.MaterialRecord())
        all_meshes = []
        all_parts = []
        all_names = []
        shape_count = 0
        for ind, part_id in enumerate([source_anno_id, target_anno_id]):
            data_dir = os.path.join(self.data_dir, "data_v0")
            parts, category = load_parts(part_id, data_dir)
            for part in parts:
                part['label'] = f"{part['label']}_{shape_count}"
            mesh_dir = os.path.join(self.data_dir, "data_v0", str(part_id), "objs")
            parts = add_obb_and_mesh(mesh_dir, parts)
            if ind == 0:
                translation_vec = np.array([-1, 0, 0])
            else:
                translation_vec = np.array([1, 0, 0])
            colors = np.random.uniform(0, 1, (len(parts), 3))
            for i, part in enumerate(parts):
                part['obb'].translate(translation_vec)
                obb = obb_to_mesh(part['obb'], half=False, return_mat=False)
                part['obb_mesh'] = obb
                part['mesh'].translate(translation_vec)
                part['mesh'].paint_uniform_color(colors[i])
                part['obb_mesh'].paint_uniform_color(colors[i])

            meshes = [p['mesh'] for p in parts]

            if plant_to_ground:
                shape_mesh = o3d.geometry.TriangleMesh()
                for mesh in meshes:
                    shape_mesh += mesh
                bounds = shape_mesh.get_axis_aligned_bounding_box()
                min_y = bounds.get_min_bound()[1]
                for i, part in enumerate(parts):
                    part['mesh'].translate((0, -min_y, 0))
                    part['obb'].translate((0, -min_y, 0))
                    part['obb_mesh'].translate((0, -min_y, 0))

            if self.show_mesh:
                self.add_mesh(parts)
            if self.show_bbox:
                self.add_bbox(parts)
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

        mega_mesh = o3d.geometry.TriangleMesh()
        for mesh in all_meshes:
            mega_mesh += mesh
        bounds = mega_mesh.get_axis_aligned_bounding_box()
        self._widget3d.setup_camera(60, bounds, bounds.get_center())
        return all_parts, all_names

    def remove_mesh(self, parts):
        for ind, part in enumerate(parts):
            label = part['label']
            name = f"{label}_{ind}"
            self._widget3d.scene.remove_geometry(name)
    
    def remove_bbox(self, parts):
        for ind, part in enumerate(parts):
            label = part['label']
            name = f"{label}_{ind}_obb"
            self._widget3d.scene.remove_geometry(name)
    
    def remove_labels(self, name_3ds):
        for ind, part in enumerate(name_3ds):
            self._widget3d.remove_3d_label(part)
    
    def add_mesh(self, parts):
        for ind, part in enumerate(parts):
            label = part['label']
            name = f"{label}_{ind}"
            self._widget3d.scene.add_geometry(name, part['mesh'], rendering.MaterialRecord())
    
    def add_bbox(self, parts):
        for ind, part in enumerate(parts):
            label = part['label']
            name = f"{label}_{ind}_obb"
            self._widget3d.scene.add_geometry(name, part['obb_mesh'], rendering.MaterialRecord())

    def add_labels(self, parts):
        name_3ds = []
        for ind, part in enumerate(parts):
            label = part['label']
            name = f"{label}"
            coordinate = part['obb'].get_center().tolist()
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


def load_valid_shapetalk_utterances(target_category="Chair", both_in_partnet=True, only_hard_context=False, avoid_train=False):
    st_item_list = load_shapetalk_metadata(SHAPETALK_FILE)
    # get subset in partnet
    # relevant_partnet_info, partnet_item_list = load_partnet_metadata(PARTNET_FILE, target_category)
    # relevant_partnet_info, partnet_item_list = load_partnet_metadata(PARTNET_FILE, target_category_)

    if both_in_partnet: 
        min_count = 2
    else:
        min_count = 1

    with open(PARTNET_FILE, 'rb') as f:
        class_wise_dict = pickle.load(f)
    selected_utts = []
    relevant_partnet_info = []
    for target_category_ in class_wise_dict.keys():
        cur_relevant_partnet_info = class_wise_dict[target_category_]
        partnet_item_list = []
        for item in cur_relevant_partnet_info:
            partnet_item_list.append(item['shapenet_model_id'])
        partnet_item_list = set(partnet_item_list)

        st_in_partnet_count = [(x['source_model_name'] in partnet_item_list) + (x['target_model_name'] in partnet_item_list) for x in st_item_list]
        st_in_partnet_count_chairs = [(ind, st_in_partnet_count[ind]) for ind, x in enumerate(st_item_list) if x['source_object_class'] == target_category.lower()]
        cur_selected_utts = [(ind, st_item_list[ind]) for ind, x in st_in_partnet_count_chairs if x == min_count]
        selected_utts.extend(cur_selected_utts)
        if len(selected_utts) > 0:
            relevant_partnet_info.extend(cur_relevant_partnet_info)

    
    # TODO: Correct this.
    if avoid_train:
        selected_utts = [x for x in selected_utts if x[1]['listening_split'] != 'train']
    if only_hard_context:
        selected_utts = [x for x in selected_utts if x[1]['hard_context']== 'True']
    # Also check with the partnete dataset?

    model_id_to_anno_id = defaultdict(list)
    for item in relevant_partnet_info:
        model_id_to_anno_id[item['shapenet_model_id']].append(item['anno_id'])

    final_list = []
    for (ind, item) in selected_utts:
        info = dict()
        info['index'] = ind
        info['utterance'] = item['utterance']
        source_model = item['source_model_name']
        target_model = item['target_model_name']
        # get partnet model_ids
        if source_model in model_id_to_anno_id.keys() and target_model in model_id_to_anno_id.keys():
            info['source_anno_id'] =  str(model_id_to_anno_id[source_model][0])
            info['target_anno_id'] =  str(model_id_to_anno_id[target_model][0])
        else:
            continue
        final_list.append(info)
    return final_list

def load_shapetalk_metadata(file_name):
    my_file = open(file_name, 'r')
    reader = csv.DictReader(my_file)
    st_item_list = list()
    for dictionary in reader:
        st_item_list.append(dictionary)
    return st_item_list

def load_partnet_metadata(file_name, target_category):
    with open(file_name, 'rb') as f:
        class_wise_dict = pickle.load(f)
    partnet_item_list = []
    relevant_partnet_info = class_wise_dict[target_category]
    for item in relevant_partnet_info:
        partnet_item_list.append(item['shapenet_model_id'])
    partnet_item_list = set(partnet_item_list)
    return relevant_partnet_info, partnet_item_list
    
if __name__ == "__main__":
    # target_category = "Table"
    target_category = "Chair"
    # target_category = "Vase"
    if target_category == "Chair":
        only_hard_context = False
    else:
        only_hard_context = False

    dataset = load_valid_shapetalk_utterances(target_category, only_hard_context=only_hard_context)
    # now load the parts for each
    data_dir = "/media/aditya/OS/data/partnet/"
    gui.Application.instance.initialize()
    w = ShapeTalkApp(target_category, dataset, data_dir, start_index=0)
    gui.Application.instance.run()