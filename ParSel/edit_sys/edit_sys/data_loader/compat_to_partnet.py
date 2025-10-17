""""
Dataloaders for the 3D 3DCoMPaT tasks.
Based on code at https://github.com/Vision-CAIR/3DCoMPaT-v2
"""
import json
import os
import zipfile
import enum
import json
import os

import numpy as np
import trimesh
from pathlib import Path
from .gltf_utils import (load_style_json, load_gltf, load_styles, apply_style, ZipTextureResolver)

def open_meta(meta_dir, file_name):
    json_file = os.path.join(meta_dir, file_name)
    return json.load(open(json_file))

MAX_VERT_COUNT = 10_000
AVOID_KEYS = ["wheel"]
class SemanticLevel(enum.Enum):
    FINE = "fine"
    MEDIUM = "medium"
    COARSE = "coarse"

    def __str__(self):
        return self.value
    
    def get_remap(self, meta_dir):
        if self == self.FINE: return None
        remap = open_meta(meta_dir, 'hier_%s.json' % str(self))
        # Convert hex string keys to int
        return {int(k, 16): v for k, v in remap.items()}

    def get_parts(self, meta_dir):
        return open_meta(meta_dir, 'parts_%s.json' % str(self))


class CompatLoader3D:
    """
    Base class for 3D dataset loaders.

    Args:
    ----
        zip_path:        3DCoMPaT models zip directory.
        meta_dir:        Metadata directory.
        split:           Split to load from, one of {train, valid}.
        semantic_level:  Segmentation level, one of {fine, medium, coarse}.
        n_points:        Number of sampled points.
        load_mesh:       Only load meshes.
        shape_only:      Ignore part segments while sampling pointclouds.
        get_normals:     Also return normal vectors for each sampled point.
        seed:            Initial random seed for pointcloud sampling.
    """

    ZIP_MODELS_DIR = "models/"
    ZIP_STYLES_DIR = "styles/"
    SHAPE_ID_LENGTH = 6

    def __init__(
        self,
        zip_path,
        meta_dir,
        split="train",
        semantic_level="fine",
        n_points=None,
        load_mesh=False,
        shape_only=False,
        get_normals=False,
        seed=None,
        shuffle=False,
    ):
        # Parameters check
        if split not in ["train", "test", "valid"]:
            raise RuntimeError("Invalid split: [%s]." % split)
        if semantic_level not in ["fine", "medium", "coarse"]:
            raise RuntimeError("Invalid semantic level: [%s]." % split)
        if load_mesh and n_points is not None:
            raise RuntimeError("Cannot sample pointclouds in mesh mode.")
        if shape_only and n_points is None:
            raise RuntimeError(
                "shape_only set to true but no number of points are specified."
            )
        if shape_only and load_mesh:
            raise RuntimeError("shape_only set to true alongside load_mesh.")

        self.split = split
        self.semantic_level = SemanticLevel(semantic_level)

        # Opening the zip file
        if not os.path.exists(zip_path):
            raise RuntimeError("Raw models zip not found: [%s]." % zip_path)
        models_zip = os.path.normpath(zip_path)
        self.zip_f = zipfile.ZipFile(models_zip, "r")

        # Opening textures map
        self.textures_map = json.load(self.zip_f.open("textures_map.json", "r"))

        self.text_labels = open_meta(meta_dir, "classes.json")
        # Setting parameters
        self.n_points = n_points
        self.load_mesh = load_mesh
        self.shape_only = shape_only
        self.get_normals = get_normals

        # ====================================================================================

        # Parts index and reversed index
        all_parts = self.semantic_level.get_parts(meta_dir)
        self.parts_to_idx = dict(zip(all_parts, range(len(all_parts))))

        mat_cats = open_meta(meta_dir, "mat_categories.json")
        # Fine material to 8-bits value mapping
        self.mat_to_fine_idx = {}
        for mat_cat in mat_cats:
            for mat in mat_cats[mat_cat]:
                self.mat_to_fine_idx[mat] = mat_cats[mat_cat].index(mat)

        # Coarse material to 8-bits value mapping
        mat_list_coarse = list(mat_cats.keys())
        mat_list_fine = [mat for mat_cat in mat_cats for mat in mat_cats[mat_cat]]
        self.mat_to_coarse_idx = {
            mat: mat_list_coarse.index(mat.split("_")[0]) for mat in mat_list_fine
        }

        # Part remap
        self.part_remap = self.semantic_level.get_remap(meta_dir)

        # ====================================================================================

        # Indexing 3D models
        split_models = open_meta(meta_dir, "split.json")[split]
        self._list_models(split_models)

        # Shuffling indices
        self.shuffle = shuffle
        self.seed = seed

        self.index = -1

    def _get_part_to_mat(self, model_style):
        """
        Get the part to 8-bits coarse and fine material codes mapping for a given model style.
        """
        # Get the part to 8-bits coarse and fine material codes
        parts_to_mat_coarse_idx = {
            p: self.mat_to_coarse_idx[model_style[p]] for p in model_style
        }
        parts_to_mat_fine_idx = {
            p: self.mat_to_fine_idx[model_style[p]] for p in model_style
        }

        return parts_to_mat_coarse_idx, parts_to_mat_fine_idx

    def _get_split_list(self, split_models, shape_list):
        shape_ids = set(split_models) & shape_list
        shape_ids = list(shape_ids)
        shape_ids.sort()
        return shape_ids

    def _get_mesh_map(self, obj, shape_id, shape_label):
        remap_dict = None
        if self.part_remap is not None:
            remap_dict = self.part_remap[shape_label]
        try:
            return pc.map_meshes(obj, self.parts_to_idx, remap_dict)
        except (ValueError, AttributeError):
            raise ("Error while mapping meshes for shape %s" % shape_id) from None

    def _list_zip_dir(self, dir_name):
        k = len(dir_name)
        item_list = set(
            [
                f[k : k + self.SHAPE_ID_LENGTH]
                for f in self.zip_f.namelist()
                if f.startswith(dir_name) and f != dir_name
            ]
        )
        return item_list

    def _list_models(self, split_models):
        """
        Indexing 3D models.
        """
        raise NotImplementedError()

    def __len__(self):
        return len(self.shape_ids)

    def __getitem__(self, index):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.__len__() - 1:
            raise StopIteration
        else:
            self.index += 1
            return self.__getitem__(self.index)


class StylizedShapeLoader(CompatLoader3D):
    """
    Sylized 3D shape loader.

    Args:
    ----
        ...:             See CompatLoader3D.
        n_compositions:  Number of compositions to use.
        get_mats:        Whether to return material labels.
    """

    def __init__(self, n_compositions=1, get_mats=False, **kwargs):
        # Raise error if shape_only is set to True
        if "shape_only" in kwargs and kwargs["shape_only"]:
            raise ValueError("Shape only is not supported for unstylized shapes.")
        self.n_compositions = n_compositions
        self.get_mats = get_mats
        super().__init__(**kwargs)

        # Shuffling indices
        if self.shuffle:
            indices = np.arange(len(self.model_style_ids))
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            self.model_style_ids = [self.model_style_ids[i] for i in indices]
            self.stylized_shape_labels = [
                self.stylized_shape_labels[i] for i in indices
            ]


    def _list_models(self, split_models):
        """
        Indexing 3D models.
        """
        # Opening all JSON style files
        self.model_style_ids = []
        self.stylized_shape_labels = []
        for comp_k in range(self.n_compositions):
            styles = load_style_json(
                split=self.split,
                comp_k=comp_k,
                sem_level=str(self.semantic_level),
                zip_file=self.zip_f,
                styles_dir=self.ZIP_STYLES_DIR,
            )
            shape_style_ids = list(styles.keys())
            new_style_ids = [
                shape_key.split("__") + [comp_k] for shape_key in shape_style_ids
            ]
            self.model_style_ids += new_style_ids

            self.stylized_shape_labels += [
                int(shape_id[:2], 16) for shape_id, _, _ in new_style_ids
            ]
        assert len(self.model_style_ids) == len(self.stylized_shape_labels)

    def __len__(self):
        return len(self.model_style_ids)

    def _get_mesh_map(self, obj, shape_id, shape_label, obj_style):
        remap_dict = None
        parts_to_mat_coarse_idx = self._get_part_to_mat(obj_style)[0]
        if self.part_remap is not None:
            remap_dict = self.part_remap[shape_label]
        try:
            return pc.map_meshes(
                obj,
                self.parts_to_idx,
                remap_dict,
                parts_to_mat_coarse_idx if self.get_mats else None,
            )
        except (ValueError, AttributeError):
            raise ("Error while mapping meshes for shape %s" % shape_id) from None

    def __getitem__(self, index):
        """
        Get raw 3D shape given index.
        """
        # Convert index to shape index and composition index
        shape_id, style_id, comp_k = self.model_style_ids[index]
        shape_label = self.stylized_shape_labels[index]


        gltf_f = load_gltf(
            shape_id, zip_file=self.zip_f, models_dir=self.ZIP_MODELS_DIR
        )
        style_entry = load_styles(
            shape_id=shape_id,
            style_id=style_id,
            split=self.split,
            comp_k=comp_k,
            sem_level=str(self.semantic_level),
            zip_file=self.zip_f,
            styles_dir=self.ZIP_STYLES_DIR,
        )
        shape_part_remap = (
            self.part_remap[shape_label] if self.part_remap is not None else None
        )
        gltf_f = apply_style(
            gltf_f,
            style_entry,
            textures_file_map=self.textures_map,
            shape_part_remap=shape_part_remap,
        )
        obj = trimesh.load(
            gltf_f,
            file_type=".gltf",
            force="scene",
            resolver=ZipTextureResolver(zip_f=self.zip_f),
        )

        shape_text_label = self.text_labels[shape_label]
        # Directly return the trimesh object
        if self.load_mesh:
            return shape_id, style_id, shape_text_label, obj
        # Sample a pointcloud from the 3D shape
        else:
            mesh_map = self._get_mesh_map(obj, shape_id, shape_label, style_entry)
            sample = pc.sample_pointcloud(
                in_mesh=mesh_map,
                n_points=self.n_points,
                sample_color=True,
                shape_only=self.shape_only,
                get_normals=self.get_normals,
                get_mats=self.get_mats,
            )
            return [shape_id, style_id, shape_label] + [s for s in sample]

    def save_to_our_form(self, index, obj, shape_text_label, save_dir, allow_split=False):
        counter = 0
        obj_list = []
        global_children_list = []
        part_save_dir = os.path.join(save_dir, "objs")
        Path(part_save_dir).mkdir(parents=True, exist_ok=True)
        
        for part_name in obj.graph.nodes:
            print("at node ", part_name)
            if part_name == 'world':
                continue
            if 'camera' in part_name:
                continue
            R, mesh_name = obj.graph[part_name]
            if not isinstance(mesh_name, str):
                continue
            # R = obj.graph['world'][0]
            mesh = obj.geometry[mesh_name].copy()
            mesh.apply_transform(R)
            # Process mesh

            mesh_file_name = os.path.join(part_save_dir, f"obj_{counter}.glb")
            with open(mesh_file_name, 'wb') as f:
                mesh.export(f, file_type='glb')
            label = "_".join(part_name.split("_")[:-1])
            n_verts = mesh.vertices.shape[0]
            if allow_split:
                if n_verts < MAX_VERT_COUNT:
                    get_children = True
                    for avoid_key in AVOID_KEYS:
                        if avoid_key in label:
                            get_children = False
                            break
                    if get_children:
                        try:
                            cur_children_list = self.generate_children(mesh, label, counter, part_save_dir)
                        except:
                            print("Failed to generate children")
                            cur_children_list = []
                else:
                    cur_children_list = []
            else:
                cur_children_list = []
            part_dict = {
                'name': label,
                'objs': [f"obj_{counter}.glb",],
                'children': cur_children_list
            }
            global_children_list.append(part_dict)
            obj_list.append(mesh_name)

            counter += 1

        parent = {
            'name': 'bench',
            'objs': obj_list,
            'children': global_children_list
        }
        with open(f"{save_dir}/result_after_merging.json", 'w') as f:
            json.dump([parent], f)
    
    def generate_children(self, mesh, label, counter, part_save_dir):
        # save them all as glb
        # Copy it to 
        mesh_copy = mesh.copy()
        mesh_copy.merge_vertices(merge_tex=True, merge_norm=True)
        cluster = trimesh.graph.connected_component_labels(mesh_copy.face_adjacency)
        n_clusters = np.max(cluster) + 1
        if n_clusters == 1 or n_clusters > 8:
            cur_children_list = []
        else:
            # now get bodies from trimesh mesh
            split_meshes = mesh.split(only_watertight=False, adjacency=mesh_copy.face_adjacency)
            cur_children_list = []
            for index, split_mesh in enumerate(split_meshes):
                # find the closest cluster.
                mesh_file_name = os.path.join(part_save_dir, f"obj_{counter}_{index}.glb")
                with open(mesh_file_name, 'wb') as f:
                    split_mesh.export(f, file_type='glb')
                # only considering one level for now - make it recursive later.
                part_dict = {
                    'name': f"{label}_bar",
                    'objs': [ f"obj_{counter}_{index}.glb"],
                    'children': []
                }
                cur_children_list.append(part_dict)
        return cur_children_list