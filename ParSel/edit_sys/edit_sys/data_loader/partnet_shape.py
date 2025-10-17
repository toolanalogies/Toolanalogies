import itertools
import time
import os
import numpy as np

import _pickle as cPickle
import open3d as o3d
import copy
import networkx as nx
from collections import defaultdict

from .io import enrich_data, get_save_format
from .utils import (obb_to_corner_points, get_oriented_bounding_box_with_fixing, BASE_CORNERS)
from .partnet import main_loader
from edit_sys.shape_system.shape_atoms import Hexahedron, PrimitiveCuboid, Part
from edit_sys.shape_system.shape_atoms import PointFeature
from edit_sys.shape_system.relations import (PointContact, LineContact, FaceContact, VolumeContact, HeightContact,
                                             ReflectionSymmetry, RotationSymmetry, TranslationSymmetry, 
                                             FeatureRelation, MatchingRotationGroup)
from edit_sys.shape_system.geometric_atoms import Line, Plane
from edit_sys.shape_system.constants import PART_ACTIVE, RELATION_ACTIVE, RELATION_INACTIVE, PART_INACTIVE


class HackyOBB:
    def __init__(self, obb):

        self.R = obb.R
        self.extent = obb.extent
        self.center = obb.get_center()
    def get_center(self):
        return self.center
X_AXIS = np.array([1, 0, 0])
Y_AXIS = np.array([0, 1, 0])
Z_AXIS = np.array([0, 0, 1])

def convert_to_our_part_format(part, ind, mode):
    obb = part["obb"]
    corner_points = obb_to_corner_points(obb)
    # center = obb.get_center().astype(np.float32)
    # size = obb.extent.astype(np.float32)
    # axis = obb.R.astype(np.float32)

    primitive = Hexahedron(corner_points, HackyOBB(obb))
    if len(part['children']) > 0:
        has_children = 1
    else:
        has_children = 0
    format_part = Part(part["label"], primitive, part_index=ind, has_children=has_children,
                       original_label=part['original_label'], mode=mode)
    if part['hier'] > 0:
        format_part.state[0] = PART_INACTIVE
    return format_part

def create_base_model(model, mode="new", category="Chair"):
    part_dict = {}
    for ind, part in model.items():
        format_part = convert_to_our_part_format(part, ind, mode)
        part_dict[ind] = format_part
    
    for ind, part in model.items():
        for child_ind in part['children']:
            part_dict[ind].sub_parts.add(part_dict[child_ind])
            part_dict[child_ind].parent = part_dict[ind]

    
    # center = np.array([0, 0, 0])
    # size = np.array([1, 1, 1])
    # axis = np.eye(3)
    # bounding_prim = PrimitiveCuboid(center, size, axis)
    bounding_prim = Hexahedron(BASE_CORNERS)
    obj = Part(category, bounding_prim, mode=mode)
    for ind, part in part_dict.items():
        if model[ind]['hier'] == 0:
            obj.sub_parts.add(part)
    
    for part in obj.partset:
        part.primitive.create_interesting_points()

    return obj, part_dict


def derive_symmetry_relations(parts, part_dict, clean_relations):
    sym_relations = []
    for relation_ind, relation in clean_relations.items():
        indices = relation['indices']
        params = relation['params']
        sym_group_type = relation['type']
        # compressed_indices += indices
        sel_parts  = [part_dict[ind].primitive for ind in indices]
        instance_part = part_dict[indices[0]]
        if sym_group_type in "R":
            parent_ind = [ind for ind in parts if indices[0] in parts[ind]['children']][0]
            parent_part = part_dict[parent_ind]
            # Get the sym group.
            axis, origin = params
            angle = np.linalg.norm(axis)
            axis = axis / angle
            axis = Line(origin, axis)
            relation = RotationSymmetry(sel_parts, axis, angle, parent_part, relation_index=relation_ind)
            
        elif sym_group_type in "T":
            parent_ind = [ind for ind in parts if indices[0] in parts[ind]['children']][0]
            parent_part = part_dict[parent_ind]
            delta = params[0]

            relation = TranslationSymmetry(sel_parts, delta, parent_part, relation_index=relation_ind)
        elif sym_group_type in "REF":
            plane_normal, plane_origin = params
            plane = Plane(plane_origin, plane_normal)
            relation = ReflectionSymmetry(sel_parts, plane, relation_index=relation_ind)
        sym_relations.append(relation)
    return sym_relations


def derive_point_contacts(parts, part_dict, intersections, n_syms):
    intersection_relations = []
    for relation_ind, contact_set in intersections.items():
        id_1, id_2 = contact_set['indices']
        points = contact_set['points']
        order = contact_set['order']
        point_feature_1 = []
        point_feature_2 = []
        # initially set to inactive if hier > 0
        part_1_hier = parts[id_1]['hier']
        part_2_hier = parts[id_2]['hier']
        hier = max(part_1_hier, part_2_hier)
        # has_child = max(part_dict[id_1].state[2], part_dict[id_2].state[2])
        if hier > 0:
            state = RELATION_INACTIVE
        else:
            state = RELATION_ACTIVE
        for point in points:
            # w.r.t. obj 1
            part_our_format = part_dict[id_1]
            obb = parts[id_1]['obb']
            # save coords as the volume based coords
            local_coord = global_to_local(point, obb)
            cube_coord = local_to_cube(local_coord)

            primitive = part_our_format.primitive
            point_1 = PointFeature(primitive, cube_coord)
            point_feature_1.append(point_1)
            # w.r.t. obj 2.
            part_our_format = part_dict[id_2]
            obb = parts[id_2]['obb']
            local_coord = global_to_local(point, obb)
            cube_coord = local_to_cube(local_coord)
            primitive = part_our_format.primitive
            point_2 = PointFeature(primitive, cube_coord)
            point_feature_2.append(point_2)
        if order == 0:
            relation = PointContact(point_feature_1 + point_feature_2, relation_index=relation_ind+n_syms)
        elif order == 1:
            relation = LineContact(point_feature_1 + point_feature_2, relation_index=relation_ind+n_syms)
        elif order == 2:
            relation = FaceContact(point_feature_1 + point_feature_2, relation_index=relation_ind+n_syms)
        elif order == 3:
            relation = VolumeContact(point_feature_1 + point_feature_2, relation_index=relation_ind+n_syms)
        relation.state[0] = state
        intersection_relations.append(relation)
        # point1 and point2 are 3D points in the local coordinate system of the part
        # label1 and label2 are the labels of the parts that intersect
    return intersection_relations

# Non homo coords
def global_to_local(point, obb):
    local_coord = ((point - obb.get_center()) @ obb.R) / (obb.extent/2)
    if np.max(np.abs(local_coord)) > 0.99:
        print("what?")
    return local_coord

def local_to_global(point, obb):
    global_coord = obb.R @ (point * (obb.extent/2)) + obb.get_center()
    return global_coord

# go from local coordinates to coordinates w.r.t. the cube
def local_to_cube(local_coords):
    # for 0
    opp_coords = np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1],
    ])
    opp_coords = opp_coords[None, :]
    volumes = []
    for i in range(8):
        opp_coord = opp_coords[:, i]
        diff = local_coords - opp_coord
        volume = np.prod(np.abs(diff), axis=1)
        volumes.append(volume)
    volumes = np.array(volumes).T
    volumes = volumes / 8.
    return volumes

def correct_translations(relations):
    cleaned_relations = []

    for relation in relations:
        sym_group_type = relation[2]
        if sym_group_type in "T":
            indices = relation[0]
            params = relation[1]
            delta = params[0]
            delta_norm = np.linalg.norm(delta)
            delta_normed = delta / delta_norm
            prod_x = np.dot(delta_normed, X_AXIS)
            prod_y = np.dot(delta_normed, Y_AXIS)
            prod_z = np.dot(delta_normed, Z_AXIS)
            max_abs_prod_ind = np.argmax(np.abs([prod_x, prod_y, prod_z]))
            max_prod = [prod_x, prod_y, prod_z][max_abs_prod_ind]
            if max_prod < 0:
                delta = -delta
                indices = indices[::-1]
            relation = (indices, [delta], sym_group_type)
        cleaned_relations.append(relation)
    return cleaned_relations

def get_obj(selected_obj_file, data_dir, redo_search=False, mode="new", add_ground=False):
    model_id = selected_obj_file.split("/")[-1].split(".")[0]

    if not os.path.exists(selected_obj_file):
        redo_search = True
    if redo_search:
        model, clean_relations, intersections, category = main_loader(model_id, data_dir)
        clean_relations = correct_translations(clean_relations)
        model = get_save_format(model)
        items = (model, clean_relations, intersections, category)
        cPickle.dump(items, open(selected_obj_file, "wb"))
    else:
        try:
            model, clean_relations, intersections, category = cPickle.load(open(selected_obj_file, "rb"))
        except:

            model, clean_relations, intersections, category = main_loader(model_id, data_dir)
            model = get_save_format(model)
            items = (model, clean_relations, intersections, category)
            cPickle.dump(items, open(selected_obj_file, "wb"))
            model, clean_relations, intersections, category = cPickle.load(open(selected_obj_file, "rb"))
    if 'compat' in selected_obj_file:
        loader_mode = 'compat'
    else:
        loader_mode = 'partnet'
    model, clean_relations, intersections, add_color = enrich_data(model_id, model, clean_relations, intersections, 
                                                                   data_dir=data_dir, mode=loader_mode)
    # create the part hier
    shape, part_dict = create_base_model(model, mode=mode, category=category)
    # now for reach relation, we need to record the conversion from 0th item to nth item.
    shape_syms = derive_symmetry_relations(model, part_dict, clean_relations)
    n_syms = len(shape_syms)
    shape_intersections = derive_point_contacts(model, part_dict, intersections, n_syms)
    vis_set = [model, clean_relations, intersections, add_color]
    if add_ground:
        shape = add_world_support(shape, model, mode="ground")
    # optional
    # shape = add_rot_matcher(shape, model, RotationSymmetry, match_mode="param")
    # shape = add_rot_matcher(shape, model, TranslationSymmetry)
    symbolic_set = [shape, shape_syms, shape_intersections]

    return vis_set, symbolic_set

def add_rot_matcher(shape, parts, relation_type, match_mode="contact"):

    rotational_relations = defaultdict(list)

    for relation in shape.all_relations():
        if isinstance(relation, relation_type):
            n_children = len(relation.primitives)
            rotational_relations[n_children].append(relation)
    
    intersection_relations = [x for x in shape.all_relations() if isinstance(x, FeatureRelation)]
    intersection_indices = [x.uid for x in intersection_relations]
    # create a graph
    max_relation_index = max([x.relation_index for x in intersection_relations])
    if match_mode == "contact":
        graph = nx.Graph()
        graph.add_edges_from(intersection_indices)
        for n_children, relations in rotational_relations.items():
            # check if these are in a contact sequence. 
            relation_indices = [x.parent_part.part_index for x in relations]
            # if they are, then we can add a rotational matcher.
            subgraph = graph.subgraph(relation_indices)
            # get connected componenets
            connected_components = list(nx.connected_components(subgraph))
            for component in connected_components:
                if len(component) > 1:
                    # get the relations
                    cur_relations = [x for x in relations if x.parent_part.part_index in component]
                    matching_relation = MatchingRotationGroup(cur_relations, max_relation_index+1)
                    max_relation_index += 1
    elif match_mode == "param":

        for n_children, relations in rotational_relations.items():
            # check if these are in a contact sequence. 
            subgraph = nx.Graph()
            relation_indices = [x.parent_part.part_index for x in relations]
            subgraph.add_nodes_from(relation_indices)
            for ind_1, ind_2 in itertools.combinations(range(len(relation_indices)), 2):
                r1_axis = relations[ind_1].axis
                r2_axis = relations[ind_2].axis
                if r1_axis == r2_axis:
                    subgraph.add_edge(relation_indices[ind_1], relation_indices[ind_2])
            # get connected componenets
            connected_components = list(nx.connected_components(subgraph))
            for component in connected_components:
                if len(component) > 1:
                    # get the relations
                    cur_relations = [x for x in relations if x.parent_part.part_index in component]
                    matching_relation = MatchingRotationGroup(cur_relations, max_relation_index+1)
                    max_relation_index += 1
    return shape
def add_world_support(shape, parts, mode="ground"):
    if mode == "ground":
        face = "down"
    # get parts which are at the bottom. How? Down face center, and within some delta. 
        bottom_face_centers = [(x.part_index, x.face_center(face)) for x in shape.partset if x.state[0] == PART_ACTIVE]
        # get the min z value
        min_z = np.min([x[1][1] for x in bottom_face_centers])
        # get the parts which are at the bottom
        HEIGHT_DELTA = 0.1

        all_bottom_face_centers = [(x.part_index, x.face_center(face)) for x in shape.partset]
        # Now we will not just add all, get the lowest one first.
        bottom_part_inds = [x[0] for x in all_bottom_face_centers if x[1][1] < min_z + HEIGHT_DELTA]
        bottom_parts = []
        for part_ind in bottom_part_inds:
            part = [x for x in shape.partset if x.part_index == part_ind][0]
            bottom_parts.append(part)

        bottom_parts.sort(key=lambda x: x.face_center(face)[1])
        bottom_most = bottom_parts[0]
        other_valid = [x for x in bottom_parts if x.original_label == bottom_most.original_label]
        bottom_parts = other_valid + [bottom_most]
        if hasattr(bottom_most, "parent") and bottom_most.parent is not None:
            bottom_parts.append(bottom_most.parent)
        bottom_parts = list(set(bottom_parts))
        selected_indices = [x.part_index for x in bottom_parts]
    else:
        face = "up"
        top_face_centers = [(x.part_index, x.face_center(face)) for x in shape.partset if x.state[0] == PART_ACTIVE]
        # get the min z value
        max_z = np.max([x[1][1] for x in top_face_centers])
        # get the parts which are at the bottom
        HEIGHT_DELTA = 0.1
        all_top_face_centers = [(x.part_index, x.face_center(face)) for x in shape.partset]

        top_parts_inds = [x[0] for x in all_top_face_centers if x[1][1] > min_z - HEIGHT_DELTA]
        top_parts = []
        for part_ind in top_parts_inds:
            part = [x for x in shape.partset if x.part_index == part_ind][0]
            top_parts.append(part)

        top_parts.sort(key=lambda x: -x.face_center(face)[1])
        top_most = top_parts[0]
        other_valid = [x for x in top_parts if x.original_label == top_most.original_label]
        top_parts = other_valid + [top_most]
        if hasattr(top_most, "parent"):
            top_parts.append(top_most.parent)
        top_parts = list(set(top_parts))
        selected_indices = [x.part_index for x in top_parts]
    corner_points = BASE_CORNERS.copy() # obb_to_corner_points(obb)
    corner_points = corner_points.astype(np.float32)
    bottom_faces = [0, 1, 4, 5]
    top_faces = [2, 3, 6, 7]
    for face_ind in bottom_faces:
        corner_points[face_ind, 1] = min_z - HEIGHT_DELTA
    for face_ind in top_faces:
        corner_points[face_ind, 1] = min_z + HEIGHT_DELTA
    primitive = Hexahedron(corner_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(corner_points)
    ground_obb = get_oriented_bounding_box_with_fixing(pcd)

    label = mode
    ground_part = Part(label, primitive, 
                       part_index=len(shape.partset), 
                       original_label=label,
                       has_children=0, mode="new")
    ground_part.primitive.create_interesting_points()
    shape.sub_parts.add(ground_part)
    # Add point contact
    relation_ind = max([x.relation_index for x in shape.all_relations(only_active=False)]) + 1
    for part_ind in selected_indices:
        point_feature_1 = []
        point_feature_2 = []
        part = [x for x in shape.partset if x.part_index == part_ind][0]
        point = part.face_center(face)
        point = np.array(point).astype(np.float32)
        # derive this from the primitive
        obb = parts[part_ind]['obb']
        local_coord = global_to_local(point, obb)

        cube_coord = local_to_cube(local_coord)

        primitive = part.primitive
        point_1 = PointFeature(primitive, cube_coord)
        point_feature_1.append(point_1)

        local_coord = global_to_local(point, ground_obb)

        cube_coord = local_to_cube(local_coord)
        primitive = ground_part.primitive
        point_2 = PointFeature(primitive, cube_coord)
        point_feature_2.append(point_2)
        relation = HeightContact(point_feature_1 + point_feature_2, relation_index=relation_ind)
        relation_ind += 1
    return shape
