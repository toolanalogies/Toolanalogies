# visualization for each step
# now view them
import open3d as o3d
import numpy as np
import networkx as nx
from .utils import obb_to_mesh

def show_all_parts_and_box(parts, selected_ids=None):
    colors = np.random.uniform(0, 1, (len(parts), 3))
    meshes = [p['mesh'] for p in parts]
    obbs = [obb_to_mesh(p['obb'], half=True) for p in parts]
    for i, mesh in enumerate(meshes):
        mesh.paint_uniform_color(colors[i])
        obbs[i].paint_uniform_color(colors[i])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
    origin = [-1, -1, -1]
    mesh_frame.translate(origin)
    if selected_ids is not None:
        meshes = [meshes[i] for i in selected_ids]
        obbs = [obbs[i] for i in selected_ids]
    o3d.visualization.draw_geometries([mesh_frame,] + obbs + meshes, mesh_show_back_face=True, mesh_show_wireframe=True)

# from label_to_sel_syms, get the symmetries
def get_readable_annotations(parts, clean_relation_set, intersections_indices):
    sym_name_map = {
        "T": "Translation",
        "R": "Rotation",
        "REF": "Reflection",
    }
    order_name_map = {
        0: "Point",
        1: "Line",
        2: "Face",
        3: "Volume",
    }
    sym_list = []

    for sel_syms in clean_relation_set:
        indices, params, sym_type = sel_syms
        part_0 = parts[indices[0]]
        if part_0['hier'] > 0:
            if sym_type == "REF":
                if len(clean_relation_set) > 20:
                    continue
        sym_type = sym_name_map[sym_type]
        part_labels = []
        for item in indices:
            index_a = item
            label = parts[index_a]['label']
            part_labels.append(f"{label}")
        part_labels = set(part_labels)
        part_labels = ", ".join(part_labels)
        sym_list.append(f"{part_labels} are {sym_type} symmetric with params: {params}")
    contact_list = []

    for intersection_item in intersections_indices:
        index_a, index_b, _, order = intersection_item
        if parts[index_a]['hier'] > 0 or parts[index_b]['hier'] > 0:
            if len(intersections_indices) > 30:
                continue
        label_a = parts[index_a]['label']
        label_b = parts[index_b]['label']
        contact_items = f"{label_a} and {label_b}"
        order_name = order_name_map[order]
        contact_list.append(f"{contact_items} have a {order_name} contact")
    return sym_list, contact_list

def get_params(sel_sym, sym_type):
    if sym_type == "Translation":
        graph = nx.Graph()
        for item in sel_sym:
            # use smallest path between 1 and 2 always
            ind_1 = item[0]
            ind_2 = item[1]
            delta = item[2]
            amount = np.linalg.norm(delta)
            if not ind_1 in graph.nodes():
                graph.add_node(ind_1)
            if not ind_2 in graph.nodes():
                graph.add_node(ind_2)
            if not graph.has_edge(ind_1, ind_2):
                graph.add_edge(ind_1, ind_2)
                graph[ind_1][ind_2]["delta"] = delta
                graph[ind_1][ind_2]["amount"] = amount
            else:
                # replace the edit content if the angle is smaller
                if amount < graph[ind_1][ind_2]["amount"]:
                    graph[ind_1][ind_2]["delta"] = delta
                    graph[ind_1][ind_2]["amount"] = amount
        T = nx.minimum_spanning_tree(graph, weight='amount')
        edge = list(T.edges(data=True))[0]
        delta = edge[2]["delta"]
        param = f"Delta: {delta}"
    elif sym_type == "Rotation":
        graph = nx.Graph()
        for item in sel_sym:
            # use smallest path between 1 and 2 always
            ind_1 = item[0]
            ind_2 = item[1]
            axis = item[2]
            origin = item[3]
            angle = np.linalg.norm(axis)
            axis = axis / angle
            if not ind_1 in graph.nodes():
                graph.add_node(ind_1)
            if not ind_2 in graph.nodes():
                graph.add_node(ind_2)
            if not graph.has_edge(ind_1, ind_2):
                graph.add_edge(ind_1, ind_2)
                graph[ind_1][ind_2]["axis"] = axis
                graph[ind_1][ind_2]["origin"] = origin
                graph[ind_1][ind_2]["angle"] = angle
            else:
                # replace the edit content if the angle is smaller
                if angle < graph[ind_1][ind_2]["angle"]:
                    graph[ind_1][ind_2]["axis"] = axis
                    graph[ind_1][ind_2]["origin"] = origin
                    graph[ind_1][ind_2]["angle"] = angle
        T = nx.minimum_spanning_tree(graph, weight='angle')
        edge = list(T.edges(data=True))[0]
        axis = edge[2]["axis"]
        angle = edge[2]["angle"]
        origin = edge[2]["origin"]
        param = f"Axis: {axis}, Angle: {angle}, Origin: {origin}"
    elif sym_type == "Reflection":
        for item in sel_sym:
            ind_1 = item[0]
            ind_2 = item[1]
            plane_normal = item[2]
            plane_origin = item[3]
            break
        param = f"Normal: {plane_normal}, Origin: {plane_origin}"
    else:
        raise NotImplementedError
    return param