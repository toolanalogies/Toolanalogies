
# 1) Identify if the relation is broken.
# 2) If broken, identify the edit that broke it.
# 3) If the edit is not in the list of all edits, then the relation is broken.



from collections import defaultdict
import numpy as np
import sympy as sp
from string import Template
import itertools
from .constants import MOVE_LIMIT, TRIALS, UPPER_BOUND, NUMERIC_EVAL
from .constants import QUANT_COMPARE_THRESHOLD_MODE_1, QUANT_COMPARE_THRESHOLD_MODE_2
from .constants import RELATION_UNCHECKED, RELATION_RETAINED, PART_UNEDITED

DIRECTIONS = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1)
]
DIR_TO_NAME = {
    (1, 0, 0): "right",
    (-1, 0, 0): "left",
    (0, 1, 0): "up",
    (0, -1, 0): "down",
    (0, 0, 1): "front",
    (0, 0, -1): "back"
}

def always_real(expr):
    symbols = list(expr.free_symbols)
    values = np.random.uniform(size=(TRIALS, len(symbols))) * MOVE_LIMIT
    # Substitute the values into the expressions
    evaluations = []
    real_bool = True
    for trial_ind in range(TRIALS):    
        expr_val = expr.subs({symbols[i]: values[trial_ind, i] for i in range(len(symbols))})
        if expr_val.is_real:
            # evaluations.append(np.abs(float(expr_val)))
            evaluations.append(expr_val)
        else:
            real_bool = False
            break
    return real_bool

def always_within_bounds(expr):

    symbols = list(expr.free_symbols)
    values = np.random.uniform(size=(TRIALS, len(symbols))) * MOVE_LIMIT
    # Substitute the values into the expressions
    evaluations = []
    within_bound_bool = True
    for trial_ind in range(TRIALS):    
        expr_val = expr.subs({symbols[i]: values[trial_ind, i] for i in range(len(symbols))})
        if sp.Abs(expr_val) < UPPER_BOUND:
            evaluations.append(np.abs(float(expr_val)))
        else:
            within_bound_bool = False
            break
    return within_bound_bool

def evaluate_equals_zero(delta_expr, order=1, mode=1, value=None):
    # if true_quant is not None:
    #     true_quant = QUANT_COMPARE_THRESHOLD
    if mode == 1:
        true_quant = QUANT_COMPARE_THRESHOLD_MODE_1
    elif mode == 2:
        true_quant = QUANT_COMPARE_THRESHOLD_MODE_2
    elif mode == 3:
        true_quant = value * QUANT_COMPARE_THRESHOLD_MODE_1
    if order != 1:
        true_quant = ((true_quant ** 2) /order) ** (1/2)

    if isinstance(delta_expr, (sp.Float, sp.Integer)):
        if np.abs(float(delta_expr)) < true_quant:
            equal = True
        else:
            equal = False
    elif isinstance(delta_expr, sp.Expr):

        equal = False # delta_expr.equals(0)
        if not equal and NUMERIC_EVAL:
            symbols = list(delta_expr.free_symbols)
            if not len(symbols) > 0:
                raise ValueError("At least one expression must be symbolic.")
            # Create a random vector of values
            # delta = (2 * MOVE_LIMIT) / TRIALS
            # values = np.arange(-MOVE_LIMIT, MOVE_LIMIT + delta, delta)
            # values = np.repeat(values, len(symbols)).reshape(len(values), len(symbols))
            # # random shuffle
            # np.random.shuffle(values)
            values = np.random.uniform(size=(TRIALS, len(symbols))) * MOVE_LIMIT
            boundary_values = [0, MOVE_LIMIT]
            # get all combinations of boundary values
            boundary_combinations = list(itertools.product(boundary_values, repeat=len(symbols)))
            # add the boundary values to the values
            values = np.vstack([values, boundary_combinations])
            


            # Substitute the values into the expressions
            evaluations = []
            equal = True
            for trial_ind in range(TRIALS):    
                expr_val = delta_expr.subs({symbols[i]: values[trial_ind, i] for i in range(len(symbols))})
                if expr_val.is_real:
                    value = np.abs(float(expr_val))
                    evaluations.append(value)
                    if value > true_quant:
                        equal = False
                        break
                else:
                    has_imaginary = True
                    equal = False
                    break

    else:
        if np.abs(float(delta_expr)) < true_quant:
            equal = True
        else:
            equal = False
    return equal

def evaluate_intersection(set_1, set_2):
    final_set = set()
    for item_1 in set_1:
        for item_2 in set_2:
            diff_eq = item_1 - item_2
            if evaluate_equals_zero(diff_eq):
                final_set.add(item_1)
                break
    return final_set


def numeric_limited_eval(expr):
    symbols = list(expr.free_symbols)
    if not len(symbols) > 0:
        raise ValueError("At least one expression must be symbolic.")
    # Create a random vector of values
    values = np.random.uniform(size=(TRIALS, len(symbols))) * MOVE_LIMIT
    # Substitute the values into the expressions
    eval_list = []
    for trial_ind in range(TRIALS):    
        expr_val = expr.subs({symbols[i]: values[trial_ind, i] for i in range(len(symbols))})
        eval_list.append(expr_val)
    return eval_list

def rotation_matrix_sympy(axis, angle):
    """
    Create a rotation matrix for a given axis and angle using Rodrigues' rotation formula.
    
    Parameters:
    axis (sp.Matrix): A sympy Matrix of shape (3, 1) representing the axis of rotation.
    angle (float or sp.Expr): The angle of rotation (in radians).
    
    Returns:
    sp.Matrix: A sympy Matrix representing the rotation matrix.
    """
    
    # Normalize the axis to make sure it's a unit vector
    axis = axis / sp.sqrt(axis.dot(axis))
    
    # Extract components of the axis
    u_x, u_y, u_z = axis
    
    # Define the cross-product matrix for the axis
    K = sp.Matrix([
        [0, -u_z, u_y],
        [u_z, 0, -u_x],
        [-u_y, u_x, 0]
    ])
    
    # Compute the rotation matrix using Rodrigues' rotation formula
    R = sp.eye(3) + sp.sin(angle) * K + (1 - sp.cos(angle)) * K**2
    R.simplify()
    return R

def create_cuboid_vertices_sympy(center, size, axis):
    """
    Create the vertices of a 3D cuboid using sympy.

    Parameters:
    center (Matrix): The center of the cuboid (1x3).
    size (Matrix): The size of the cuboid along each axis (1x3).
    axis (Matrix): The 3x3 matrix with each row representing an axis vector.

    Returns:
    Matrix: An 8x3 matrix where each row represents a vertex of the cuboid.
    """

    # Ensure the input is in the correct format and convert to sympy Matrices if necessary
    center = sp.Matrix(center).reshape(3, 1)
    size = sp.Matrix(size).reshape(3, 1)
    axis = sp.Matrix(axis)

    # Calculate the half sizes
    half_size = size / 2

    # Each corner of the cuboid is a combination of half the size along each axis.
    # We start with an 8x3 matrix where each row will be one of the corners of the cuboid.
    corners = sp.Matrix([[-1, -1, -1],
                      [1, -1,-1],
                      [1, 1, -1],
                      [-1, 1, -1],
                      [-1, -1, 1],
                      [1, -1, 1],
                      [1, 1, 1],
                      [-1, 1, 1]])

    # We multiply the corners by the half sizes to scale them
    # Here we do element-wise multiplication between each row of corners and half_size
    half_size = sp.Matrix.hstack(*[half_size,]*8)
    scaled_corners = corners.multiply_elementwise(half_size.T)

    # We then rotate the corners by the axis orientation
    vertices = scaled_corners * axis.T

    # Finally, we translate the vertices to the center of the cuboid
    center = sp.Matrix.hstack(*[center,] * 8)
    vertices = vertices + center.T

    return vertices.T


def create_cuboid_face_edge_centers_sympy(center, size, axis):
    """
    Create the center points of each face and edge of a 3D cuboid using sympy.

    Parameters:
    center (Matrix): The center of the cuboid (1x3).
    size (Matrix): The size of the cuboid along each axis (1x3).
    axis (Matrix): The 3x3 matrix with each row representing an axis vector.

    Returns:
    Tuple[Matrix, Matrix]: Two matrices, one for the centers of the faces (6x3) and one for the centers of the edges (12x3).
    """
    # Create vertices using the previous function
    vertices = create_cuboid_vertices_sympy(center, size, axis)

    # Define face and edge vertex indices
    # down up left right back frout
    face_indices = [[0, 1, 4, 6], [3, 7, 6, 2], [0, 4, 7, 3], [5, 1, 2, 6], [0, 3, 2, 1], [4, 5, 6, 7]]
    edge_indices = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], 
                    [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    # Calculate face centers
    face_centers = []
    for face in face_indices:
        face_center = sp.Matrix([0, 0, 0])
        for vertex in face:
            face_center += vertices[:, vertex]
        face_center /= len(face)
        face_centers.append(face_center)
    face_centers = sp.Matrix.hstack(*face_centers)

    edge_centers = []
    for edge in edge_indices:
        edge_center = sp.Matrix([0, 0, 0])
        for vertex in edge:
            edge_center += vertices[:, vertex]
        edge_center /= len(edge)
        edge_centers.append(edge_center)
    edge_centers = sp.Matrix.hstack(*edge_centers)
    

    return face_centers, edge_centers

    
def create_cuboid_vertices_numpy(center, size, axis):
    """
    Create the vertices of a 3D cuboid.

    Parameters:
    center (np.array): The center of the cuboid (1x3).
    size (np.array): The size of the cuboid along each axis (1x3).
    axis (np.array): The 3x3 matrix with each row representing an axis vector.

    Returns:
    np.array: An 8x3 matrix where each row represents a vertex of the cuboid.
    """

    # Ensure the input is in the correct format
    center = np.asarray(center).reshape(3)
    size = np.asarray(size).reshape(3)
    axis = np.asarray(axis).reshape(3, 3)

    # Calculate the half sizes
    half_size = size / 2.0

    # Each corner of the cuboid is a combination of half the size along each axis.
    # We start with an 8x3 matrix where each row will be one of the corners of the cuboid.
    corners = np.array([[1, 1, 1],
                        [1, 1, -1],
                        [1, -1, 1],
                        [1, -1, -1],
                        [-1, 1, 1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [-1, -1, -1]])

    # We multiply the corners by the half sizes to scale them
    corners = corners * half_size

    # We then rotate the corners by the axis orientation
    vertices = corners @ axis.T

    # Finally, we translate the vertices to the center of the cuboid
    vertices += center

    return vertices

def to_homogeneous_4x4(mat_3x3):
    # Create a 4x4 zero matrix
    mat_4x4 = sp.Matrix.zeros(4)
    # Embed the 3x3 matrix into the 4x4 matrix
    for i in range(3):
        for j in range(3):
            mat_4x4[i,j] = mat_3x3[i,j]
    # Set the bottom-right corner to 1 for homogeneous coordinates
    mat_4x4[3,3] = 1
    return mat_4x4

def to_translation_matrix(translation_vec):
    if translation_vec.shape != (3,):
        raise ValueError("Input array must be of shape (3,)")
    
    # Create a 4x4 identity matrix using sympy
    matrix = sp.Matrix.eye(4)
    
    # Set the translation values
    matrix[0,3], matrix[1,3], matrix[2,3] = translation_vec
    
    return matrix


def rotation_matrix_4x4(kx, ky, kz, theta):
    # Define the skew-symmetric matrix of K
    K_skew = sp.Matrix([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])
    # Calculate the 3x3 rotation matrix using Rodrigues' rotation formula
    R = sp.Matrix.eye(3) + sp.sin(theta) * K_skew + (1 - sp.cos(theta)) * K_skew**2
    # Extend to 4x4 matrix
    R_4x4 = sp.Matrix(4, 4, lambda i, j: 0)
    for i in range(3):
        for j in range(3):
            R_4x4[i,j] = R[i,j]
    R_4x4[3,3] = 1
    return R_4x4


def rotation_matrix_4x4(kx, ky, kz, theta):
    # Define the skew-symmetric matrix of K
    K_skew = sp.Matrix([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])
    # Calculate the 3x3 rotation matrix using Rodrigues' rotation formula
    R = sp.Matrix.eye(3) + sp.sin(theta) * K_skew + (1 - sp.cos(theta)) * K_skew**2
    # Extend to 4x4 matrix
    R_4x4 = sp.Matrix(4, 4, lambda i, j: 0)
    for i in range(3):
        for j in range(3):
            R_4x4[i,j] = R[i,j]
    R_4x4[3,3] = 1
    return R_4x4

def matrix_multiply_4x4(A, B):
    # Ensure matrices are 4x4
    if len(A) != 4 or len(B) != 4 or any(len(row) != 4 for row in A) or any(len(row) != 4 for row in B):
        raise ValueError("Both matrices must be 4x4 in size.")
    
    # Perform matrix multiplication
    C = [[0 for _ in range(4)] for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] + A[i][3] * B[3][j]

    return C

def matrix_vector_multiply_4x4(matrix, vector):
    # Ensure the matrix is 4x4
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("The matrix must be 4x4 in size.")
    
    # Ensure the vector has size 3
    if len(vector) != 3:
        raise ValueError("The vector must have 3 elements.")
    
    # Convert 3-vector to 4x1 by appending '1'
    vector_4x1 = vector + [1]

    # Multiply the matrix with the 4x1 vector
    result = [0, 0, 0, 0]
    for i in range(4):
        result[i] = sum(matrix[i][j] * vector_4x1[j] for j in range(4))

    return result


def rotation_matrix(axis, angle):
    """
    Compute 3x3 rotation matrix given an axis and an angle.
    """
    K = sp.Matrix([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = sp.eye(3) + sp.sin(angle) * K + (1 - sp.cos(angle)) * K**2
    return R

def reflection_matrix(normal):
    """
    Compute 3x3 reflection matrix given a plane normal.
    """
    n = normal / normal.norm()
    R = sp.eye(3) - 2 * n * n.T
    return R

def matrix_to_axis_angle(R):
    """
    Convert 3x3 rotation matrix to axis-angle representation.
    """
    angle = sp.acos((R.trace() - 1) / 2)
    axis = sp.Matrix([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ])
    axis = axis / axis.norm()
    return axis, angle

def reflect_rotation(axis, angle, plane_normal):
    # Convert axis-angle to matrix
    R = rotation_matrix(axis, angle)
    
    # Get reflection matrix
    R_reflect = reflection_matrix(plane_normal)
    
    # Reflect rotation matrix
    R_new = R_reflect * R * R_reflect
    
    # Convert reflected matrix back to axis-angle
    new_axis, new_angle = matrix_to_axis_angle(R_new)
    
    return new_axis, new_angle



def assign_directions(point_set):
    face_1_indices = (3, 2, 1, 0)
    face_2_indices = (4, 5, 6, 7)
    face_3_indices = (0, 1, 5, 4)
    face_4_indices = (2, 3, 7, 6)
    face_5_indices = (1, 2, 6, 5)
    face_6_indices = (0, 4, 7, 3)
    face_set = [face_1_indices, face_2_indices,
                face_3_indices, face_4_indices,
                face_5_indices, face_6_indices]
    dir_to_face = {
    }
    directions = DIRECTIONS.copy()

    center = sp.ones(1, point_set.shape[0]) * point_set / point_set.shape[0]
    for face in face_set:
        cps = point_set[face, :]
        # normal = (cps[1, :] - cps[0, :]).cross(cps[2, :] - cps[1, :])
        # normal = normal.normalized()
        face_center = sp.ones(1, cps.shape[0]) * cps / cps.shape[0]
        normal = (face_center - center).normalized()
        stack_normal = sp.Matrix.vstack(*[normal for _ in range(len(directions))] )
        stack_directions = sp.Matrix.vstack(*[sp.Matrix(1, 3, direction) for direction in directions])
        dot_products = stack_normal.multiply_elementwise(stack_directions)
        dot_products = dot_products * sp.ones(dot_products.shape[1], 1)
        max_ind = np.argmax(np.array(dot_products))
        max_dir = directions[max_ind]
        max_dir_name = DIR_TO_NAME[max_dir]
        dir_to_face[max_dir_name] = face
        directions.pop(max_ind)
    
    dir_to_edge = {}
    for dir_1, dir_2 in itertools.combinations(dir_to_face.keys(), 2):
        face_1 = dir_to_face[dir_1]
        face_2 = dir_to_face[dir_2]
        face_1_set = set(face_1)
        face_2_set = set(face_2)
        edge_set = face_1_set.intersection(face_2_set)
        edge_set = list(edge_set)
        edge_set.sort() # Should this be done?
        edge_set = tuple(edge_set)
        if len(edge_set) == 2:
            key = tuple(sorted([dir_1, dir_2]))
            dir_to_edge[key] = edge_set

    dir_to_corner = {}
    for dir_1, dir_2, dir_3 in itertools.combinations(dir_to_face.keys(), 3):
        face_1 = dir_to_face[dir_1]
        face_2 = dir_to_face[dir_2]
        face_3 = dir_to_face[dir_3]
        face_1_set = set(face_1)
        face_2_set = set(face_2)
        face_3_set = set(face_3)
        corner_set = face_1_set.intersection(face_2_set, face_3_set)
        corner_set = list(corner_set)
        corner_set.sort()
        corner_set = tuple(corner_set)
        if len(corner_set) == 1:
            key = tuple(sorted([dir_1, dir_2, dir_3]))
            dir_to_corner[key] = corner_set
    dir_to_face.update(dir_to_edge)
    dir_to_face.update(dir_to_corner)
    return dir_to_face