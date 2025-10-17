import itertools
from edit_sys.shape_system.geometric_atoms import *
from edit_sys.shape_system.shape_atoms import *
from edit_sys.shape_system.relations import *
from edit_sys.shape_system.utils import create_cuboid_vertices_numpy
axis_X = LeftToRightLine()
axis_Y = BackToFrontLine()
axis_Z = DownToUpLine()

plane_XY = HorizontalFacingUpPlane()
plane_XZ = VerticalFacingBackPlane()
plane_YZ = VerticalFacingRightPlane()


center = np.array([0.5, -0.5, -0.5])
size = np.array([0.1, 0.1, 1])
axis = np.array([[1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]])
# flr_cuboid_points = create_cuboid_vertices(center, size, axis)
# flr_cuboid_points = sp.Matrix(flr_cuboid_points)    
frl_prim = PrimitiveCuboid(center, size, axis)

center = np.array([-0.5, -0.5, -0.5])
size = np.array([0.1, 0.1, 1])
axis = np.array([[-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]])
# fll_cuboid_points = create_cuboid_vertices(center, size, axis)
fll_prim = PrimitiveCuboid(center, size, axis)

center = np.array([0.5, 0.5, -0.5])
size = np.array([0.1, 0.1, 1])
axis = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
# brl_cuboid_points = create_cuboid_vertices(center, size, axis)
brl_prim = PrimitiveCuboid(center, size, axis)

center = np.array([-0.5, 0.5, -0.5])
size = np.array([0.1, 0.1, 1])
axis = np.array([[-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

# bll_cuboid_points = create_cuboid_vertices(center, size, axis)
bll_prim = PrimitiveCuboid(center, size, axis)


center = np.array([0.0, 0, 0])
size = np.array([1.1, 1.1, 0.1])
axis = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
# seat_cuboid_points = create_cuboid_vertices(center, size, axis)
# seat_cuboid_points = sp.Matrix(seat_cuboid_points)    
seat_prim = PrimitiveCuboid(center, size, axis)

center = np.array([0.0, 0.5, 0.5])
size = np.array([1.1, 0.1, 1.1])
axis = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
# back_cuboid_points = create_cuboid_vertices(center, size, axis)
# back_cuboid_points = sp.Matrix(back_cuboid_points)    
back_prim = PrimitiveCuboid(center, size, axis)

front_right_leg = Part("front_right_leg", frl_prim)
front_left_leg = Part("front_left_leg", fll_prim)
back_right_leg = Part("back_right_leg", brl_prim)
back_left_leg = Part("back_left_leg", bll_prim)
seat = Part("seat", seat_prim)
back = Part("back", back_prim)

center = np.array([0.0, 0.0, 0.0])
size = np.array([1.5, 1.5, 1.5])
axis = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
# back_cuboid_points = create_cuboid_vertices(center, size, axis)
# back_cuboid_points = sp.Matrix(back_cuboid_points)    
chair_prim = PrimitiveCuboid(center, size, axis)

part_list = [front_right_leg, front_left_leg, back_right_leg, back_left_leg, seat, back]
simple_chair = Part("Chair", chair_prim, sub_parts=part_list)

seat_point_back_right = PointFeature(seat_prim, Vector(np.array([0.91, 0.91, -0.0])))
seat_point_front_right = PointFeature(seat_prim, Vector(np.array([0.91, -0.91, -0.0])))
seat_point_back_left = PointFeature(seat_prim, Vector(np.array([-0.91, 0.91, -0.0])))
seat_point_front_left = PointFeature(seat_prim, Vector(np.array([-0.91, -0.91, -0.0])))

seat_point_b_r = PointFeature(seat_prim, Vector(np.array([0.91, 0.909, -1.0])))
seat_point_b_l = PointFeature(seat_prim, Vector(np.array([-0.91, 0.909, -1.0])))

back_point_right = PointFeature(back_prim, Vector(np.array([0.91, 0, -1])))
back_point_left = PointFeature(back_prim, Vector(np.array([-0.91, 0, -1])))

front_right_leg_point = PointFeature(frl_prim, Vector(UNIT_Z.copy()))
front_left_leg_point = PointFeature(fll_prim, Vector(UNIT_Z.copy()))
back_right_leg_point = PointFeature(brl_prim, Vector(UNIT_Z.copy()))
back_left_leg_point = PointFeature(bll_prim, Vector(UNIT_Z.copy()))

ReflectionSymmetry((frl_prim, fll_prim), plane_YZ)
ReflectionSymmetry((frl_prim, brl_prim), plane_XZ)
ReflectionSymmetry((brl_prim, bll_prim), plane_YZ)
ReflectionSymmetry((fll_prim, bll_prim), plane_XZ)
# RotationSymmetry((front_left_leg, front_right_leg, back_right_leg, back_left_leg),
#                    axis_Z, 90)
PointContact((seat_point_front_right, front_right_leg_point))
PointContact((seat_point_front_left, front_left_leg_point))
PointContact((seat_point_back_right, back_right_leg_point))
PointContact((seat_point_back_left, back_left_leg_point))
PointContact((seat_point_b_r, back_point_right))
PointContact((seat_point_b_l, back_point_left))


