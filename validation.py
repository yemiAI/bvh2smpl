import numpy as np
import argparse
import smplx
import pickle
from bvh import Bvh

from utils import quat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/smpl/")
    parser.add_argument("--gender", type=str, default="male", choices=["male", "female", "neutral"])
    #the poses bit need to be fixed 
    parser.add_argument("--poses", type=str, default="data/trial.bvh")
    #fix it here 
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=str, default="data/output.npz")
    parser.add_argument("--mirror", action="store_true")
    
    parser.add_argument("--tpose", action="store_true")
    parser.add_argument("--random", action="store_true")
    
    parser.add_argument("bvhfile", type=str)
    return parser.parse_args()

#def euler_to_axis_angle(euler_angles):
    # Convert Euler angles to axis-angle representation (adjust as needed for your BVH data)
 #   roll, pitch, yaw = euler_angles
  #  angle = np.sqrt(roll**2 + pitch**2 + yaw**2)
   # axis = np.array([roll, pitch, yaw]) / angle if angle != 0 else np.zeros(3)
    #return axis * angle

def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]

    return quat.ik_rot(grot_mirror, parents)


test_data= np.array([4.48230864e-02,  2.19861197e+00,  2.33427666e+00, -1.43781344e-01,
        -5.01852225e-02, -9.10971754e-02, -3.66620460e-01, -4.26968013e-02,
       -1.63511801e-01,  5.77447311e-02, -5.99430332e-03, -4.72619659e-02,
        3.70787492e-01, -5.72388575e-02, -1.09751429e-01,  1.04470884e+00,
       -7.67636696e-02,  5.49034695e-02,  1.23821047e-01, -6.29301423e-02,
        5.97639102e-02, -2.92350050e-01,  2.97306571e-01,  1.64224668e-01,
       -1.93856549e-01, -1.86292090e-01,  6.69726046e-03, -2.24882835e-02,
       -2.84418843e-02, -1.00866697e-02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.53993880e-01, -6.58021852e-02, -4.92713036e-02, -3.59147976e-02,
        3.76255075e-04, -2.36591430e-01, -9.38961478e-03, -1.06391863e-01,
        1.98024289e-01, -1.69262142e-01,  3.53572520e-02,  7.56902705e-02,
        1.89125596e-01, -4.29778823e-01, -8.54607710e-01,  3.30852186e-01,
       -6.86773108e-02,  9.32141783e-01,  1.82157482e-01, -1.44703345e+00,
        3.24325556e-01,  3.05808263e-01,  1.22662733e+00, -1.37924960e-01,
       -6.26150294e-02, -2.09987214e-01,  1.93827248e-01, -3.01103213e-01,
        2.23994474e-01, -1.38783545e-01,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.11678716e-01,
        4.28921748e-02, -4.16441837e-01,  1.08811325e-01, -6.59856789e-02,
       -7.56220010e-01, -9.63929651e-02, -9.09156592e-02, -1.88459291e-01,
       -1.18095039e-01,  5.09438526e-02, -5.29584500e-01, -1.43698409e-01,
        5.52417000e-02, -7.04857141e-01, -1.91829168e-02, -9.23368482e-02,
       -3.37913524e-01, -4.57032983e-01, -1.96283945e-01, -6.25457533e-01,
       -2.14652379e-01, -6.59982865e-02, -5.06894207e-01, -3.69724357e-01,
       -6.03446264e-02, -7.94902279e-02, -1.41869695e-01, -8.58526333e-02,
       -6.35528257e-01, -3.03341587e-01, -5.78809752e-02, -6.31389210e-01,
       -1.76120885e-01, -1.32093076e-01, -3.73354576e-01,  8.50964279e-01,
        2.76922742e-01, -9.15480698e-02, -4.99839438e-01,  2.65564722e-02,
        5.28808767e-02,  5.35559148e-01,  4.59610410e-02, -2.77358021e-01,
        1.11678716e-01, -4.28921748e-02,  4.16441837e-01,  1.08811325e-01,
        6.59856789e-02,  7.56220010e-01, -9.63929651e-02,  9.09156592e-02,
        1.88459291e-01, -1.18095039e-01, -5.09438526e-02,  5.29584500e-01,
       -1.43698409e-01, -5.52417000e-02,  7.04857141e-01, -1.91829168e-02,
        9.23368482e-02,  3.37913524e-01, -4.57032983e-01,  1.96283945e-01,
        6.25457533e-01, -2.14652379e-01,  6.59982865e-02,  5.06894207e-01,
       -3.69724357e-01,  6.03446264e-02,  7.94902279e-02, -1.41869695e-01,
        8.58526333e-02,  6.35528257e-01, -3.03341587e-01,  5.78809752e-02,
        6.31389210e-01, -1.76120885e-01,  1.32093076e-01,  3.73354576e-01,
        8.50964279e-01, -2.76922742e-01,  9.15480698e-02, -4.99839438e-01,
       -2.65564722e-02, -5.28808767e-02,  5.35559148e-01, -4.59610410e-02,
        2.77358021e-01])



  #functions that turns bvh euler angles into SMpl axis angle and reverse operation

  
def forwardA_E(a_rots):
    rots = quat.from_axis_angle(a_rots)
    
    order = "zyx"
    #pos = offsets[None].repeat(len(rots), axis=0)
    #positions = pos.copy()
    #positions[:,0] += trans * 100
    rotations = np.degrees(quat.to_euler(rots, order=order))


    return rotations


def BackwardE_A(a_rots):
    
    order = "zyx"
    b_rots = quat.from_euler(np.radians(a_rots), order=order)
    
    b_rotations = quat.to_scaled_angle_axis(b_rots)

    return b_rotations

#forwardA_E(test_data[3:6])
#euler= forwardA_E(test_data[3:6])
axis_angles = test_data[3:6]

euler= forwardA_E(axis_angles)

axis = BackwardE_A(euler)


print(euler)
print(axis_angles)
print(axis)
