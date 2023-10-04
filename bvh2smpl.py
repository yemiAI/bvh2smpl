#import torch
import numpy as np
import argparse
import pickle
import smplx
from bvh import Bvh

from utils import bvh, quat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/smpl/")
    parser.add_argument("--gender", type=str, default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--poses", type=str, default="data/gWA_sFM_cAll_d27_mWA5_ch20.pkl")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=str, default="data/gWA_sFM_cAll_d27_mWA5_ch20.bvh")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("bvhfile", type = str)
    return parser.parse_args()

def euler_to_axis_angle(euler_angles):
    # Convert Euler angles to axis-angle representation (adjust as needed for your BVH data)
    roll, pitch, yaw = euler_angles
    angle = np.sqrt(roll**2 + pitch**2 + yaw**2)
    axis = np.array([roll, pitch, yaw]) / angle if angle != 0 else np.zeros(3)
    return axis * angle


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

    return quat.ik_rot(grot_mirror, parents), trans_mirror


def bvh2smpl(model_path:str , poses:str , output:str, mirror:bool, 
             model_type='bvh', gender='MALE', fps=120)-> None:
    """Save SMPL file created by bvh parameters.

    Args:
        model_path(str): path to bvh models
        poses (str): Path to 
        output (str): Where to save SMPL. 
        mirror (bool):  Whether save mirror motion or not. 
        fps(int, optional): Frame per second. Default to 120.
        """
    names = ["Hips", 
             "Spine", 
             "Spine1", 
             "Neck", 
             "Head", 
             "LeftShoulder", 
             "LeftArm", 
             "LeftForeArm", 
             "LeftHand", 
             "LeftHandThumb1", 
             "LeftHandThumb2", 
             "LeftHandThumb3", 
             "LeftHandIndex1", 
             "LeftHandIndex2", 
             "LeftHandIndex3", 
             "LeftHandMiddle1", 
             "LeftHandMiddle2", 
             "LeftHandMiddle3",
             "LeftHandRing1", 
             "LeftHandRing2", 
             "LeftHandRing3", 
             "LeftHandPinky1", 
             "LeftHandPinky2", 
             "LeftHandPinky3", 
             "RightShoulder", 
             "RightArm",
             "RightForeArm",
             "RightHand",
             "RightHandThumb1",
             "RightHandThumb",
             "RightHandThumb3", 
             "RightHandIndex1", 
             "RightHandIndex2", 
             "RightHandIndex3", 
             "RightHandMiddle",
             "RightHandMiddle2",
             "RightHandMiddle3",
             "RightHandRing1",
             "RightHandRing2", 
             "RightHandRing3", 
             "RightHandPinky1",
             "RightHandPinky2",
             "RightHandPinky3",
             "LeftUpLeg",
             "LeftLeg", 
             "LeftFoot", 
             "LeftToeBase", 
             "RightUpLeg", 
             "RightLeg", 
             "RightFoot", 
             "RightToeBase"
        ]
        

        
# Load the SMPL model
smpl_model = smplx.create(model_path, model_type=model_type, gender=gender)

#model = bvh.create(model_path=model_path)
#args = parse_args()
with open (args.bvhfile) as fp:
    model = Bvh(fp.read())
    
    
    # Initialize arrays to store the converted poses
    out_poses = []
    
     for frame in range(bvh_data.nframes):
            # Extract joint rotations from BVH data (you may need to adjust this based on your BVH file)
            joint_rotations = []
            for joint_name in names:
                rotation = bvh_data.joint_channels(frame, joint_name)
                joint_rotations.extend(rotation)

            # Convert Euler angles to axis-angle representation
            axis_angle_rotations = []
            for i in range(0, len(joint_rotations), 3):
                euler_angles = joint_rotations[i:i + 3]
                axis_angle = euler_to_axis_angle(euler_angles)
                axis_angle_rotations.extend(axis_angle)

            # Adjust the axis-angle rotations for the BVH-to-SMPL bone mapping
            if mirror:
                # If mirroring, use mirror_rot_trans function to adjust rotations and translations
                mirrored_rotations, mirrored_translation = mirror_rot_trans(axis_angle_rotations, np.zeros(3), names, [])
                out_poses.append(np.concatenate((mirrored_rotations, mirrored_translation)))
            else:
                # Otherwise, use the axis-angle rotations as-is
                out_poses.append(np.concatenate((axis_angle_rotations, np.zeros(3))))
    
    # Now using the bvh documentation at https://github.com/20tab/bvh-python, get what information you need to match the bvh file with the amass output
    # The files to output with bare AMASS are:
    
    # ['poses', 'gender', 'mocap_framerate', 'betas', 'marker_data', 'dmpls', 'marker_labels', 'trans']
        
    

    # Gender is whatever you tell it to be
    out_gender = np.array(args.gender)
    
    # Maybe override this with args.fps if you like
    out_framerate = np.array(1.0 / model.frame_time)
    
    # Find out what 'betas' is and set the numbers appropriately. Alternatively: 
    out_betas = np.random.random([16])
    
    # Likewise with dmpls. Set to random, or zeros, or whatever
    out_dmpls = np.zeros([model.nframes, 8])
    
    # and do something similar with marker_data and markers. It probably doesn't matter much what you put here, providing you can display the AMASS file, and it goes into your machine learning framework

    # The equivalent of amass translation seems to be that of the *position components in the root bone of a BVH skeleton
    out_trans = np.array([[model.frame_joint_channel(frame, 'Hips', pos) for pos in ['Xposition', 'Yposition', 'Zposition']] for frame in range(model.nframes)])
    
    # Poses is the hard bit. You have to do the conversion from BVH Euler Angles to the SMPL axis-angle format, doing the inverse operations to the one in the smpl2bvh script
    # You also have to correspond the bones in the bvh with the output skeleton bones (hopefully it's a matching skeleton, otherwise you might have to build the appropriate skel)
    
    # Convert the list of poses to a numpy array
    out_poses = np.array(out_poses)
    
    # When you're done, save
    np.savez(args.output , gender = out_gender, mocap_framerate = out_framerate, betas = out_betas, dmpls = out_dmps, marker_labels = out_marker_labels, marker_data = out_marker_data, poses = out_poses, trans = out_trans)
    
    if __name__ == "__main__":
        args = parse_args()
        bvh2smpl(args.model_path, args.poses, args.output, args.mirror, gender=args.gender, fps=args.fps)
    
  
    