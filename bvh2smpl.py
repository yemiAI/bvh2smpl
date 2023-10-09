import numpy as np
import argparse
import smplx
import pickle
from bvh import Bvh

from utils import quat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/smpl/")
    parser.add_argument("--gender", type=str, default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    #the poses bit need to be fixed 
    parser.add_argument("--poses", type=str, default="data/trial.bvh")
    #fix it here 
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=str, default="data/output.npz")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("bvhfile", type=str)
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

    return quat.ik_rot(grot_mirror, parents)

def bvh2smpl(model_path: str, bvhfile: str, output: str, mirror: bool,
             model_type='bvh', gender='MALE', fps=120):
    """Convert BVH file to SMPL format and save as NPZ file.

    Args:
        model_path (str): Path to 
        bvhfile (str): Path to BVH file to convert.
        output (str): Path to save the converted SMPL NPZ file.
        mirror (bool): Whether to save mirrored motion or not.
        model_type (str, optional): Type of SMPL model (e.g., 'bvh'). Default is 'bvh'.
        gender (str, optional): Gender of the SMPL model ('MALE', 'FEMALE', or 'NEUTRAL'). Default is 'MALE'.
        fps (int, optional): Frame per second. Default is 120.
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
    #smpl_model = smplx.create(model_path, model_type='smpl', gender='MALE')
    
    #args = parse_args()

    # Load the BVH file
    with open(args.bvhfile) as fp:
        model = Bvh(fp.read())

    # Initialize arrays to store the converted poses
    out_poses = []

    for frame in range(model.nframes):
        # Extract joint rotations from BVH data
        joint_rotations = []
        for joint_name in names:

            if (joint_name in model.get_joints_names()):
                rotation = model.frame_joint_channels(frame, joint_name, ['Xrotation', 'Yrotation', 'Zrotation'])
            else:
                rotation = [0.0, 0.0, 0.0]
                
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
            mirrored_rotations = mirror_rot_trans(axis_angle_rotations, np.zeros(3), names, [])
            out_poses.append(mirrored_rotations)
        else:
            # Otherwise, use the axis-angle rotations as-is
            out_poses.append(axis_angle_rotations)

    # Prepare other SMPL parameters (you may need to customize this part)
    out_gender = np.array(gender)
    out_framerate = np.array(1.0 / model.frame_time)
    out_betas = np.random.random([16])
    #out_dmpls = np.zeros([len(out_poses), 8])
    out_dmpls = np.zeros([model.nframes, 8])
    
    out_trans = np.array([[model.frame_joint_channel(frame, 'Hips', pos) for pos in ['Xposition', 'Yposition', 'Zposition']] for frame in range(model.nframes)])

    # TODO: Populate these
    out_marker_labels = np.array([])
    out_marker_data = np.array([])
    
    # Convert the list of poses to a numpy array
    out_poses = np.array(out_poses)

    # Save the SMPL parameters as an NPZ file
    np.savez(args.output , gender = out_gender, mocap_framerate = out_framerate, betas = out_betas, dmpls = out_dmpls, marker_labels = out_marker_labels, marker_data = out_marker_data, poses = out_poses, trans = out_trans)
    
    

if __name__ == "__main__":
    args = parse_args()
    bvh2smpl(args.model_path, args.bvhfile, args.output, args.mirror, gender=args.gender, fps=args.fps)
