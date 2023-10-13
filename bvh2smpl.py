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
    
    parser.add_argument("--bone_tester", type=int)
    
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


def euler_to_axis(a_rots):
    
    order = "zyx"
    b_rots = quat.from_euler(np.radians(a_rots), order=order) #rename
    
    b_rotations = quat.to_scaled_angle_axis(b_rots)

    return b_rotations

bone_map =[0,6,9,12,15,13,16,18,20,37,38,39,25,26,27,28,29,30,34,35,36,31,32,33,17,14,19,21,52,53,54,40,41,42,43,44,45,49,50,51,46,47,48,1,10,4,7,2,11,5,8]    

#write numpy

#bone_map = [] 


def bone_mapping(input_poses) :
    out_poses= np.zeros([165])
    #rearranging input poses
    #out_poses[48:51]= input_pose[12:15] #head ..can be improved with a look up table (an array which maps the bone indices )
    for b_from, b_toX in enumerate(bone_map):
        
        if b_toX <= 53 :
            
            b_to = b_toX + 1
            
        
            #print(b_to)
            #print(b_from)
            out_poses[3*b_to: 3*b_to+3] = input_poses[3*b_from:3*b_from + 3]
        
    return out_poses
        
        

def bvh2smpl(model_path: str, bvhfile: str, output: str, mirror: bool,
             model_type='bvh', gender='male', fps=120):
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
    
    #map along the x axis 
    smpl_names = ["pelvis",
            "left_hip",
             "right_hip",
             "spine1",
             "left_knee",
             "right_knee",
             "spine2",
             "left_ankle",
             "right_ankle",
             "spine3",
             "left_foot",
             "right_foot",
             "neck",
             "left_collar",
             "right_collar",
             "head",
             "left_shoulder",
             "right_shoulder",
             "left_elbow",
             "right_elbow",
             "left_wrist",
             "right_wrist",
             "jaw",
             "left_eye_smplhf",
             "right_eye_smplhf",
             "left_index1",
             "left_index2",
             "left_index3",
             "left_middle1",
             "left_middle2",
             "left_middle3",
             "left_pinky1",
             "left_pinky2",
             "left_pinky3",
             "left_ring1",
             "left_ring",
             "left_ring3",
             "left_thumb1",
             "left_thumb2",
             "left_thumb3",
             "right_index1",
             "right_index2",
             "right_index3",
             "right_middle1",
             "right_middle2",
             "right_middle3",
             "right_pinky1",
             "right_pinky2",
             "right_pinky3",
             "right_ring1",
             "right_ring2",
             "right_ring3",
             "right_thumb1",
             "right_thumb2",
             "right_thumb3"
             ]
    
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
    #out_poses = []
    if (not args.tpose) and (args.bone_tester is None): #populating the outposes to zero        
        

        out_poses = np.zeros([model.nframes,165])

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
                
                axis_angle = euler_to_axis(euler_angles)
                
                
                #axis_angle = euler_to_axis_angle(euler_angles)
                axis_angle_rotations.extend(axis_angle)

            # Adjust the axis-angle rotations for the BVH-to-SMPL bone mapping
            if mirror:
                # If mirroring, use mirror_rot_trans function to adjust rotations and translations
                mirrored_rotations = mirror_rot_trans(axis_angle_rotations, np.zeros(3), names, [])
                #out_poses[frame,48:51] = mirrored_rotations[15:18]
                #out_poses[frame, 66:156] = mirrored_rotations[63:153]
                
                out_poses [frame,:]= bone_mapping(mirrored_rotations)
                # Otherwise, use the axis-angle rotations as-is
            else: 
                out_poses [frame,:] = bone_mapping(axis_angle_rotations)
                #out_poses[frame, :63] = axis_angle_rotations[0:63]
                #out_poses[frame, 66:156] = axis_angle_rotations[63:153]

    else:

        out_poses = np.zeros([model.nframes, 165])
        
        if args.bone_tester is not None:
            
            out_poses[:,3*args.bone_tester: 3*args.bone_tester + 3] = 1.57
        
    if args.random : 
        
        out_poses = np.random.random(out_poses.shape)


    #amass_template = np.array([])                         
    amass_template = np.load('salsa_1_stageii.npz', allow_pickle=True )
    
    #
    #Create the pose_jaw and pose_eye arrays from np.zeros
    
    
    #Create the root_orientation and pose_body and pose_hand arrays from poses
    
    #Pull the other files in the MoSH from a sample file

        
    # Prepare other SMPL parameters (you may need to customize this part)
    out_gender = np.array(gender)
    out_framerate = np.array(1.0 / model.frame_time)
    out_betas = np.random.random([16])
    #out_dmpls = np.zeros([len(out_poses), 8])
    out_dmpls = np.zeros([model.nframes, 8])
    out_mocaptime = np.array(model.frame_time * model.nframes)
    
    print(out_poses.shape)
    out_root_orient = out_poses[:, 0:3]
    
    out_pose_body = out_poses[:, 3:66]
    out_pose_hand = out_poses[:, 75:165]
    out_pose_jaw = np.zeros([model.nframes,6])
    out_pose_eyes = np.zeros([model.nframes,6])
    out_trans = np.array([[model.frame_joint_channel(frame, 'Hips', pos) for pos in ['Xposition', 'Yposition', 'Zposition']] for frame in range(model.nframes)])

    # TODO: Populate these
    out_marker_labels = np.array([])
    out_marker_data = np.array([])
    
    
    
    # Convert the list of poses to a numpy array
    out_poses = np.array(out_poses)

    # Save the SMPL parameters as an NPZ file
    np.savez(args.output , gender = out_gender, mocap_framerate = out_framerate, betas = out_betas, dmpls = out_dmpls, marker_labels = out_marker_labels, marker_data = out_marker_data, poses = out_poses, trans = out_trans, mocap_time_length = out_mocaptime, root_orient=out_root_orient, pose_body=out_pose_body, pose_hand= out_pose_hand, pose_eyes= out_pose_eyes)
    
    np.savez("mosh_%s"%args.output,markers_latent=amass_template['markers_latent'],latent_labels=amass_template['latent_labels'], markers_latent_vids=amass_template['markers_latent_vids'], surface_model_type = amass_template['surface_model_type'], num_betas = amass_template['num_betas']) 
    

if __name__ == "__main__":
    args = parse_args()
    bvh2smpl(args.model_path, args.bvhfile, args.output, args.mirror, gender=args.gender, fps=args.fps)
