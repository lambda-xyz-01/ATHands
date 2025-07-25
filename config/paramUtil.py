import numpy as np

SNAP_PARENT = [
    0,  # 0's parent
    0,  # 1's parent
    1,
    2,
    0,  # 5's parent
    4,
    5,
    0,  # 9's parent
    7,
    8,
    0,  # 13's parent
    10,
    11,
    0,  # 17's parent
    13,
    14,
]

JOINT_ROOT_IDX = 7

REF_BONE_LINK = (0, 7)  # mid mcp

ID_ROOT_bone = np.array([0, 3, 6, 9, 12])  # ROOT_bone from wrist to MCP
ID_PIP_bone = np.array([1, 4, 7, 10, 13])  # PIP_bone from MCP to PIP
ID_DIP_bone = np.array([2, 5, 8, 11, 14])  # DIP_bone from  PIP to DIP


def get_joint_indices(joints_num_body, joints_num_hand):
    ib0 = 0
    ib1 = joints_num_body * 3
    il0 = ib1 + joints_num_body * 6
    il1 = il0 + joints_num_hand * 3
    ir0 = il1 + joints_num_hand * 6
    ir1 = ir0 + joints_num_hand * 3
    pos_indices = list(range(ib0, ib1)) + list(range(il0, il1)) + list(range(ir0, ir1))
    return pos_indices
  
BODY_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
LEFT_HAND_IDX  = [20, 34, 35, 36, 22, 23, 24, 25, 26, 27, 31, 32, 33, 28, 29, 30]
RIGHT_HAND_IDX = [21, 49, 50, 51, 37, 38, 39, 40, 41, 42, 46, 47, 48, 43, 44, 45]

HANDS_FINGERS_IDS = [
	[0, 1, 2, 3],    
	[0, 4, 5, 6],   
	[0, 7, 8, 9],      
	[0, 10, 11, 12],       
	[0, 13, 14, 15], 
]

kinematic_chain = [
    # Spine and Head
    [0, 2, 5, 8, 11, 14],  # Left_Hip → Spine1 → Spine2 → Spine3 → Neck → Head
    [0, 1, 2],                # Left_Hip → Right_Hip
    [11, 13, 16, 18, 20], # Neck → Right_Shoulder → Right_Elbow → Right_Wrist → Right_Hand
    [11, 12, 15, 17, 19], # Neck → Left_Shoulder → Left_Elbow → Left_Wrist → Left_Hand
    
    # Legs
    [0, 3, 6, 9],          # Left_Hip → Left_Knee → Left_Ankle → Left_Foot
    [1, 4, 7, 10],         # Right_Hip → Right_Knee → Right_Ankle → Right_Foot
    
    # Left Arm and Hand
    [19, 21, 22, 23],     # Left_Hand → Left_Thumb_MCP → Left_Thumb_PIP → Left_Thumb_DIP
    [19, 24, 25, 26],     # Left_Hand → Left_Index_MCP → Left_Index_PIP → Left_Index_DIP
    [19, 27, 28, 29],     # Left_Hand → Left_Middle_MCP → Left_Middle_PIP → Left_Middle_DIP
    [19, 30, 31, 32],     # Left_Hand → Left_Ring_MCP → Left_Ring_PIP → Left_Ring_DIP
    [19, 33, 34, 35],     # Left_Hand → Left_Pinky_MCP → Left_Pinky_PIP → Left_Pinky_DIP
    
    # Right Arm and Hand
    
    [20, 36, 37, 38],     # Right_Hand → Right_Thumb_MCP → Right_Thumb_PIP → Right_Thumb_DIP
    [20, 39, 40, 41],     # Right_Hand → Right_Index_MCP → Right_Index_PIP → Right_Index_DIP
    [20, 42, 43, 44],     # Right_Hand → Right_Middle_MCP → Right_Middle_PIP → Right_Middle_DIP
    [20, 45, 46, 47],     # Right_Hand → Right_Ring_MCP → Right_Ring_PIP → Right_Ring_DIP
    [20, 48, 49, 50]      # Right_Hand → Right_Pinky_MCP → Right_Pinky_PIP → Right_Pinky_DIP
]

JOINTS_NUM_BODY = len(BODY_IDX) 
JOINTS_NUM_HAND = len(LEFT_HAND_IDX) 

POS_INDICES = get_joint_indices(JOINTS_NUM_BODY, JOINTS_NUM_HAND)

NUM_BODY_FEATS = JOINTS_NUM_BODY * 9 # 3 (pos) + 6 (rot)
NUM_HAND_FEATS = JOINTS_NUM_HAND * 9

# print('features:', 'body >', NUM_BODY_FEATS, '| hand >', NUM_HAND_FEATS)