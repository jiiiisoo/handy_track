# CRITICAL: Import order matters for some dependencies
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
import math
import joblib
from torch.autograd import Variable
from termcolor import cprint
from tqdm import tqdm

# ManipTrans imports (after basic imports)
IMPORTS_AVAILABLE = False
try:
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
    from main.dataset.transform import (
        aa_to_quat,
        aa_to_rotmat,
        quat_to_rotmat,
        rot6d_to_aa,
        rot6d_to_quat,
        rot6d_to_rotmat,
        rotmat_to_aa,
        rotmat_to_quat,
        rotmat_to_rot6d,
    )
    IMPORTS_AVAILABLE = True
    print("✅ ManipTrans imports successful")
except ImportError as e:
    print(f"⚠️ ManipTrans imports not available: {e}")
    IMPORTS_AVAILABLE = False

# Try to import MANO model
MANO_AVAILABLE = False
try:
    # You might need to install manopth or use a different MANO implementation
    # This is a placeholder - adjust based on your MANO implementation
    import manopth
    MANO_AVAILABLE = True
    print("✅ MANO imported successfully")
except ImportError:
    print("⚠️ MANO model not available. Using mock MANO for structure.")
    MANO_AVAILABLE = False

def find_inspire_hand_urdf():
    """Find Inspire Hand URDF file with comprehensive search"""
    
    # Try ManipTrans factory first
    if IMPORTS_AVAILABLE:
        try:
            inspire_hand = DexHandFactory.create_hand("inspire", "right")
            if hasattr(inspire_hand, 'urdf_path') and os.path.exists(inspire_hand.urdf_path):
                print(f"✅ Found URDF via factory: {inspire_hand.urdf_path}")
                return inspire_hand.urdf_path
        except Exception as e:
            print(f"Factory method failed: {e}")
    
    # Manual search in common locations
    search_paths = [
        "/workspace/ManipTrans/maniptrans_envs/assets/urdf/inspire_hand_right.urdf",
        "/workspace/ManipTrans/maniptrans_envs/assets/urdf/inspire_hand.urdf",
        "/workspace/ManipTrans/maniptrans_envs/assets/inspire_hand/urdf/inspire_hand_right.urdf",
        "/workspace/ManipTrans/assets/urdf/inspire_hand_right.urdf",
        "./maniptrans_envs/assets/urdf/inspire_hand_right.urdf",
        "./assets/urdf/inspire_hand_right.urdf"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"✅ Found URDF at: {path}")
            return path
    
    print("⚠️ No Inspire Hand URDF found")
    return None

class MockMANO:
    """Enhanced Mock MANO class when the actual MANO model is not available"""
    def __init__(self, mano_root="./mano", use_pca=False, ncomps=45, flat_hand_mean=False):
        self.faces = np.random.randint(0, 778, (1538, 3))  # Mock faces
        print("🤖 Initialized mock MANO model")
        
    def forward(self, global_orient, hand_pose, betas, transl=None):
        batch_size = global_orient.shape[0]
        device = global_orient.device
        
        # Mock outputs with realistic proportions
        vertices = torch.randn(batch_size, 778, 3, device=device) * 0.05
        joints = torch.zeros(batch_size, 21, 3, device=device)
        
        # Create realistic hand structure
        palm_size = 0.08
        
        # Wrist at origin
        joints[:, 0] = 0  # wrist
        
        # Realistic MANO-style joint layout
        joint_positions = [
            # Thumb chain
            [palm_size*0.3, palm_size*0.4, 0.01],
            [palm_size*0.5, palm_size*0.6, 0.02],
            [palm_size*0.65, palm_size*0.8, 0.02],
            [palm_size*0.75, palm_size*1.0, 0.02],
            # Index finger
            [palm_size*0.25, palm_size*1.1, 0.0],
            [palm_size*0.25, palm_size*1.4, 0.0],
            [palm_size*0.25, palm_size*1.7, 0.0],
            [palm_size*0.25, palm_size*1.9, 0.0],
            # Middle finger
            [0.0, palm_size*1.2, 0.0],
            [0.0, palm_size*1.5, 0.0],
            [0.0, palm_size*1.8, 0.0],
            [0.0, palm_size*2.0, 0.0],
            # Ring finger
            [-palm_size*0.25, palm_size*1.1, 0.0],
            [-palm_size*0.25, palm_size*1.4, 0.0],
            [-palm_size*0.25, palm_size*1.7, 0.0],
            [-palm_size*0.25, palm_size*1.9, 0.0],
            # Pinky finger
            [-palm_size*0.45, palm_size*0.9, 0.0],
            [-palm_size*0.45, palm_size*1.2, 0.0],
            [-palm_size*0.45, palm_size*1.5, 0.0],
            [-palm_size*0.45, palm_size*1.7, 0.0],
        ]
        
        for i, pos in enumerate(joint_positions):
            if i + 1 < 21:
                joints[:, i + 1] = torch.tensor(pos, device=device)
        
        # Apply betas influence with more realistic effects
        for i in range(min(10, betas.shape[1])):
            beta_val = betas[:, i:i+1]  # [batch_size, 1]
            
            if i == 0:  # Overall scale
                scale_factor = 1.0 + beta_val * 0.2
                joints = joints * scale_factor.unsqueeze(-1)
                vertices = vertices * scale_factor.unsqueeze(-1)
            elif i == 1:  # Hand width
                joints[:, :, 0] *= (1.0 + beta_val * 0.15).squeeze(-1).unsqueeze(-1)
            elif i == 2:  # Finger length
                joints[:, 1:, 1] *= (1.0 + beta_val * 0.2).squeeze(-1).unsqueeze(-1)
            elif i == 3:  # Palm thickness
                joints[:, :, 2] *= (1.0 + beta_val * 0.1).squeeze(-1).unsqueeze(-1)
            # Additional shape variations can be added
        
        # Apply hand pose influence (simplified)
        if hand_pose.shape[1] >= 45:
            pose_reshaped = hand_pose.view(batch_size, 15, 3)
            
            for finger in range(5):
                for joint in range(3):
                    pose_idx = finger * 3 + joint
                    if pose_idx < 15:
                        joint_id = 1 + finger * 4 + joint
                        if joint_id < 21:
                            # Apply finger bending
                            bend = pose_reshaped[:, pose_idx, 0] * 0.02
                            joints[:, joint_id, 2] -= bend
        
        return type('MANOOutput', (), {
            'vertices': vertices,
            'joints': joints,
            'full_pose': torch.cat([global_orient, hand_pose], dim=-1)
        })()

def get_mano_model():
    """Get MANO model (real or mock)"""
    if MANO_AVAILABLE:
        try:
            model = manopth.manolayer.ManoLayer(
                mano_root="./mano",
                use_pca=False,
                ncomps=45,
                flat_hand_mean=False
            )
            print("✅ Real MANO model loaded")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load real MANO: {e}")
    
    print("🤖 Using mock MANO model")
    return MockMANO()

def get_inspire_hand_fk():
    """Get Inspire Hand forward kinematics (mock for basic version)"""
    urdf_path = find_inspire_hand_urdf()
    
    if urdf_path:
        print(f"📁 URDF found but using mock FK in basic version: {urdf_path}")
    else:
        print("🤖 Using mock FK - no URDF found")
    
    return None  # Basic version always uses mock

def get_hand_correspondences():
    """
    Define correspondences between MANO and Inspire Hand joints
    Returns indices for matching joints
    """
    # MANO joint names (21 joints)
    mano_joint_names = [
        "wrist",
        "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip", 
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]
    
    # Inspire Hand main joints (subset that corresponds to MANO)
    # Note: Actual joint names depend on the URDF structure
    inspire_hand_pick = [
        "wrist",  # base
        "thumb_proximal", "thumb_middle", "thumb_distal", "thumb_tip",
        "index_proximal", "index_middle", "index_distal", "index_tip",
        "middle_proximal", "middle_middle", "middle_distal", "middle_tip", 
        "ring_proximal", "ring_middle", "ring_distal", "ring_tip",
        "pinky_proximal", "pinky_middle", "pinky_distal", "pinky_tip"
    ]
    
    # For now, use 1:1 correspondence (21 joints each)
    mano_joint_pick_idx = list(range(21))
    inspire_joint_pick_idx = list(range(min(21, len(inspire_hand_pick))))
    
    return mano_joint_pick_idx, inspire_joint_pick_idx, mano_joint_names, inspire_hand_pick

def compute_inspire_hand_fk(inspire_hand, joint_angles=None):
    """
    Compute forward kinematics for Inspire Hand
    Returns joint positions in world coordinates
    """
    if inspire_hand is None:
        # Mock FK for when actual hand is not available
        cprint("Using mock Inspire Hand FK", "yellow")
        batch_size = 1 if joint_angles is None else joint_angles.shape[0]
        device = "cpu" if joint_angles is None else joint_angles.device
        
        # Create mock hand positions
        mock_joints = torch.zeros(batch_size, 21, 3, device=device)
        # Wrist at origin
        mock_joints[:, 0] = torch.tensor([0, 0, 0])
        
        # Fingers extending from wrist
        for i in range(1, 21):
            finger_idx = (i - 1) // 4
            joint_idx = (i - 1) % 4
            mock_joints[:, i, 0] = finger_idx * 0.02 - 0.04  # spread fingers
            mock_joints[:, i, 1] = joint_idx * 0.03  # extend along fingers  
            mock_joints[:, i, 2] = 0
            
        return mock_joints
    
    # TODO: Implement actual FK using inspire_hand
    # This would depend on the specific FK implementation available
    cprint("TODO: Implement actual Inspire Hand FK", "yellow")
    return torch.zeros(1, 21, 3)

def main():
    """Main shape fitting function"""
    cprint("🤖 Starting MANO to Inspire Hand Shape Optimization", "blue")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cprint(f"Using device: {device}", "green")
    
    # Get models
    mano_model = get_mano_model()
    inspire_hand = get_inspire_hand_fk()
    
    # Get joint correspondences
    mano_joint_pick_idx, inspire_joint_pick_idx, mano_names, inspire_names = get_hand_correspondences()
    
    cprint(f"Matching {len(mano_joint_pick_idx)} MANO joints with {len(inspire_joint_pick_idx)} Inspire Hand joints", "cyan")
    
    #### Preparing fitting variables
    batch_size = 1
    
    # MANO default pose (open hand)
    global_orient = torch.zeros(batch_size, 3, device=device)  # wrist orientation
    hand_pose = torch.zeros(batch_size, 45, device=device)     # finger poses (15 joints * 3 DOF)
    
    # Inspire Hand default pose (open hand)
    inspire_joint_angles = torch.zeros(batch_size, 20, device=device)  # Adjust based on actual DOF count
    
    # Shape parameters to optimize
    betas = torch.zeros(batch_size, 10, device=device)  # MANO shape parameters
    scale = torch.ones(batch_size, device=device)       # Global scale factor
    
    # MANO forward pass with default parameters
    if hasattr(mano_model, 'forward'):
        mano_output = mano_model.forward(
            global_orient=global_orient,
            hand_pose=hand_pose, 
            betas=betas
        )
        mano_joints_default = mano_output.joints
    else:
        mano_joints_default = mano_model(global_orient, hand_pose, betas).joints
    
    # Inspire Hand forward pass
    inspire_joints_default = compute_inspire_hand_fk(inspire_hand, inspire_joint_angles)
    
    cprint(f"MANO joints shape: {mano_joints_default.shape}", "blue")
    cprint(f"Inspire joints shape: {inspire_joints_default.shape}", "blue")
    
    ###### Shape fitting
    # Convert to learnable parameters
    shape_params = Variable(torch.zeros(batch_size, 10, device=device), requires_grad=True)
    scale_param = Variable(torch.ones(batch_size, device=device), requires_grad=True)
    
    # Optimizer
    optimizer_shape = torch.optim.Adam([shape_params, scale_param], lr=0.01)
    
    # Fitting loop
    cprint("🔄 Starting optimization...", "yellow")
    
    best_loss = float('inf')
    best_params = None
    
    for iteration in tqdm(range(1000), desc="Optimizing shape"):
        # MANO forward pass with current shape
        if hasattr(mano_model, 'forward'):
            mano_output = mano_model.forward(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=shape_params
            )
            mano_joints = mano_output.joints
        else:
            mano_joints = mano_model(global_orient, hand_pose, shape_params).joints
        
        # Apply scale and center on wrist
        wrist_pos = mano_joints[:, 0:1]  # wrist position
        mano_joints_scaled = (mano_joints - wrist_pos) * scale_param.unsqueeze(-1).unsqueeze(-1) + wrist_pos
        
        # Inspire Hand forward pass (static for now)
        inspire_joints = compute_inspire_hand_fk(inspire_hand, inspire_joint_angles)
        
        # Compute correspondence loss
        num_joints = min(len(mano_joint_pick_idx), len(inspire_joint_pick_idx))
        mano_subset = mano_joints_scaled[:, mano_joint_pick_idx[:num_joints]]
        inspire_subset = inspire_joints[:, inspire_joint_pick_idx[:num_joints]]
        
        # Joint position loss
        diff = mano_subset - inspire_subset
        loss_joints = diff.norm(dim=-1).mean()
        
        # Regularization losses
        loss_shape_reg = (shape_params ** 2).mean() * 0.01  # shape regularization
        loss_scale_reg = ((scale_param - 1.0) ** 2).mean() * 0.1  # scale regularization
        
        # Total loss
        loss = loss_joints + loss_shape_reg + loss_scale_reg
        
        # Backward pass
        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()
        
        # Track best parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {
                'shape': shape_params.detach().clone(),
                'scale': scale_param.detach().clone()
            }
        
        # Logging
        if iteration % 100 == 0:
            cprint(f"Iter {iteration:4d} | Loss: {loss.item():.6f} | "
                   f"Joint: {loss_joints.item():.6f} | "
                   f"Shape: {loss_shape_reg.item():.6f} | "
                   f"Scale: {loss_scale_reg.item():.6f}", "green")
    
    # Save results
    os.makedirs("data/inspire_hand", exist_ok=True)
    output_path = "data/inspire_hand/mano_shape_optimized.pkl"
    
    save_data = {
        'shape_params': best_params['shape'].cpu(),
        'scale_param': best_params['scale'].cpu(),
        'loss': best_loss,
        'mano_joint_names': mano_names,
        'inspire_joint_names': inspire_names,
        'correspondence_indices': {
            'mano': mano_joint_pick_idx,
            'inspire': inspire_joint_pick_idx
        }
    }
    
    joblib.dump(save_data, output_path)
    
    cprint(f"✅ Shape optimization completed!", "green")
    cprint(f"📁 Results saved to: {output_path}", "blue")
    cprint(f"🎯 Final loss: {best_loss:.6f}", "cyan")
    cprint(f"📏 Optimized scale: {best_params['scale'].item():.4f}", "cyan")
    cprint(f"🎨 Shape params norm: {best_params['shape'].norm().item():.4f}", "cyan")

if __name__ == "__main__":
    main() 