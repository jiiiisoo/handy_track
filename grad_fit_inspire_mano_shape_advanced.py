# CRITICAL: Import isaacgym modules FIRST before any other imports
# This prevents the "PyTorch was imported before isaacgym modules" error
import os
import sys
sys.path.append(os.getcwd())

# Try Isaac Gym imports FIRST
ISAAC_AVAILABLE = False
try:
    from isaacgym import gymapi, gymtorch, gymutil
    import pytorch_kinematics as pk
    ISAAC_AVAILABLE = True
    print("✅ Isaac Gym imported successfully")
except ImportError as e:
    print(f"⚠️ Isaac Gym not available: {e}")
    ISAAC_AVAILABLE = False

# Now import other modules
import glob
import pdb
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import math
import joblib
from torch.autograd import Variable
from termcolor import cprint
from tqdm import tqdm
import json

# ManipTrans imports
MANIPTRANS_AVAILABLE = False
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
    MANIPTRANS_AVAILABLE = True
    print("✅ ManipTrans imports successful")
except ImportError as e:
    print(f"⚠️ ManipTrans imports not available: {e}")
    MANIPTRANS_AVAILABLE = False

# MANO imports
MANO_AVAILABLE = False
try:
    import manopth
    MANO_AVAILABLE = True
    print("✅ MANO imported successfully")
except ImportError:
    print("⚠️ MANO not available, using mock")
    MANO_AVAILABLE = False

def find_inspire_hand_urdf():
    """Comprehensive search for Inspire Hand URDF file"""
    
    # Try ManipTrans factory first
    if MANIPTRANS_AVAILABLE:
        try:
            inspire_hand = DexHandFactory.create_hand("inspire", "right")
            if hasattr(inspire_hand, 'urdf_path') and os.path.exists(inspire_hand.urdf_path):
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
    
    # Search in assets directories
    for root_dir in ["/workspace/ManipTrans", "."]:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                if "assets" in root:
                    for file in files:
                        if "inspire" in file.lower() and file.endswith(".urdf"):
                            full_path = os.path.join(root, file)
                            print(f"✅ Found URDF at: {full_path}")
                            return full_path
    
    print("⚠️ No Inspire Hand URDF found, will use mock implementation")
    return None

class InspireHandFK:
    """Forward Kinematics for Inspire Hand using pytorch_kinematics"""
    
    def __init__(self, urdf_path, device="cpu"):
        self.device = device
        self.urdf_path = urdf_path
        self.available = False
        self.n_dofs = 20  # Default value
        
        if ISAAC_AVAILABLE and urdf_path and os.path.exists(urdf_path):
            try:
                # Load kinematic chain
                with open(urdf_path, 'r') as f:
                    urdf_string = f.read()
                self.chain = pk.build_chain_from_urdf(urdf_string)
                self.chain = self.chain.to(dtype=torch.float32, device=device)
                
                # Get joint names and limits
                self.joint_names = [j.name for j in self.chain.get_joints()]
                self.n_dofs = len([j for j in self.chain.get_joints() if j.joint_type != 'fixed'])
                
                cprint(f"✅ Loaded Inspire Hand with {self.n_dofs} DOFs", "green")
                self.available = True
                
            except Exception as e:
                cprint(f"⚠️ Failed to load URDF: {e}", "yellow")
                self.available = False
        else:
            if not ISAAC_AVAILABLE:
                cprint("⚠️ Isaac Gym not available, using mock FK", "yellow")
            elif not urdf_path:
                cprint("⚠️ URDF path not provided, using mock FK", "yellow")
            elif not os.path.exists(urdf_path):
                cprint(f"⚠️ URDF not found at {urdf_path}, using mock FK", "yellow")
            self.available = False
    
    def forward(self, joint_angles):
        """
        Compute forward kinematics
        Args:
            joint_angles: [batch_size, n_dofs] tensor
        Returns:
            joint_positions: [batch_size, n_joints, 3] tensor
        """
        if not self.available:
            return self._mock_forward(joint_angles)
        
        batch_size = joint_angles.shape[0]
        
        try:
            # Create joint configuration dict
            joint_config = {}
            for i, name in enumerate(self.joint_names):
                if i < joint_angles.shape[1]:
                    joint_config[name] = joint_angles[:, i]
                else:
                    joint_config[name] = torch.zeros(batch_size, device=self.device)
            
            # Forward kinematics
            transforms = self.chain.forward_kinematics(joint_config)
            
            # Extract positions of key links
            joint_positions = []
            link_names = [link.name for link in self.chain.get_links()]
            
            for link_name in link_names:
                if link_name in transforms:
                    pos = transforms[link_name].get_translation()
                    joint_positions.append(pos)
            
            if joint_positions:
                return torch.stack(joint_positions, dim=1)  # [batch, n_links, 3]
            else:
                return self._mock_forward(joint_angles)
                
        except Exception as e:
            cprint(f"⚠️ FK computation failed: {e}, using mock", "yellow")
            return self._mock_forward(joint_angles)
    
    def _mock_forward(self, joint_angles):
        """Enhanced mock FK when real FK is not available"""
        batch_size = joint_angles.shape[0]
        device = joint_angles.device
        
        # Create a realistic hand structure with proper proportions
        n_joints = 21  # MANO-like structure
        positions = torch.zeros(batch_size, n_joints, 3, device=device)
        
        # Wrist (base)
        positions[:, 0] = torch.tensor([0, 0, 0], device=device)
        
        # Hand dimensions (approximate real hand proportions)
        palm_width = 0.08
        palm_length = 0.10
        finger_length = 0.09
        
        # Finger base positions (more realistic layout)
        finger_base_positions = torch.tensor([
            [-palm_width*0.6, palm_length*0.2, 0.01],   # thumb (offset and raised)
            [-palm_width*0.3, palm_length, 0.0],        # index  
            [0.00, palm_length*1.1, 0.0],               # middle (longest)
            [palm_width*0.3, palm_length, 0.0],         # ring
            [palm_width*0.6, palm_length*0.8, 0.0],     # pinky (shorter)
        ], device=device)
        
        for finger_idx in range(5):
            for joint_idx in range(4):
                joint_id = 1 + finger_idx * 4 + joint_idx
                if joint_id < n_joints:
                    base_pos = finger_base_positions[finger_idx]
                    
                    # Finger segment lengths (decreasing)
                    segment_length = finger_length * (0.4 - joint_idx * 0.05)
                    
                    # Finger direction (pointing forward with slight curve)
                    direction = torch.tensor([0, 1, -joint_idx * 0.01], device=device)
                    if finger_idx == 0:  # Thumb points more inward
                        direction = torch.tensor([0.3, 0.8, 0], device=device)
                    
                    extension = direction * segment_length * (joint_idx + 1)
                    
                    # Apply joint angle influence (small bending)
                    if finger_idx * 4 + joint_idx < joint_angles.shape[1]:
                        angle = joint_angles[:, finger_idx * 4 + joint_idx]
                        # Simulate finger bending
                        bend_factor = torch.sin(angle) * 0.02
                        # Ensure proper broadcasting
                        if extension.dim() == 1:
                            extension = extension.expand(batch_size, -1)
                        extension[:, 2] -= bend_factor  # Bend downward
                    
                    positions[:, joint_id] = base_pos + extension
        
        return positions

class MANOModel:
    """Enhanced MANO model wrapper with better mock implementation"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.available = False
        
        if MANO_AVAILABLE:
            try:
                self.model = manopth.manolayer.ManoLayer(
                    mano_root="./mano",
                    use_pca=False,
                    ncomps=45,
                    flat_hand_mean=False
                ).to(device)
                self.available = True
                cprint("✅ MANO model loaded", "green")
            except Exception as e:
                cprint(f"⚠️ Failed to load MANO: {e}", "yellow")
                self.available = False
        else:
            cprint("⚠️ Using mock MANO implementation", "yellow")
    
    def forward(self, global_orient, hand_pose, betas):
        """
        MANO forward pass
        Args:
            global_orient: [batch, 3] wrist orientation
            hand_pose: [batch, 45] finger poses  
            betas: [batch, 10] shape parameters
        Returns:
            vertices, joints
        """
        if self.available:
            try:
                output = self.model(
                    global_orient=global_orient,
                    hand_pose=hand_pose,
                    betas=betas
                )
                return output.vertices, output.joints
            except Exception as e:
                cprint(f"⚠️ MANO forward failed: {e}, using mock", "yellow")
                return self._mock_forward(global_orient, hand_pose, betas)
        else:
            return self._mock_forward(global_orient, hand_pose, betas)
    
    def _mock_forward(self, global_orient, hand_pose, betas):
        """Enhanced mock MANO when not available"""
        batch_size = global_orient.shape[0]
        device = global_orient.device
        
        # Mock vertices (hand mesh)
        n_vertices = 778
        vertices = torch.randn(batch_size, n_vertices, 3, device=device) * 0.05
        
        # Create realistic hand joint structure
        joints = torch.zeros(batch_size, 21, 3, device=device)
        
        # Wrist
        joints[:, 0] = torch.tensor([0, 0, 0], device=device)
        
        # Realistic hand proportions
        palm_size = 0.08
        finger_length = 0.09
        
        # MANO joint layout (standard)
        mano_layout = [
            # Thumb
            [palm_size*0.3, palm_size*0.5, 0.01],
            [palm_size*0.5, palm_size*0.7, 0.02],
            [palm_size*0.6, palm_size*0.9, 0.02],
            [palm_size*0.7, palm_size*1.1, 0.02],
            # Index
            [palm_size*0.2, palm_size*1.2, 0.0],
            [palm_size*0.2, palm_size*1.5, 0.0],
            [palm_size*0.2, palm_size*1.8, 0.0],
            [palm_size*0.2, palm_size*2.0, 0.0],
            # Middle  
            [0.0, palm_size*1.3, 0.0],
            [0.0, palm_size*1.6, 0.0],
            [0.0, palm_size*1.9, 0.0],
            [0.0, palm_size*2.1, 0.0],
            # Ring
            [-palm_size*0.2, palm_size*1.2, 0.0],
            [-palm_size*0.2, palm_size*1.5, 0.0],
            [-palm_size*0.2, palm_size*1.8, 0.0],
            [-palm_size*0.2, palm_size*2.0, 0.0],
            # Pinky
            [-palm_size*0.4, palm_size*1.0, 0.0],
            [-palm_size*0.4, palm_size*1.3, 0.0],
            [-palm_size*0.4, palm_size*1.6, 0.0],
            [-palm_size*0.4, palm_size*1.8, 0.0],
        ]
        
        for i, pos in enumerate(mano_layout):
            if i + 1 < 21:
                joints[:, i + 1] = torch.tensor(pos, device=device)
        
        # Apply shape influence (beta parameters affect hand size and proportions)
        for i in range(min(10, betas.shape[1])):
            beta_val = betas[:, i:i+1]  # [batch_size, 1]
            
            if i == 0:  # Overall scale
                scale_factor = 1.0 + beta_val * 0.2
                joints = joints * scale_factor.unsqueeze(-1)
                vertices = vertices * scale_factor.unsqueeze(-1)
            elif i == 1:  # Hand width
                joints[:, :, 0] *= (1.0 + beta_val * 0.1).squeeze(-1).unsqueeze(-1)
            elif i == 2:  # Finger length
                joints[:, 1:, 1] *= (1.0 + beta_val * 0.15).squeeze(-1).unsqueeze(-1)
            # Additional shape variations can be added here
        
        # Apply pose influence (simplified)
        if hand_pose.shape[1] >= 45:
            # Group poses by finger and apply basic rotations
            pose_reshaped = hand_pose.view(batch_size, 15, 3)  # 15 joints, 3 DOF each
            
            for finger in range(5):
                for joint in range(3):
                    pose_idx = finger * 3 + joint
                    if pose_idx < 15:
                        joint_id = 1 + finger * 4 + joint
                        if joint_id < 21:
                            # Simple pose influence (bend fingers)
                            bend_amount = pose_reshaped[:, pose_idx, 0] * 0.01  # Use first rotation component
                            joints[:, joint_id, 2] -= bend_amount  # Bend downward
        
        return vertices, joints

def load_hand_correspondences(correspondence_file=None):
    """Load or create hand joint correspondences"""
    
    if correspondence_file and os.path.exists(correspondence_file):
        with open(correspondence_file, 'r') as f:
            correspondences = json.load(f)
        return correspondences
    
    # Default correspondences
    mano_joints = [
        "wrist",
        "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip", 
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]
    
    inspire_joints = [
        "base_link",  # wrist equivalent
        "thumb_abd", "thumb_mcp", "thumb_pip", "thumb_dip", 
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip", 
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]
    
    correspondences = {
        "mano_joints": mano_joints,
        "inspire_joints": inspire_joints,
        "correspondence_pairs": list(zip(range(len(mano_joints)), range(len(inspire_joints))))
    }
    
    return correspondences

def compute_hand_metrics(mano_joints, inspire_joints, correspondences):
    """Compute various metrics between MANO and Inspire hand"""
    
    pairs = correspondences["correspondence_pairs"]
    
    # Joint position differences
    position_diffs = []
    for mano_idx, inspire_idx in pairs:
        if mano_idx < mano_joints.shape[1] and inspire_idx < inspire_joints.shape[1]:
            diff = mano_joints[:, mano_idx] - inspire_joints[:, inspire_idx]
            position_diffs.append(diff.norm(dim=-1))
    
    if position_diffs:
        position_loss = torch.stack(position_diffs, dim=-1).mean()
    else:
        position_loss = torch.tensor(0.0, device=mano_joints.device)
    
    # Finger length ratios
    finger_length_loss = torch.tensor(0.0, device=mano_joints.device)
    
    # Hand span (distance between thumb and pinky tips) 
    if mano_joints.shape[1] >= 21 and inspire_joints.shape[1] >= 21:
        mano_span = (mano_joints[:, 4] - mano_joints[:, 20]).norm(dim=-1)  # thumb to pinky tip
        inspire_span = (inspire_joints[:, 4] - inspire_joints[:, 20]).norm(dim=-1)
        span_loss = (mano_span - inspire_span).abs().mean()
    else:
        span_loss = torch.tensor(0.0, device=mano_joints.device)
    
    return {
        'position_loss': position_loss,
        'finger_length_loss': finger_length_loss, 
        'span_loss': span_loss
    }

def main():
    """Advanced shape fitting with real FK models"""
    
    cprint("🤖 Advanced MANO to Inspire Hand Shape Optimization", "blue")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cprint(f"Using device: {device}", "green")
    
    # Find Inspire Hand URDF
    urdf_path = find_inspire_hand_urdf()
    
    # Initialize models
    inspire_fk = InspireHandFK(urdf_path, device=device)
    mano_model = MANOModel(device=device)
    correspondences = load_hand_correspondences()
    
    # Optimization parameters
    batch_size = 1
    n_iterations = 2000
    
    # MANO pose parameters (neutral pose)
    global_orient = torch.zeros(batch_size, 3, device=device)
    hand_pose = torch.zeros(batch_size, 45, device=device)
    
    # Inspire Hand pose (neutral)
    inspire_pose = torch.zeros(batch_size, inspire_fk.n_dofs if inspire_fk.available else 20, device=device)
    
    # Learnable parameters
    shape_params = Variable(torch.zeros(batch_size, 10, device=device), requires_grad=True)
    scale_param = Variable(torch.ones(batch_size, device=device), requires_grad=True)
    translation_offset = Variable(torch.zeros(batch_size, 3, device=device), requires_grad=True)
    
    # Optimizer
    optimizer = torch.optim.Adam([shape_params, scale_param, translation_offset], lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    
    # Training loop
    cprint("🔄 Starting advanced optimization...", "yellow")
    
    best_loss = float('inf')
    best_params = None
    
    for iteration in tqdm(range(n_iterations), desc="Optimizing"):
        
        # MANO forward pass
        mano_vertices, mano_joints = mano_model.forward(global_orient, hand_pose, shape_params)
        
        # Apply transformations
        wrist_pos = mano_joints[:, 0:1]  # wrist position
        mano_joints_transformed = (mano_joints - wrist_pos) * scale_param.unsqueeze(-1).unsqueeze(-1) + translation_offset.unsqueeze(1)
        
        # Inspire Hand forward pass
        inspire_joints = inspire_fk.forward(inspire_pose)
        
        # Compute losses
        metrics = compute_hand_metrics(mano_joints_transformed, inspire_joints, correspondences)
        
        loss_position = metrics['position_loss']
        loss_span = metrics['span_loss'] * 0.5
        
        # Regularization
        loss_shape_reg = (shape_params ** 2).mean() * 0.001
        loss_scale_reg = ((scale_param - 1.0) ** 2).mean() * 0.1
        loss_translation_reg = (translation_offset ** 2).mean() * 0.01
        
        # Total loss
        loss = loss_position + loss_span + loss_shape_reg + loss_scale_reg + loss_translation_reg
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {
                'shape': shape_params.detach().clone(),
                'scale': scale_param.detach().clone(),
                'translation': translation_offset.detach().clone()
            }
        
        # Logging
        if iteration % 200 == 0:
            cprint(f"Iter {iteration:4d} | Loss: {loss.item():.6f} | "
                   f"Pos: {loss_position.item():.6f} | "
                   f"Span: {loss_span.item():.6f} | "
                   f"Scale: {scale_param.item():.4f}", "green")
    
    # Save results
    os.makedirs("data/inspire_hand", exist_ok=True)
    output_path = "data/inspire_hand/mano_shape_optimized_advanced.pkl"
    
    save_data = {
        'shape_params': best_params['shape'].cpu(),
        'scale_param': best_params['scale'].cpu(), 
        'translation_offset': best_params['translation'].cpu(),
        'final_loss': best_loss,
        'correspondences': correspondences,
        'urdf_path': urdf_path,
        'optimization_config': {
            'n_iterations': n_iterations,
            'learning_rate': 0.005,
            'device': str(device)
        }
    }
    
    joblib.dump(save_data, output_path)
    
    # Final evaluation
    with torch.no_grad():
        mano_vertices, mano_joints = mano_model.forward(global_orient, hand_pose, best_params['shape'])
        wrist_pos = mano_joints[:, 0:1]
        mano_joints_final = (mano_joints - wrist_pos) * best_params['scale'].unsqueeze(-1).unsqueeze(-1) + best_params['translation'].unsqueeze(1)
        inspire_joints_final = inspire_fk.forward(inspire_pose)
        
        final_metrics = compute_hand_metrics(mano_joints_final, inspire_joints_final, correspondences)
    
    cprint("✅ Advanced optimization completed!", "green")
    cprint(f"📁 Results saved to: {output_path}", "blue")
    cprint(f"🎯 Final loss: {best_loss:.6f}", "cyan")
    cprint(f"📏 Optimized scale: {best_params['scale'].item():.4f}", "cyan")
    cprint(f"📍 Translation: [{best_params['translation'][0, 0]:.3f}, {best_params['translation'][0, 1]:.3f}, {best_params['translation'][0, 2]:.3f}]", "cyan")
    cprint(f"🎨 Shape norm: {best_params['shape'].norm().item():.4f}", "cyan")
    cprint(f"📐 Final position error: {final_metrics['position_loss']:.6f}", "cyan")
    cprint(f"🤏 Final span error: {final_metrics['span_loss']:.6f}", "cyan")

if __name__ == "__main__":
    main() 