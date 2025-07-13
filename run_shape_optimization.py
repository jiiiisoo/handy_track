#!/usr/bin/env python3
"""
Shape Optimization Runner for MANO to Inspire Hand
Based on human2humanoid's grad_fit_h1_shape.py

Usage:
    python run_shape_optimization.py --mode basic
    python run_shape_optimization.py --mode advanced --iterations 2000 --device cuda
    python run_shape_optimization.py --load_result data/inspire_hand/mano_shape_optimized.pkl
"""

import argparse
import os
import sys
sys.path.append(os.getcwd())

import torch
import joblib
import numpy as np
from termcolor import cprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_optimization_results(result_path):
    """Visualize the optimization results"""
    
    if not os.path.exists(result_path):
        cprint(f"❌ Result file not found: {result_path}", "red")
        return
    
    cprint(f"📊 Loading results from: {result_path}", "blue")
    
    try:
        results = joblib.load(result_path)
        
        # Print summary
        cprint("\n📋 Optimization Summary:", "cyan")
        cprint(f"   Final Loss: {results.get('loss', results.get('final_loss', 'N/A')):.6f}", "green")
        cprint(f"   Scale Factor: {results['scale_param'].item():.4f}", "green")
        
        # Print shape parameters with more detail
        shape_params = results['shape_params'].numpy().flatten()
        cprint(f"   Shape Parameters (first 5): {shape_params[:5]}", "green")
        cprint(f"   Shape Parameters norm: {np.linalg.norm(shape_params):.4f}", "green")
        
        if 'translation_offset' in results:
            trans = results['translation_offset']
            cprint(f"   Translation: [{trans[0, 0]:.3f}, {trans[0, 1]:.3f}, {trans[0, 2]:.3f}]", "green")
        
        if 'optimization_config' in results:
            config = results['optimization_config']
            cprint(f"   Iterations: {config.get('n_iterations', 'N/A')}", "green")
            cprint(f"   Device: {config.get('device', 'N/A')}", "green")
            cprint(f"   Learning Rate: {config.get('learning_rate', 'N/A')}", "green")
        
        # Enhanced plotting
        fig = plt.figure(figsize=(15, 8))
        
        # Shape parameters
        plt.subplot(2, 3, 1)
        shape_params = results['shape_params'].numpy().flatten()
        bars = plt.bar(range(len(shape_params)), shape_params, 
                      color=['red' if abs(x) > 0.1 else 'blue' for x in shape_params])
        plt.title('MANO Shape Parameters (Beta)')
        plt.xlabel('Parameter Index')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, shape_params)):
            if abs(val) > 0.05:  # Only label significant values
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Scale parameter
        plt.subplot(2, 3, 2)
        scale_val = results['scale_param'].item()
        plt.bar(['Scale'], [scale_val], color='green' if 0.8 <= scale_val <= 1.2 else 'orange')
        plt.title('Scale Parameter')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal (1.0)')
        plt.text(0, scale_val + 0.02, f'{scale_val:.3f}', ha='center', va='bottom')
        plt.legend()
        
        # Translation (if available)
        if 'translation_offset' in results:
            plt.subplot(2, 3, 3)
            trans = results['translation_offset'][0].numpy()
            bars = plt.bar(['X', 'Y', 'Z'], trans, color=['red', 'green', 'blue'])
            plt.title('Translation Offset')
            plt.ylabel('Value (m)')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            for bar, val in zip(bars, trans):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.3f}', ha='center', va='bottom')
        
        # Shape parameter distribution
        plt.subplot(2, 3, 4)
        plt.hist(shape_params, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Shape Parameter Distribution')
        plt.xlabel('Parameter Value')
        plt.ylabel('Count')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero (neutral)')
        plt.legend()
        
        # Loss information (if available)
        if 'optimization_config' in results:
            plt.subplot(2, 3, 5)
            final_loss = results.get('loss', results.get('final_loss', 0))
            plt.bar(['Final Loss'], [final_loss], color='purple')
            plt.title('Optimization Result')
            plt.ylabel('Loss Value')
            plt.yscale('log')
            plt.text(0, final_loss * 1.1, f'{final_loss:.2e}', ha='center', va='bottom')
        
        # Summary stats
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Create summary text
        summary_text = f"""
Optimization Summary:
• Final Loss: {results.get('loss', results.get('final_loss', 'N/A')):.2e}
• Scale Factor: {results['scale_param'].item():.4f}
• Shape Norm: {np.linalg.norm(shape_params):.4f}
• Max |Beta|: {np.max(np.abs(shape_params)):.4f}
• Non-zero Betas: {np.sum(np.abs(shape_params) > 0.01)}/10
"""
        
        if 'translation_offset' in results:
            trans_norm = np.linalg.norm(results['translation_offset'][0].numpy())
            summary_text += f"• Translation Norm: {trans_norm:.4f}\n"
        
        if 'optimization_config' in results:
            config = results['optimization_config']
            summary_text += f"• Iterations: {config.get('n_iterations', 'N/A')}\n"
            summary_text += f"• Device: {config.get('device', 'N/A')}"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = result_path.replace('.pkl', '_visualization.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        cprint(f"📈 Visualization saved to: {plot_path}", "blue")
        
        plt.show()
        
    except Exception as e:
        cprint(f"❌ Error loading results: {e}", "red")
        import traceback
        traceback.print_exc()

def run_basic_optimization():
    """Run basic shape optimization"""
    cprint("🚀 Running basic shape optimization...", "blue")
    
    try:
        # Import here to avoid Isaac Gym conflicts
        import grad_fit_inspire_mano_shape
        
        # Run the main function
        grad_fit_inspire_mano_shape.main()
        return "data/inspire_hand/mano_shape_optimized.pkl"
        
    except ImportError as e:
        cprint(f"❌ Failed to import basic optimization: {e}", "red")
        return None
    except Exception as e:
        cprint(f"❌ Error during basic optimization: {e}", "red")
        import traceback
        traceback.print_exc()
        return None

def run_advanced_optimization(iterations=2000, device="auto"):
    """Run advanced shape optimization"""
    cprint("🚀 Running advanced shape optimization...", "blue")
    
    try:
        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Import here to handle Isaac Gym properly
        import grad_fit_inspire_mano_shape_advanced
        
        # Run the main function
        grad_fit_inspire_mano_shape_advanced.main()
        return "data/inspire_hand/mano_shape_optimized_advanced.pkl"
        
    except ImportError as e:
        cprint(f"❌ Failed to import advanced optimization: {e}", "red")
        return None
    except Exception as e:
        cprint(f"❌ Error during advanced optimization: {e}", "red")
        import traceback
        traceback.print_exc()
        return None

def check_dependencies():
    """Check if required dependencies are available"""
    cprint("🔍 Checking dependencies...", "yellow")
    
    dependencies = {
        'torch': True,
        'numpy': True,
        'termcolor': True,
        'tqdm': True,
        'joblib': True,
        'matplotlib': True
    }
    
    optional_deps = {
        'isaacgym': False,
        'pytorch_kinematics': False,
        'manopth': False
    }
    
    # Check required dependencies
    missing_required = []
    for dep in dependencies:
        try:
            __import__(dep)
            cprint(f"   ✅ {dep}", "green")
        except ImportError:
            cprint(f"   ❌ {dep} (required)", "red")
            missing_required.append(dep)
    
    # Check optional dependencies
    cprint("\n🔍 Checking optional dependencies...", "yellow")
    for dep in optional_deps:
        try:
            if dep == 'isaacgym':
                # Special handling for Isaac Gym to avoid import conflicts
                import importlib.util
                spec = importlib.util.find_spec('isaacgym')
                if spec is not None:
                    cprint(f"   ✅ {dep} (available but not imported to avoid conflicts)", "green")
                    optional_deps[dep] = True
                else:
                    cprint(f"   ⚠️  {dep} (optional, will use mock)", "yellow")
            else:
                __import__(dep)
                cprint(f"   ✅ {dep} (available)", "green")
                optional_deps[dep] = True
        except ImportError:
            cprint(f"   ⚠️  {dep} (optional, will use mock)", "yellow")
    
    if missing_required:
        cprint(f"\n❌ Missing required dependencies: {missing_required}", "red")
        cprint("Please install them with: pip install " + " ".join(missing_required), "yellow")
        return False
    
    # Show additional info
    cprint(f"\n📋 System Information:", "cyan")
    cprint(f"   Python: {sys.version.split()[0]}", "blue")
    cprint(f"   PyTorch: {torch.__version__}", "blue")
    cprint(f"   CUDA Available: {torch.cuda.is_available()}", "blue")
    if torch.cuda.is_available():
        cprint(f"   CUDA Devices: {torch.cuda.device_count()}", "blue")
        cprint(f"   Current Device: {torch.cuda.current_device()}", "blue")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="MANO to Inspire Hand Shape Optimization")
    parser.add_argument('--mode', choices=['basic', 'advanced'], default='basic',
                       help="Optimization mode (default: basic)")
    parser.add_argument('--iterations', type=int, default=1000,
                       help="Number of optimization iterations (default: 1000)")
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help="Device to use (default: auto)")
    parser.add_argument('--load_result', type=str,
                       help="Load and visualize existing results from pickle file")
    parser.add_argument('--check_deps', action='store_true',
                       help="Check dependencies and exit")
    parser.add_argument('--visualize_only', action='store_true',
                       help="Only visualize results without running optimization")
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        check_dependencies()
        return
    
    if not check_dependencies():
        return
    
    # Load and visualize existing results
    if args.load_result:
        visualize_optimization_results(args.load_result)
        return
    
    # Only visualize default results
    if args.visualize_only:
        default_paths = [
            "data/inspire_hand/mano_shape_optimized_advanced.pkl",
            "data/inspire_hand/mano_shape_optimized.pkl"
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                visualize_optimization_results(path)
                return
        
        cprint("❌ No default result files found", "red")
        cprint("Available files in data/inspire_hand/:", "yellow")
        if os.path.exists("data/inspire_hand/"):
            for f in os.listdir("data/inspire_hand/"):
                if f.endswith('.pkl'):
                    cprint(f"   - {f}", "blue")
        return
    
    # Run optimization
    cprint(f"🤖 Starting {args.mode} optimization with {args.iterations} iterations on {args.device}", "blue")
    
    result_path = None
    if args.mode == 'basic':
        result_path = run_basic_optimization()
    else:
        result_path = run_advanced_optimization(args.iterations, args.device)
    
    # Visualize results
    if result_path and os.path.exists(result_path):
        cprint(f"\n🎉 Optimization completed successfully!", "green")
        visualize_optimization_results(result_path)
    else:
        cprint("❌ Optimization failed or result file not found", "red")
        if result_path:
            cprint(f"Expected result file: {result_path}", "yellow")

if __name__ == "__main__":
    main() 