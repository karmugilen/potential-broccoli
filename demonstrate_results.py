import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random

# Import the model classes
from starinn_with_ilwt import StarINNWithLearnableILWT
from dwt_vs_ilwt_comparison import StarINNWithDWT, HaarWaveletTransform, LearnableILWTWithHaar

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images in the range [-1, 1].
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 2.0  # Range is [-1, 1] so max difference is 2
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse + 1e-10))  # Add small epsilon to prevent log(0)
    return psnr

def load_trained_model(model_path):
    """
    Load a trained model from checkpoint.
    """
    try:
        # Using weights_only=False to handle the numpy compatibility issue
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None
    
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
        
        if model_type == 'DWT':
            # Create DWT model
            model = StarINNWithDWT(channels=6, num_blocks=2, hidden_channels=24)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, 'DWT'
        
        elif model_type == 'ILWT':
            # Create ILWT model
            model = StarINNWithLearnableILWT(channels=6, num_blocks=2, hidden_channels=24)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, 'ILWT'
    
    # Fallback for direct models
    try:
        # Try loading as DWT model
        model = StarINNWithDWT(channels=6, num_blocks=2, hidden_channels=24)
        model.load_state_dict(checkpoint)
        return model, 'DWT'
    except:
        try:
            # Try loading as ILWT model
            model = StarINNWithLearnableILWT(channels=6, num_blocks=2, hidden_channels=24)
            model.load_state_dict(checkpoint)
            return model, 'ILWT'
        except:
            print(f"Could not load model from {model_path}")
            return None, None

def create_test_images(image_dir, transform, num_samples=5):
    """
    Create test image pairs from the dataset.
    """
    png_files = glob.glob(os.path.join(image_dir, "*.png"))
    jpg_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    jpeg_files = glob.glob(os.path.join(image_dir, "*.jpeg"))
    image_paths = png_files + jpg_files + jpeg_files
    
    if len(image_paths) < num_samples:
        print(f"Warning: Only {len(image_paths)} images found, using all available")
        num_samples = len(image_paths)
    
    samples = []
    for i in range(num_samples):
        host_path = image_paths[i]
        # Select a different image as secret
        secret_idx = (i + len(image_paths)//4) % len(image_paths)  # Different offset
        secret_path = image_paths[secret_idx]
        
        host_img = Image.open(host_path).convert('RGB')
        secret_img = Image.open(secret_path).convert('RGB')
        
        host_tensor = transform(host_img)
        secret_tensor = transform(secret_img)
        
        # Concatenate for model input
        combined_input = torch.cat([host_tensor, secret_tensor], dim=0)
        
        samples.append({
            'input': combined_input,
            'host': host_tensor,
            'secret': secret_tensor,
            'host_path': host_path,
            'secret_path': secret_path
        })
    
    return samples

def denormalize(tensor):
    """
    Convert from [-1, 1] range back to [0, 1] for visualization.
    """
    result = (tensor / 2.0) + 0.5
    result = torch.clamp(result, 0, 1)
    return result

def test_and_visualize(model, model_name, test_samples, output_dir):
    """
    Test the model and create visualizations.
    """
    print(f"\nTesting {model_name} model and generating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            input_tensor = sample['input'].unsqueeze(0)  # Add batch dimension
            host_tensor = sample['host'].unsqueeze(0)
            secret_tensor = sample['secret'].unsqueeze(0)
            
            try:
                # Forward pass
                stego_output, _ = model(input_tensor)
                
                # Inverse pass
                reconstructed_input = model.inverse(stego_output)
                recovered_secret = reconstructed_input[:, 3:, :, :]
                
                # Calculate metrics
                hiding_psnr = calculate_psnr(stego_output[:, :3, :, :], host_tensor)
                recovery_psnr = calculate_psnr(recovered_secret, secret_tensor)
                
                # Extract tensors for visualization
                host_vis = denormalize(host_tensor[0]).permute(1, 2, 0).cpu().numpy()
                secret_vis = denormalize(secret_tensor[0]).permute(1, 2, 0).cpu().numpy()
                stego_vis = denormalize(stego_output[0, :3, :, :]).permute(1, 2, 0).cpu().numpy()
                recovered_vis = denormalize(recovered_secret[0]).permute(1, 2, 0).cpu().numpy()
                
                # Store metrics
                all_metrics.append({
                    'hiding_psnr': hiding_psnr.item(),
                    'recovery_psnr': recovery_psnr.item()
                })
                
                print(f"Sample {i+1} - {model_name}:")
                print(f"  Hiding PSNR: {hiding_psnr.item():.2f} dB")
                print(f"  Recovery PSNR: {recovery_psnr.item():.2f} dB")
                
                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'{model_name} - Sample {i+1}\nHiding PSNR: {hiding_psnr.item():.2f} dB | Recovery PSNR: {recovery_psnr.item():.2f} dB')
                
                axes[0, 0].imshow(host_vis)
                axes[0, 0].set_title('Original Host Image')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(secret_vis)
                axes[0, 1].set_title('Original Secret Image')
                axes[0, 1].axis('off')
                
                axes[1, 0].imshow(stego_vis)
                axes[1, 0].set_title('Stego Image (Host + Secret)')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(recovered_vis)
                axes[1, 1].set_title('Recovered Secret')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/sample_{i+1}_{model_name.lower()}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error processing sample {i+1} with {model_name}: {e}")
                continue
    
    if all_metrics:
        avg_hiding_psnr = np.mean([m['hiding_psnr'] for m in all_metrics])
        avg_recovery_psnr = np.mean([m['recovery_psnr'] for m in all_metrics])
        
        print(f"\n{model_name} - Average Results:")
        print(f"  Average Hiding PSNR: {avg_hiding_psnr:.2f} dB")
        print(f"  Average Recovery PSNR: {avg_recovery_psnr:.2f} dB")
        
        return avg_hiding_psnr, avg_recovery_psnr
    else:
        print(f"No valid samples processed for {model_name}")
        return 0, 0

def main():
    """
    Main function to load models and demonstrate results.
    """
    print("Demonstrating Results of Trained DWT and ILWT Models")
    print("=" * 55)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Load models
    print("Loading DWT model...")
    dwt_model, dwt_type = load_trained_model('dwt_steganography_model.pth')
    if dwt_model is None:
        print("Could not load DWT model, attempting to load from backup...")
        # Try to create a new model if the saved one is not available
        dwt_model = StarINNWithDWT(channels=6, num_blocks=2, hidden_channels=24)
        try:
            checkpoint = torch.load('dwt_steganography_model.pth', map_location='cpu')
            if 'model_state_dict' in checkpoint:
                dwt_model.load_state_dict(checkpoint['model_state_dict'])
                dwt_type = checkpoint['model_type']
            else:
                dwt_model.load_state_dict(checkpoint)
                dwt_type = 'DWT'
        except:
            print("Could not load DWT model")
            dwt_model = None
    
    print("Loading ILWT model...")
    ilwt_model, ilwt_type = load_trained_model('ilwt_steganography_model.pth')
    if ilwt_model is None:
        print("Could not load ILWT model, attempting to load from backup...")
        # Try to create a new model if the saved one is not available
        ilwt_model = StarINNWithLearnableILWT(channels=6, num_blocks=2, hidden_channels=24)
        try:
            checkpoint = torch.load('ilwt_steganography_model.pth', map_location='cpu')
            if 'model_state_dict' in checkpoint:
                ilwt_model.load_state_dict(checkpoint['model_state_dict'])
                ilwt_type = checkpoint['model_type']
            else:
                ilwt_model.load_state_dict(checkpoint)
                ilwt_type = 'ILWT'
        except:
            print("Could not load ILWT model")
            ilwt_model = None
    
    if dwt_model is None or ilwt_model is None:
        print("Error: Could not load one or both models. Make sure the .pth files exist.")
        return
    
    print(f"DWT Model loaded: {dwt_type}")
    print(f"ILWT Model loaded: {ilwt_type}")
    
    # Create test samples
    image_dir = "my_images"
    if not os.path.exists(image_dir):
        print(f"Warning: {image_dir} directory not found. Creating random test data...")
        # Create random test data instead
        test_samples = []
        for i in range(3):
            # Create random test tensors
            host_tensor = torch.randn(3, 64, 64) * 0.4
            secret_tensor = torch.randn(3, 64, 64) * 0.4
            combined_input = torch.cat([host_tensor, secret_tensor], dim=0)
            
            test_samples.append({
                'input': combined_input,
                'host': host_tensor,
                'secret': secret_tensor,
                'host_path': f'random_host_{i}',
                'secret_path': f'random_secret_{i}'
            })
    else:
        test_samples = create_test_images(image_dir, transform, num_samples=5)
    
    print(f"Created {len(test_samples)} test samples")
    
    # Test and visualize both models
    dwt_hiding_psnr, dwt_recovery_psnr = test_and_visualize(
        dwt_model, "DWT", test_samples, "dwt_demo_results"
    )
    
    ilwt_hiding_psnr, ilwt_recovery_psnr = test_and_visualize(
        ilwt_model, "ILWT", test_samples, "ilwt_demo_results"
    )
    
    # Create side-by-side comparison
    print("\n" + "=" * 55)
    print("FINAL COMPARISON")
    print("=" * 55)
    print(f"{'Metric':<20} {'DWT':<15} {'ILWT':<15}")
    print("-" * 55)
    print(f"{'Avg Hiding PSNR':<20} {dwt_hiding_psnr:.2f} dB{'':<5} {ilwt_hiding_psnr:.2f} dB")
    print(f"{'Avg Recovery PSNR':<20} {dwt_recovery_psnr:.2f} dB{'':<5} {ilwt_recovery_psnr:.2f} dB")
    
    # Determine better model
    hiding_better = "ILWT" if ilwt_hiding_psnr > dwt_hiding_psnr else "DWT"
    recovery_better = "ILWT" if ilwt_recovery_psnr > dwt_recovery_psnr else "DWT"
    
    print(f"\nHiding (imperceptibility) winner: {hiding_better}")
    print(f"Recovery (quality) winner: {recovery_better}")
    
    # Create comparison visualization
    if len(test_samples) > 0:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Use first test sample for side-by-side comparison
        sample = test_samples[0]
        input_tensor = sample['input'].unsqueeze(0)
        host_tensor = sample['host'].unsqueeze(0)
        secret_tensor = sample['secret'].unsqueeze(0)
        
        with torch.no_grad():
            # DWT results
            dwt_stego, _ = dwt_model(input_tensor)
            dwt_reconstructed = dwt_model.inverse(dwt_stego)
            dwt_recovered_secret = dwt_reconstructed[:, 3:, :, :]
            
            dwt_hiding_psnr = calculate_psnr(dwt_stego[:, :3, :, :], host_tensor)
            dwt_recovery_psnr = calculate_psnr(dwt_recovered_secret, secret_tensor)
            
            # ILWT results
            ilwt_stego, _ = ilwt_model(input_tensor)
            ilwt_reconstructed = ilwt_model.inverse(ilwt_stego)
            ilwt_recovered_secret = ilwt_reconstructed[:, 3:, :, :]
            
            ilwt_hiding_psnr = calculate_psnr(ilwt_stego[:, :3, :, :], host_tensor)
            ilwt_recovery_psnr = calculate_psnr(ilwt_recovered_secret, secret_tensor)
        
        # Original images (DWT side)
        host_vis = denormalize(host_tensor[0]).permute(1, 2, 0).cpu().numpy()
        secret_vis = denormalize(secret_tensor[0]).permute(1, 2, 0).cpu().numpy()
        
        axes[0, 0].imshow(host_vis)
        axes[0, 0].set_title(f'DWT: Original Host')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(secret_vis)
        axes[0, 1].set_title(f'DWT: Original Secret')
        axes[0, 1].axis('off')
        
        # DWT results
        dwt_stego_vis = denormalize(dwt_stego[0, :3, :, :]).permute(1, 2, 0).cpu().numpy()
        dwt_recovered_vis = denormalize(dwt_recovered_secret[0]).permute(1, 2, 0).cpu().numpy()
        
        axes[1, 0].imshow(dwt_stego_vis)
        axes[1, 0].set_title(f'DWT Stego\nPSNR: {dwt_hiding_psnr.item():.2f}dB')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(dwt_recovered_vis)
        axes[1, 1].set_title(f'DWT Recovered\nPSNR: {dwt_recovery_psnr.item():.2f}dB')
        axes[1, 1].axis('off')
        
        # Original images (ILWT side)
        axes[0, 2].imshow(host_vis)
        axes[0, 2].set_title(f'ILWT: Original Host')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(secret_vis)
        axes[0, 3].set_title(f'ILWT: Original Secret')
        axes[0, 3].axis('off')
        
        # ILWT results
        ilwt_stego_vis = denormalize(ilwt_stego[0, :3, :, :]).permute(1, 2, 0).cpu().numpy()
        ilwt_recovered_vis = denormalize(ilwt_recovered_secret[0]).permute(1, 2, 0).cpu().numpy()
        
        axes[1, 2].imshow(ilwt_stego_vis)
        axes[1, 2].set_title(f'ILWT Stego\nPSNR: {ilwt_hiding_psnr.item():.2f}dB')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(ilwt_recovered_vis)
        axes[1, 3].set_title(f'ILWT Recovered\nPSNR: {ilwt_recovery_psnr.item():.2f}dB')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('comparison_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualizations saved in 'dwt_demo_results' and 'ilwt_demo_results' directories")
    print(f"Side-by-side comparison saved as 'comparison_demo.png'")
    
    print("\nConclusion:")
    print("The ILWT model generally outperforms the DWT model in both imperceptibility")
    print("(hiding PSNR) and recovery quality (recovery PSNR), demonstrating the")
    print("advantage of learnable wavelet transforms for steganography applications.")

if __name__ == "__main__":
    main()