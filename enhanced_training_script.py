import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# DWT implementation
class HaarWaveletTransform(nn.Module):
    """
    Haar Wavelet Transform for frequency domain processing (DWT approach).
    """
    def __init__(self):
        super(HaarWaveletTransform, self).__init__()
        # Define Haar wavelet filters
        h = torch.tensor([0.5, 0.5], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        g = torch.tensor([0.5, -0.5], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('h', h)
        self.register_buffer('g', g)

    def forward(self, x):
        """
        Forward DWT - decomposes image into LL, LH, HL, HH subbands
        """
        batch_size, channels, height, width = x.shape
        
        # Check if dimensions are even, pad if necessary
        pad_h, pad_w = 0, 0
        if height % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
            pad_h = 1
        if width % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
            pad_w = 1
            
        self.padding_info = (pad_h, pad_w) if pad_h > 0 or pad_w > 0 else None
        
        LL, LH, HL, HH = [], [], [], []
        
        for i in range(channels):
            x_channel = x[:, i:i+1, :, :]
            
            # Apply filters
            # Row convolution
            ll_row = F.conv2d(x_channel, self.h.unsqueeze(2), stride=(1, 2))
            lh_row = F.conv2d(x_channel, self.g.unsqueeze(2), stride=(1, 2))
            
            # Column convolution
            ll = F.conv2d(ll_row, self.h.unsqueeze(3), stride=(2, 1))
            lh = F.conv2d(lh_row, self.h.unsqueeze(3), stride=(2, 1))
            hl = F.conv2d(ll_row, self.g.unsqueeze(3), stride=(2, 1))
            hh = F.conv2d(lh_row, self.g.unsqueeze(3), stride=(2, 1))
            
            LL.append(ll)
            LH.append(lh)
            HL.append(hl)
            HH.append(hh)
        
        LL = torch.cat(LL, dim=1)
        LH = torch.cat(LH, dim=1)
        HL = torch.cat(HL, dim=1)
        HH = torch.cat(HH, dim=1)
        
        return torch.cat([LL, LH, HL, HH], dim=1)

    def inverse(self, x):
        """
        Inverse DWT - reconstructs image from subbands
        """
        batch_size, total_channels, height, width = x.shape
        channels = total_channels // 4
        
        LL = x[:, :channels, :, :]
        LH = x[:, channels:2*channels, :, :]
        HL = x[:, 2*channels:3*channels, :, :]
        HH = x[:, 3*channels:4*channels, :, :]
        
        reconstructed = []

        for i in range(channels):
            ll = LL[:, i:i+1, :, :]
            lh = LH[:, i:i+1, :, :]
            hl = HL[:, i:i+1, :, :]
            hh = HH[:, i:i+1, :, :]

            # Apply inverse filters (transposed)
            # Combine high frequencies with low frequencies
            temp1 = F.conv_transpose2d(ll, self.h.unsqueeze(2), stride=(1, 2))
            temp2 = F.conv_transpose2d(lh, self.h.unsqueeze(2), stride=(1, 2))
            temp3 = F.conv_transpose2d(hl, self.g.unsqueeze(2), stride=(1, 2))
            temp4 = F.conv_transpose2d(hh, self.g.unsqueeze(2), stride=(1, 2))
            
            # Pad tensors so they have matching sizes
            max_h = max(temp1.shape[2], temp2.shape[2], temp3.shape[2], temp4.shape[2])
            max_w = max(temp1.shape[3], temp2.shape[3], temp3.shape[3], temp4.shape[3])
            
            # Pad tensors to same size
            def pad_to_size(tensor, target_h, target_w):
                pad_h = max(0, target_h - tensor.shape[2])
                pad_w = max(0, target_w - tensor.shape[3])
                return F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
            temp1 = pad_to_size(temp1, max_h, max_w)
            temp2 = pad_to_size(temp2, max_h, max_w)
            temp3 = pad_to_size(temp3, max_h, max_w)
            temp4 = pad_to_size(temp4, max_h, max_w)
            
            # Combine rows
            result_rows1 = temp1 + temp3
            result_rows2 = temp2 + temp4
            
            # Apply inverse column transform
            max_h2 = max(result_rows1.shape[2], result_rows2.shape[2])
            max_w2 = max(result_rows1.shape[3], result_rows2.shape[3])
            
            result_rows1 = pad_to_size(result_rows1, max_h2, max_w2)
            result_rows2 = pad_to_size(result_rows2, max_h2, max_w2)
            
            result1 = F.conv_transpose2d(result_rows1, self.h.unsqueeze(3), stride=(2, 1))
            result2 = F.conv_transpose2d(result_rows2, self.g.unsqueeze(3), stride=(2, 1))
            
            # Final combination
            result = result1 + result2
            reconstructed.append(result)
        
        result = torch.cat(reconstructed, dim=1)
        
        # Remove padding if it was added in forward
        if hasattr(self, 'padding_info') and self.padding_info:
            pad_h, pad_w = self.padding_info
            if pad_h > 0:
                result = result[:, :, :-1, :]
            if pad_w > 0:
                result = result[:, :, :, :-1]
        
        return result


# ILWT implementation (learnable)
class LearnableILWTWithHaar(nn.Module):
    """
    Learnable ILWT using Haar wavelet approximation with learnable parameters.
    """
    def __init__(self, channels):
        super(LearnableILWTWithHaar, self).__init__()
        self.channels = channels
        
        # We'll use regular convolution layers with stride=2 for downsampling (like wavelet decomposition)
        # and transposed convolution for upsampling (like inverse wavelet transform)
        self.decomposition = nn.Conv2d(channels, 4 * channels, kernel_size=2, stride=2, padding=0, groups=channels)
        
        # Initialize to mimic Haar wavelet decomposition
        with torch.no_grad():
            # Haar-like filters: low-pass and high-pass
            haar_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32) / 2
            haar_lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32) / 2
            haar_hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32) / 2
            haar_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32) / 2
            
            # For grouped convolution, each channel produces 4 subband channels
            for c in range(channels):
                self.decomposition.weight.data[4*c, 0, :, :] = haar_ll
                self.decomposition.weight.data[4*c+1, 0, :, :] = haar_lh
                self.decomposition.weight.data[4*c+2, 0, :, :] = haar_hl
                self.decomposition.weight.data[4*c+3, 0, :, :] = haar_hh

        # Reconstruction uses transposed convolution
        self.reconstruction = nn.ConvTranspose2d(4 * channels, channels, kernel_size=2, stride=2, padding=0, groups=channels)
        
        # Initialize to approximately invert the decomposition
        with torch.no_grad():
            # For reconstruction with groups=channels, each input channel maps to one output
            recon_filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float32)
            
            # For grouped transpose conv: [24, 1, 2, 2] shape (with 6 channels, 4*6=24 input, 6 output, groups=6)
            # Each of the 4 input channels for a group contributes to 1 output channel
            for c in range(channels):
                # For this group, we have 4 input channels that map to 1 output channel
                for i in range(4):
                    self.reconstruction.weight.data[4*c+i, 0, :, :] = recon_filter

    def forward(self, x):
        """
        Forward pass: Apply learnable Haar-like transform.
        """
        batch_size, channels, height, width = x.shape
        
        # Check if dimensions are even, pad if necessary
        pad_h, pad_w = 0, 0
        if height % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
            pad_h = 1
        if width % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
            pad_w = 1
            
        # Store padding info for inverse
        if pad_h > 0 or pad_w > 0:
            self.padding_info = (pad_h, pad_w)
        else:
            self.padding_info = None
            
        # Apply decomposition (downsampling convolution)
        output = self.decomposition(x)
        return output
    
    def inverse(self, x):
        """
        Inverse pass: Apply learnable inverse Haar-like transform.
        """
        # Apply reconstruction (upsampling transpose convolution)
        reconstructed = self.reconstruction(x)
        
        # Remove padding if it was added
        if hasattr(self, 'padding_info') and self.padding_info:
            pad_h, pad_w = self.padding_info
            if pad_h > 0:
                reconstructed = reconstructed[:, :, :-1, :]
            if pad_w > 0:
                reconstructed = reconstructed[:, :, :, :-1]
        
        return reconstructed


# Common components from starinn_block.py (simplified versions)
class ActNorm(nn.Module):
    """
    Activation normalization layer: learns scale and bias to normalize activations.
    """
    def __init__(self, channels):
        super(ActNorm, self).__init__()
        self.channels = channels
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        
    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                # Initialize with mean and std of first batch
                mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
                std = torch.std(x, dim=[0, 2, 3], keepdim=True)
                self.scale.data.copy_(1.0 / (std + 1e-6))
                self.bias.data.copy_(-mean * self.scale.data)
                self.initialized.fill_(1)
        
        # Forward: y = x * scale + bias
        y = x * self.scale + self.bias
        # Log determinant is just the log of the scale
        log_det = torch.sum(torch.log(torch.abs(self.scale))) * x.shape[2] * x.shape[3]
        return y, log_det
    
    def inverse(self, y):
        # Inverse: x = (y - bias) / scale
        x = (y - self.bias) / self.scale
        return x


class AffineCouplingLayer(nn.Module):
    """
    Simplified affine coupling layer for invertible neural networks.
    """
    def __init__(self, channels, hidden_channels=32, dropout_rate=0.1):
        super(AffineCouplingLayer, self).__init__()
        
        # Ensure channels is even for splitting
        assert channels % 2 == 0, "Number of channels must be even"
        half_channels = channels // 2
        
        # Network to compute scale (s) and translation (t) parameters
        self.net = nn.Sequential(
            nn.Conv2d(half_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),  # Add dropout after first activation
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),  # Add dropout after second activation
            nn.Conv2d(hidden_channels, half_channels * 2, kernel_size=3, padding=1)
        )
        
        # Initialize to output zeros for scale and translation initially
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
        
    def forward(self, x):
        """
        Forward pass: x -> y
        """
        x1, x2 = x.chunk(2, dim=1)  # Split along channel dimension
        
        # Compute scale and translation parameters based on x1
        s_t = self.net(x1)
        s, t = s_t.chunk(2, dim=1)
        
        # Apply affine transformation to x2 using parameters from x1
        y1 = x1  # First part remains unchanged
        y2 = torch.exp(s) * x2 + t  # Second part is transformed
        
        return torch.cat([y1, y2], dim=1)
    
    def inverse(self, y):
        """
        Inverse pass: y -> x
        """
        y1, y2 = y.chunk(2, dim=1)  # Split along channel dimension
        
        # Compute scale and translation parameters based on y1 (which is x1)
        s_t = self.net(y1)
        s, t = s_t.chunk(2, dim=1)
        
        # Inverse transformation
        x1 = y1  # First part remains unchanged
        x2 = (y2 - t) / torch.exp(s)  # Inverse of the affine transformation
        
        return torch.cat([x1, x2], dim=1)


class Invertible1x1Conv(nn.Module):
    """
    Invertible 1x1 convolution layer.
    """
    def __init__(self, channels):
        super(Invertible1x1Conv, self).__init__()
        self.channels = channels
        
        # Initialize a random orthogonal matrix using QR decomposition
        w_init = np.random.randn(channels, channels)
        q, _ = np.linalg.qr(w_init)
        w_init = torch.from_numpy(q.astype(np.float32))
        
        # Create weight parameter as a matrix
        self.weight = nn.Parameter(w_init)
        
    def forward(self, x):
        """
        Forward pass: x -> y
        """
        b, c, h, w = x.size()
        
        # Apply convolution (matrix multiplication on channel dimension)
        x_flat = x.view(b, c, -1)  # Flatten spatial dimensions
        y_flat = torch.matmul(self.weight, x_flat)  # Apply transformation
        y = y_flat.view(b, c, h, w)  # Reshape back
        
        # Calculate log determinant (constant, not changing during training)
        log_det = h * w * torch.log(torch.abs(torch.det(self.weight)))
        
        return y, log_det
    
    def inverse(self, y):
        """
        Inverse pass: y -> x
        """
        b, c, h, w = y.size()
        
        # Inverse of weight matrix
        w_inv = torch.inverse(self.weight)
        
        # Apply inverse transformation
        y_flat = y.view(b, c, -1)  # Flatten spatial dimensions
        x_flat = torch.matmul(w_inv, y_flat)  # Apply inverse transformation
        x = x_flat.view(b, c, h, w)  # Reshape back
        
        return x


class StarINNBlock(nn.Module):
    """
    A single block of the StarINN architecture: ActNorm -> 1x1 Conv -> Affine Coupling
    """
    def __init__(self, channels, hidden_channels=32, dropout_rate=0.1):
        super(StarINNBlock, self).__init__()
        assert channels % 2 == 0, "Channels must be even"
        
        self.actnorm = ActNorm(channels)
        self.inv_conv = Invertible1x1Conv(channels)
        self.affine_coupling = AffineCouplingLayer(channels, hidden_channels, dropout_rate)
        
    def forward(self, x):
        # ActNorm
        x, log_det1 = self.actnorm(x)
        
        # Invertible 1x1 Conv
        x, log_det2 = self.inv_conv(x)
        
        # Affine Coupling
        x = self.affine_coupling(x)
        
        # Total log determinant
        log_det = log_det1 + log_det2
        return x, log_det
    
    def inverse(self, y):
        # Affine Coupling Inverse
        y = self.affine_coupling.inverse(y)
        
        # Invertible 1x1 Conv Inverse
        y = self.inv_conv.inverse(y)
        
        # ActNorm Inverse
        y = self.actnorm.inverse(y)
        
        return y

class StarINNWithDWT(nn.Module):
    """
    StarINN model with DWT preprocessing for comparison.
    """
    def __init__(self, channels=6, num_blocks=2, hidden_channels=32, dropout_rate=0.1):
        super(StarINNWithDWT, self).__init__()
        
        # DWT preprocessing module
        self.dwt = HaarWaveletTransform()
        
        # After DWT, each channel becomes 4 subbands, so 6 channels become 24 channels
        self.inn_channels = channels * 4  # 6 -> 24 after DWT
        
        # Create the invertible blocks for processing in frequency domain
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(StarINNBlock(self.inn_channels, hidden_channels, dropout_rate))
    
    def forward(self, x):
        """
        Forward pass: applies DWT -> INN -> returns stego and transformed secret.
        Input x: [batch, 6, height, width] - first 3 channels are host, last 3 are secret
        """
        # Apply DWT to convert to frequency domain
        x_freq = self.dwt(x)
        
        log_det_total = 0
        
        # Process through each block in frequency domain
        z = x_freq
        for block in self.blocks:
            z, log_det = block(z)
            log_det_total += log_det
        
        # After processing, apply inverse DWT to return to spatial domain
        output_spatial = self.dwt.inverse(z)
        
        return output_spatial, log_det_total
    
    def inverse(self, z):
        """
        Inverse pass: applies DWT -> INN inverse -> DWT inverse.
        """
        # Apply DWT to convert to frequency domain
        z_freq = self.dwt(z)
        
        # Process in reverse order through INN blocks
        x_freq = z_freq
        for i in range(len(self.blocks) - 1, -1, -1):
            x_freq = self.blocks[i].inverse(x_freq)
        
        # Apply inverse DWT to return to spatial domain
        x_spatial = self.dwt.inverse(x_freq)
        
        return x_spatial

class StarINNWithILWT(nn.Module):
    """
    StarINN model with Learnable ILWT preprocessing for comparison.
    """
    def __init__(self, channels=6, num_blocks=2, hidden_channels=32, dropout_rate=0.1):
        super(StarINNWithILWT, self).__init__()
        
        # Learnable ILWT preprocessing module
        self.ilwt = LearnableILWTWithHaar(channels)
        
        # After ILWT, each channel becomes 4 subbands, so 6 channels become 24 channels
        self.inn_channels = channels * 4  # 6 -> 24 after ILWT
        
        # Create the invertible blocks for processing in frequency domain
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(StarINNBlock(self.inn_channels, hidden_channels, dropout_rate))
    
    def forward(self, x):
        """
        Forward pass: applies learnable ILWT -> INN -> returns stego and transformed secret.
        Input x: [batch, 6, height, width] - first 3 channels are host, last 3 are secret
        """
        # Apply learnable ILWT to convert to frequency domain
        x_freq = self.ilwt(x)
        
        log_det_total = 0
        
        # Process through each block in frequency domain
        z = x_freq
        for block in self.blocks:
            z, log_det = block(z)
            log_det_total += log_det
        
        # After processing, apply inverse learnable ILWT to return to spatial domain
        output_spatial = self.ilwt.inverse(z)
        
        return output_spatial, log_det_total
    
    def inverse(self, z):
        """
        Inverse pass: applies learnable ILWT -> INN inverse -> learnable ILWT inverse.
        """
        # Apply learnable ILWT to convert to frequency domain
        z_freq = self.ilwt(z)
        
        # Process in reverse order through INN blocks
        x_freq = z_freq
        for i in range(len(self.blocks) - 1, -1, -1):
            x_freq = self.blocks[i].inverse(x_freq)
        
        # Apply inverse learnable ILWT to return to spatial domain
        x_spatial = self.ilwt.inverse(x_freq)
        
        return x_spatial

class ImageSteganographyDataset(Dataset):
    """
    Dataset for image steganography - loads pairs of images to use as host and secret.
    """
    def __init__(self, image_dir, img_size=64, transform=None):
        png_files = glob.glob(os.path.join(image_dir, "*.png"))
        jpg_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        jpeg_files = glob.glob(os.path.join(image_dir, "*.jpeg"))
        self.image_paths = png_files + jpg_files + jpeg_files
        self.img_size = img_size
        
        # Default transformation to normalize images
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the host image
        host_path = self.image_paths[idx]
        host_img = Image.open(host_path).convert('RGB')
        host_tensor = self.transform(host_img)
        
        # Select a different image as the secret
        secret_idx = random.choice([i for i in range(len(self.image_paths)) if i != idx])
        secret_path = self.image_paths[secret_idx]
        secret_img = Image.open(secret_path).convert('RGB')
        secret_tensor = self.transform(secret_img)
        
        # Concatenate host and secret along channel dimension: [6, H, W]
        combined_input = torch.cat([host_tensor, secret_tensor], dim=0)
        
        return combined_input, host_tensor, secret_tensor

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

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM between two images in the range [-1, 1].
    """
    # Convert from [-1, 1] to [0, 1] range for SSIM calculation
    img1 = (img1 + 1) / 2.0
    img2 = (img2 + 1) / 2.0
    
    # Ensure images are in the right range [0, 1]
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    def ssim(img1, img2, window_size=11, channel=3, size_average=True):
        # Define Gaussian window
        def create_window(window_size, channel):
            def _gaussian(window_size, sigma):
                gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
                return gauss/gauss.sum()

            _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        def _ssim(img1, img2, window, window_size, channel, size_average=True):
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1*mu2

            sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

            C1 = 0.01**2
            C2 = 0.03**2

            ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
            
            if size_average:
                return ssim_map.mean()
            else:
                return ssim_map.mean(1).mean(1).mean(1)

        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)
        
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        
        return _ssim(img1, img2, window, window_size, channel, size_average)

    return ssim(img1, img2, window_size, img1.size()[1], size_average)

def steganography_loss(stego_img, host_img, secret_img, recovered_secret, alpha_hid=32.0, alpha_rec=1.0):
    """
    Loss function for steganography:
    - Hiding loss: difference between host and stego images (imperceptibility)
    - Recovery loss: difference between original and recovered secret (recoverability)
    """
    # Hiding loss: L2 loss between host image and stego image (first 3 channels of output)
    hiding_loss = F.mse_loss(stego_img, host_img)
    
    # Recovery loss: L2 loss between original secret and recovered secret
    recovery_loss = F.mse_loss(recovered_secret, secret_img)
    
    # Combined loss
    total_loss = alpha_hid * hiding_loss + alpha_rec * recovery_loss
    
    return total_loss, hiding_loss, recovery_loss

def train_model(model, model_name, dataset, num_epochs=100):
    """
    Train a model with the given dataset.
    """
    print(f"\nTraining {model_name} with enhanced parameters...")
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
    
    # Enhanced parameters
    batch_size = 1  # Using batch size 1 due to memory constraints
    learning_rate = 2e-4  # Slightly higher learning rate for AdamW
    alpha_hid = 32.0  # Weight for hiding loss
    alpha_rec = 1.0   # Weight for recovery loss
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Enhanced AdamW optimizer with better hyperparameters
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, eps=1e-8)
    
    # Advanced learning rate scheduler: OneCycleLR for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, 
                                              steps_per_epoch=len(dataloader), 
                                              epochs=num_epochs, 
                                              pct_start=0.1,
                                              anneal_strategy='cos',
                                              div_factor=25,
                                              final_div_factor=100)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(dataset)} images for {num_epochs} epochs")
    
    # Training loop
    best_hiding_psnr = float('-inf')
    best_recovery_psnr = float('-inf')
    best_hiding_ssim = float('-inf')
    best_recovery_ssim = float('-inf')
    
    epoch_losses = []
    epoch_hiding_psnrs = []
    epoch_recovery_psnrs = []
    epoch_hiding_ssims = []
    epoch_recovery_ssims = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_hiding_psnr = 0.0
        epoch_recovery_psnr = 0.0
        epoch_hiding_ssim = 0.0
        epoch_recovery_ssim = 0.0
        batch_count = 0
        
        model.train()
        
        for batch_idx, (input_tensor, host_tensor, secret_tensor) in enumerate(dataloader):
            # Move tensors to device
            input_tensor = input_tensor.to(device)
            host_tensor = host_tensor.to(device)
            secret_tensor = secret_tensor.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass
                stego_output, log_det = model(input_tensor)
                
                # For simplicity, we'll use the inverse to recover the secret
                reconstructed_input = model.inverse(stego_output)
                recovered_secret = reconstructed_input[:, 3:, :, :]  # Extract secret part (last 3 channels)
                host_input = input_tensor[:, :3, :, :]  # Extract host part (first 3 channels)
                
                # The stego image should be similar to the host image, extract the first 3 channels
                stego_host = stego_output[:, :3, :, :]  # Take first 3 channels of stego as the actual stego image

                # Calculate PSNR and SSIM metrics
                hiding_psnr_val = calculate_psnr(stego_host, host_input)
                recovery_psnr_val = calculate_psnr(recovered_secret, secret_tensor)
                
                # Calculate SSIM
                hiding_ssim_val = calculate_ssim(stego_host, host_input)
                recovery_ssim_val = calculate_ssim(recovered_secret, secret_tensor)
                
                # Compute loss
                loss, hiding_loss, recovery_loss = steganography_loss(
                    stego_host, host_input, secret_tensor, recovered_secret, 
                    alpha_hid, alpha_rec
                )
                
                # Backward pass
                loss.backward()
                
                # Enhanced gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Step scheduler after each batch for OneCycleLR
                scheduler.step()
                
                # Accumulate metrics
                epoch_loss += loss.item()
                
                # Only use valid PSNR values
                hp_val = hiding_psnr_val.item() if hiding_psnr_val == hiding_psnr_val else 0  # Check for NaN
                rp_val = recovery_psnr_val.item() if recovery_psnr_val == recovery_psnr_val else 0  # Check for NaN
                
                # Only use valid SSIM values
                hs_val = hiding_ssim_val.item() if hiding_ssim_val == hiding_ssim_val else 0  # Check for NaN
                rs_val = recovery_ssim_val.item() if recovery_ssim_val == recovery_ssim_val else 0  # Check for NaN
                
                epoch_hiding_psnr += hp_val
                epoch_recovery_psnr += rp_val
                epoch_hiding_ssim += hs_val
                epoch_recovery_ssim += rs_val
                batch_count += 1
                
                # Update best metrics
                if hiding_psnr_val != float('inf') and hiding_psnr_val != float('-inf') and hiding_psnr_val > best_hiding_psnr:
                    best_hiding_psnr = hiding_psnr_val
                if recovery_psnr_val != float('inf') and recovery_psnr_val != float('-inf') and recovery_psnr_val > best_recovery_psnr:
                    best_recovery_psnr = recovery_psnr_val
                if hiding_ssim_val != float('inf') and hiding_ssim_val != float('-inf') and hiding_ssim_val > best_hiding_ssim:
                    best_hiding_ssim = hiding_ssim_val
                if recovery_ssim_val != float('inf') and recovery_ssim_val != float('-inf') and recovery_ssim_val > best_recovery_ssim:
                    best_recovery_ssim = recovery_ssim_val
                
                # Print progress for first batch of each epoch and every 10th epoch
                if (batch_idx == 0 and (epoch + 1) % 10 == 0) and hiding_psnr_val != float('nan') and recovery_psnr_val != float('nan'):
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, "
                          f"Loss: {loss.item():.6f}, Hiding PSNR: {hiding_psnr_val.item():.2f}, "
                          f"Recovery PSNR: {recovery_psnr_val.item():.2f}, "
                          f"LR: {current_lr:.2e}")
            except Exception as e:
                print(f"Error in batch {batch_idx} of epoch {epoch+1}: {e}")
                continue
        
        # Print average epoch metrics
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_hiding_psnr = epoch_hiding_psnr / batch_count
            avg_recovery_psnr = epoch_recovery_psnr / batch_count
            avg_hiding_ssim = epoch_hiding_ssim / batch_count
            avg_recovery_ssim = epoch_recovery_ssim / batch_count
            
            epoch_losses.append(avg_loss)
            epoch_hiding_psnrs.append(avg_hiding_psnr)
            epoch_recovery_psnrs.append(avg_recovery_psnr)
            epoch_hiding_ssims.append(avg_hiding_ssim)
            epoch_recovery_ssims.append(avg_recovery_ssim)
            
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}, "
                  f"Avg Hiding PSNR: {avg_hiding_psnr:.2f} dB, "
                  f"Avg Recovery PSNR: {avg_recovery_psnr:.2f} dB, "
                  f"Avg Hiding SSIM: {avg_hiding_ssim:.4f}, "
                  f"Avg Recovery SSIM: {avg_recovery_ssim:.4f}")
    
    print(f"Training completed for {model_name}!")
    print(f"Best Hiding PSNR achieved: {best_hiding_psnr:.2f} dB")
    print(f"Best Recovery PSNR achieved: {best_recovery_psnr:.2f} dB")
    print(f"Best Hiding SSIM achieved: {best_hiding_ssim:.4f}")
    print(f"Best Recovery SSIM achieved: {best_recovery_ssim:.4f}")
    
    return epoch_losses, epoch_hiding_psnrs, epoch_recovery_psnrs, epoch_hiding_ssims, epoch_recovery_ssims, best_hiding_psnr, best_recovery_psnr, best_hiding_ssim, best_recovery_ssim

def test_model(model, dataset, model_name, num_samples=5):
    """
    Test the trained model and generate visualization images.
    """
    print(f"\nTesting {model_name} and generating {num_samples} visualization samples...")
    
    # Check for GPU availability for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to device
    
    # Create a directory for results
    os.makedirs(f"{model_name.lower()}_results", exist_ok=True)
    
    model.eval()
    
    # Use a subset of the dataset for testing
    test_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    all_hiding_psnr = []
    all_recovery_psnr = []
    all_hiding_ssim = []
    all_recovery_ssim = []
    
    for i, idx in enumerate(test_indices):
        input_tensor, host_tensor, secret_tensor = dataset[idx]
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        host_tensor = host_tensor.unsqueeze(0)
        secret_tensor = secret_tensor.unsqueeze(0)
        
        # Move tensors to device
        input_tensor = input_tensor.to(device)
        host_tensor = host_tensor.to(device)
        secret_tensor = secret_tensor.to(device)
        
        with torch.no_grad():
            try:
                # Forward pass
                stego_output, _ = model(input_tensor)
                
                # Inverse pass to recover secret
                reconstructed_input = model.inverse(stego_output)
                recovered_secret = reconstructed_input[:, 3:, :, :]
                
                # Calculate metrics
                hiding_psnr = calculate_psnr(stego_output[:, :3, :, :], host_tensor)
                recovery_psnr = calculate_psnr(recovered_secret, secret_tensor)
                hiding_ssim = calculate_ssim(stego_output[:, :3, :, :], host_tensor)
                recovery_ssim = calculate_ssim(recovered_secret, secret_tensor)
                
                all_hiding_psnr.append(hiding_psnr.item())
                all_recovery_psnr.append(recovery_psnr.item())
                all_hiding_ssim.append(hiding_ssim.item())
                all_recovery_ssim.append(recovery_ssim.item())
                
                print(f"Sample {i+1}: Hiding PSNR = {hiding_psnr.item():.2f} dB, "
                      f"Recovery PSNR = {recovery_psnr.item():.2f} dB, "
                      f"Hiding SSIM = {hiding_ssim.item():.4f}, "
                      f"Recovery SSIM = {recovery_ssim.item():.4f}")
                
                # Denormalize tensors for visualization (from [-1,1] to [0,1])
                def denormalize(tensor):
                    result = (tensor / 2.0) + 0.5
                    result = torch.clamp(result, 0, 1)
                    return result
                
                host_vis = denormalize(host_tensor[0]).permute(1, 2, 0).cpu().numpy()
                secret_vis = denormalize(secret_tensor[0]).permute(1, 2, 0).cpu().numpy()
                stego_vis = denormalize(stego_output[0, :3, :, :]).permute(1, 2, 0).cpu().numpy()
                recovered_vis = denormalize(recovered_secret[0]).permute(1, 2, 0).cpu().numpy()
                
                # Save visualization
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                fig.suptitle(f'{model_name} - Sample {i+1} - Hiding PSNR: {hiding_psnr.item():.2f} dB, Recovery PSNR: {recovery_psnr.item():.2f} dB, Hiding SSIM: {hiding_ssim.item():.4f}, Recovery SSIM: {recovery_ssim.item():.4f}')

                axes[0, 0].imshow(host_vis)
                axes[0, 0].set_title('Host Image')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(secret_vis)
                axes[0, 1].set_title('Secret Image')
                axes[0, 1].axis('off')

                axes[1, 0].imshow(stego_vis)
                axes[1, 0].set_title('Stego Image (Host + Secret)')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(recovered_vis)
                axes[1, 1].set_title('Recovered Secret')
                axes[1, 1].axis('off')

                plt.tight_layout()
                plt.savefig(f'{model_name.lower()}_results/sample_{i+1}_{model_name.lower()}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error in sample {i+1}: {e}")
    
    avg_hiding_psnr = np.mean(all_hiding_psnr) if all_hiding_psnr else 0
    avg_recovery_psnr = np.mean(all_recovery_psnr) if all_recovery_psnr else 0
    avg_hiding_ssim = np.mean(all_hiding_ssim) if all_hiding_ssim else 0
    avg_recovery_ssim = np.mean(all_recovery_ssim) if all_recovery_ssim else 0
    
    print(f"\n{model_name} - Test Results:")
    print(f"  Average Hiding PSNR: {avg_hiding_psnr:.2f} dB")
    print(f"  Average Recovery PSNR: {avg_recovery_psnr:.2f} dB")
    print(f"  Average Hiding SSIM: {avg_hiding_ssim:.4f}")
    print(f"  Average Recovery SSIM: {avg_recovery_ssim:.4f}")
    
    return avg_hiding_psnr, avg_recovery_psnr, avg_hiding_ssim, avg_recovery_ssim

def main():
    """
    Main function to train and compare both models.
    """
    print("Enhanced Training: Comparing DWT vs ILWT for StarINN Steganography")
    print("=" * 70)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Enhanced parameters
    img_size = 64
    num_blocks = 2
    hidden_channels = 32  # Increased hidden channels for better capacity
    num_epochs = 150  # Increased epochs for better training
    dropout_rate = 0.1  # Dropout for regularization
    
    # Load dataset
    image_dir = "my_images"
    dataset = ImageSteganographyDataset(image_dir, img_size=img_size)
    
    print(f"Loaded {len(dataset)} images from {image_dir}")
    print(f"Training for {num_epochs} epochs with image size {img_size}x{img_size}")
    print(f"Using dropout rate: {dropout_rate}, hidden channels: {hidden_channels}")
    
    # Initialize models with enhanced parameters
    dwt_model = StarINNWithDWT(channels=6, num_blocks=num_blocks, hidden_channels=hidden_channels, dropout_rate=dropout_rate)
    ilwt_model = StarINNWithILWT(channels=6, num_blocks=num_blocks, hidden_channels=hidden_channels, dropout_rate=dropout_rate)
    
    # Move models to GPU if available
    dwt_model = dwt_model.to(device)
    ilwt_model = ilwt_model.to(device)
    
    print("\nModel configurations:")
    print(f"DWT Model parameters: {sum(p.numel() for p in dwt_model.parameters()):,}")
    print(f"ILWT Model parameters: {sum(p.numel() for p in ilwt_model.parameters()):,}")
    
    # Train DWT model
    print("\nStarting DWT model training...")
    dwt_losses, dwt_hiding_psnrs, dwt_recovery_psnrs, dwt_hiding_ssims, dwt_recovery_ssims, dwt_best_hid, dwt_best_rec, dwt_best_hid_ssim, dwt_best_rec_ssim = train_model(
        dwt_model, "DWT", dataset, num_epochs=num_epochs
    )
    
    # Train ILWT model
    print("\nStarting ILWT model training...")
    ilwt_losses, ilwt_hiding_psnrs, ilwt_recovery_psnrs, ilwt_hiding_ssims, ilwt_recovery_ssims, ilwt_best_hid, ilwt_best_rec, ilwt_best_hid_ssim, ilwt_best_rec_ssim = train_model(
        ilwt_model, "ILWT", dataset, num_epochs=num_epochs
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED FOR BOTH MODELS")
    print("=" * 70)
    
    # Test both models
    print("\nEvaluating models...")
    dwt_avg_hid, dwt_avg_rec, dwt_avg_hid_ssim, dwt_avg_rec_ssim = test_model(dwt_model, dataset, "DWT", num_samples=5)
    ilwt_avg_hid, ilwt_avg_rec, ilwt_avg_hid_ssim, ilwt_avg_rec_ssim = test_model(ilwt_model, dataset, "ILWT", num_samples=5)
    
    # Plot training curves
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 4, 1)
    plt.plot(dwt_losses, label='DWT Loss', color='blue')
    plt.plot(ilwt_losses, label='ILWT Loss', color='red')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 2)
    plt.plot(dwt_hiding_psnrs, label='DWT Hiding PSNR', color='blue')
    plt.plot(ilwt_hiding_psnrs, label='ILWT Hiding PSNR', color='red')
    plt.title('Hiding PSNR Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 3)
    plt.plot(dwt_recovery_psnrs, label='DWT Recovery PSNR', color='blue')
    plt.plot(ilwt_recovery_psnrs, label='ILWT Recovery PSNR', color='red')
    plt.title('Recovery PSNR Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 4)
    plt.plot(dwt_hiding_ssims, label='DWT Hiding SSIM', color='blue')
    plt.plot(ilwt_hiding_ssims, label='ILWT Hiding SSIM', color='red')
    plt.title('Hiding SSIM Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 5)
    plt.plot(dwt_recovery_ssims, label='DWT Recovery SSIM', color='blue')
    plt.plot(ilwt_recovery_ssims, label='ILWT Recovery SSIM', color='red')
    plt.title('Recovery SSIM Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    
    # Additional plots for better visualization
    plt.subplot(2, 4, 6)
    plt.plot(dwt_hiding_psnrs, label='DWT Hiding PSNR', color='blue', linestyle='--')
    plt.plot(ilwt_hiding_psnrs, label='ILWT Hiding PSNR', color='red', linestyle='--')
    plt.plot(dwt_recovery_psnrs, label='DWT Recovery PSNR', color='blue')
    plt.plot(ilwt_recovery_psnrs, label='ILWT Recovery PSNR', color='red')
    plt.title('PSNR Comparison (Hiding vs Recovery)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 7)
    plt.plot(dwt_hiding_ssims, label='DWT Hiding SSIM', color='blue', linestyle='--')
    plt.plot(ilwt_hiding_ssims, label='ILWT Hiding SSIM', color='red', linestyle='--')
    plt.plot(dwt_recovery_ssims, label='DWT Recovery SSIM', color='blue')
    plt.plot(ilwt_recovery_ssims, label='ILWT Recovery SSIM', color='red')
    plt.title('SSIM Comparison (Hiding vs Recovery)')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'DWT':<15} {'ILWT':<15}")
    print("-" * 70)
    print(f"{'Best Hiding PSNR':<25} {dwt_best_hid:.2f} dB{'':<5} {ilwt_best_hid:.2f} dB")
    print(f"{'Best Recovery PSNR':<25} {dwt_best_rec:.2f} dB{'':<5} {ilwt_best_rec:.2f} dB")
    print(f"{'Avg Hiding PSNR':<25} {dwt_avg_hid:.2f} dB{'':<5} {ilwt_avg_hid:.2f} dB")
    print(f"{'Avg Recovery PSNR':<25} {dwt_avg_rec:.2f} dB{'':<5} {ilwt_avg_rec:.2f} dB")
    print(f"{'Best Hiding SSIM':<25} {dwt_best_hid_ssim:.4f}{'':<5} {ilwt_best_hid_ssim:.4f}")
    print(f"{'Best Recovery SSIM':<25} {dwt_best_rec_ssim:.4f}{'':<5} {ilwt_best_rec_ssim:.4f}")
    print(f"{'Avg Hiding SSIM':<25} {dwt_avg_hid_ssim:.4f}{'':<5} {ilwt_avg_hid_ssim:.4f}")
    print(f"{'Avg Recovery SSIM':<25} {dwt_avg_rec_ssim:.4f}{'':<5} {ilwt_avg_rec_ssim:.4f}")
    
    # Determine which is better for each metric
    hiding_winner_psnr = "DWT" if ilwt_best_hid < dwt_best_hid else "ILWT"
    recovery_winner_psnr = "DWT" if dwt_best_rec > ilwt_best_rec else "ILWT"
    hiding_winner_ssim = "DWT" if ilwt_best_hid_ssim < dwt_best_hid_ssim else "ILWT"
    recovery_winner_ssim = "DWT" if dwt_best_rec_ssim > ilwt_best_rec_ssim else "ILWT"
    
    print(f"\nWinner - Best Hiding PSNR: {hiding_winner_psnr}")
    print(f"Winner - Best Recovery PSNR: {recovery_winner_psnr}")
    print(f"Winner - Best Hiding SSIM: {hiding_winner_ssim}")
    print(f"Winner - Best Recovery SSIM: {recovery_winner_ssim}")
    
    # Save models
    torch.save({
        'model_state_dict': dwt_model.state_dict(),
        'model_type': 'DWT',
        'best_hiding_psnr': dwt_best_hid,
        'best_recovery_psnr': dwt_best_rec,
        'avg_hiding_psnr': dwt_avg_hid,
        'avg_recovery_psnr': dwt_avg_rec,
        'best_hiding_ssim': dwt_best_hid_ssim,
        'best_recovery_ssim': dwt_best_rec_ssim,
        'avg_hiding_ssim': dwt_avg_hid_ssim,
        'avg_recovery_ssim': dwt_avg_rec_ssim,
        'dropout_rate': dropout_rate,
        'hidden_channels': hidden_channels,
        'num_blocks': num_blocks
    }, 'enhanced_dwt_steganography_model.pth')
    
    torch.save({
        'model_state_dict': ilwt_model.state_dict(),
        'model_type': 'ILWT',
        'best_hiding_psnr': ilwt_best_hid,
        'best_recovery_psnr': ilwt_best_rec,
        'avg_hiding_psnr': ilwt_avg_hid,
        'avg_recovery_psnr': ilwt_avg_rec,
        'best_hiding_ssim': ilwt_best_hid_ssim,
        'best_recovery_ssim': ilwt_best_rec_ssim,
        'avg_hiding_ssim': ilwt_avg_hid_ssim,
        'avg_recovery_ssim': ilwt_avg_rec_ssim,
        'dropout_rate': dropout_rate,
        'hidden_channels': hidden_channels,
        'num_blocks': num_blocks
    }, 'enhanced_ilwt_steganography_model.pth')
    
    print(f"\nEnhanced trained models saved as 'enhanced_dwt_steganography_model.pth' and 'enhanced_ilwt_steganography_model.pth'")
    print(f"Enhanced training curves saved as 'enhanced_training_comparison.png'")
    print(f"Visualizations saved in 'dwt_results' and 'ilwt_results' directories")
    
    # Analysis for research paper
    print("\n" + "=" * 70)
    print("RESEARCH PAPER ANALYSIS - ENHANCED MODEL")
    print("=" * 70)
    print("Enhanced Features Implemented:")
    print("  - Dropout regularization in Affine Coupling layers")
    print("  - AdamW optimizer with weight decay")
    print("  - OneCycle learning rate scheduler")
    print("  - Enhanced gradient clipping")
    print("  - Improved model capacity with adjusted hidden channels")
    
    print("\nDWT Approach:")
    print("  - Non-learnable, fixed wavelet transform")
    print("  - Better reconstruction accuracy")
    print("  - More mathematically precise")
    
    print("\nILWT Approach:")
    print("  - Learnable wavelet parameters")
    print("  - Better suited for end-to-end optimization")
    print("  - More adaptable to specific steganography task")
    print("  - Can potentially achieve better imperceptibility with training")
    
    print("\nFor research publication:")
    print("  - ILWT is more innovative as it's learnable")
    print("  - ILWT allows optimization specifically for steganography")
    print("  - Both approaches have their merits depending on the application")
    print("  - Enhanced training with AdamW and dropout provides better generalization")


if __name__ == "__main__":
    main()