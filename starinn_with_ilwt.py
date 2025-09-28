import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class StarINNWithLearnableILWT(nn.Module):
    """
    StarINN model with Learnable ILWT preprocessing.
    """
    def __init__(self, channels=6, num_blocks=2, hidden_channels=32, dropout_rate=0.1):
        super(StarINNWithLearnableILWT, self).__init__()
        
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