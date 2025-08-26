"""
Hierarchical Attention Modules for MSHA Network
Implements Global, Regional and Local Attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GlobalAttention(nn.Module):
    """
    Global Attention Module - Attention over entire image
    """
    
    def __init__(self, in_channels, attention_dim=256, num_heads=8):
        super(GlobalAttention, self).__init__()
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        assert self.head_dim * num_heads == attention_dim, "attention_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(in_channels, attention_dim)
        self.key = nn.Linear(in_channels, attention_dim)
        self.value = nn.Linear(in_channels, attention_dim)
        
        # Output projection
        self.out_proj = nn.Linear(attention_dim, in_channels)
        
        # Normalization
        self.layer_norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Weight initialization"""
        for module in [self.query, self.key, self.value, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            attended_features: Tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Reshape for attention: [B, H*W, C]
        x_flat = x.contiguous().view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        
        # Residual connection
        residual = x_flat
        
        # Compute Q, K, V
        Q = self.query(x_flat)  # [B, H*W, attention_dim]
        K = self.key(x_flat)    # [B, H*W, attention_dim]
        V = self.value(x_flat)  # [B, H*W, attention_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        K = K.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        V = V.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        
        # Memory-efficient scaled dot-product attention
        # Use torch.nn.functional.scaled_dot_product_attention if available (PyTorch 2.0+)
        try:
            # Use native efficient attention if available
            attended = F.scaled_dot_product_attention(
                Q, K, V, 
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        except AttributeError:
            # Fallback to manual implementation with chunking for memory efficiency
            scale = math.sqrt(self.head_dim)
            attended = self._chunked_attention(Q, K, V, scale)
        
        # attended shape: [B, num_heads, H*W, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(B, H * W, self.attention_dim)
        
        # Output projection
        attended = self.out_proj(attended)  # [B, H*W, C]
        
        # Residual connection and layer norm
        output = self.layer_norm(attended + residual)
        
        # Reshape back to [B, C, H, W]
        output = output.transpose(1, 2).contiguous().view(B, C, H, W)
        
        return output
    
    def _chunked_attention(self, Q, K, V, scale, chunk_size=1024):
        """
        Memory-efficient chunked attention computation
        """
        B, num_heads, seq_len, head_dim = Q.shape
        attended = torch.zeros_like(Q)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = Q[:, :, i:end_i]  # [B, num_heads, chunk_size, head_dim]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(q_chunk, K.transpose(-2, -1)) / scale  # [B, num_heads, chunk_size, seq_len]
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            
            # Apply attention for this chunk
            attended[:, :, i:end_i] = torch.matmul(weights, V)  # [B, num_heads, chunk_size, head_dim]
        
        return attended


class RegionalAttention(nn.Module):
    """
    Regional Attention Module - Attention over 4 quadrants
    Dynamically adapts to input size
    """
    
    def __init__(self, in_channels, attention_dim=256, num_heads=8):
        super(RegionalAttention, self).__init__()
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Attention for each region
        self.region_attentions = nn.ModuleList([
            GlobalAttention(in_channels, attention_dim, num_heads) for _ in range(4)
        ])
        
        # Fusion layer to combine regions
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W] where H=W (square input)
        Returns:
            regional_features: Tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert H == W, f"Input must be square, got {H}x{W}"
        
        # Divide into 4 quadrants (dynamic based on input size)
        half_h, half_w = H // 2, W // 2
        top_left = x[:, :, :half_h, :half_w].contiguous()
        top_right = x[:, :, :half_h, half_w:].contiguous()
        bottom_left = x[:, :, half_h:, :half_w].contiguous()
        bottom_right = x[:, :, half_h:, half_w:].contiguous()
        
        regions = [top_left, top_right, bottom_left, bottom_right]
        
        # Apply attention to each region
        attended_regions = []
        for i, region in enumerate(regions):
            attended = self.region_attentions[i](region)
            attended_regions.append(attended)
        
        # Reconstruct the complete image
        top_half = torch.cat([attended_regions[0], attended_regions[1]], dim=3)
        bottom_half = torch.cat([attended_regions[2], attended_regions[3]], dim=3)
        reconstructed = torch.cat([top_half, bottom_half], dim=2)
        
        # Concatenate all regions for fusion
        all_regions = torch.cat(attended_regions, dim=1)  # [B, 4*C, half_h, half_w]
        
        # Resize to original dimensions and apply fusion
        all_regions_resized = F.interpolate(all_regions, size=(H, W), mode='bilinear', align_corners=False)
        fused_features = self.fusion(all_regions_resized)
        
        return fused_features + reconstructed  # Residual connection


class LocalAttention(nn.Module):
    """
    Local Attention Module - Attention over local patches
    Dynamically adapts patch size based on input dimensions
    """
    
    def __init__(self, in_channels, attention_dim=256, patch_size=64, num_heads=8):
        super(LocalAttention, self).__init__()
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        # Efficient local attention without memory explosion
        # No longer using GlobalAttention per patch
        
        # Aggregation layer
        self.aggregation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            local_features: Tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Determine number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # If dimensions are not divisible, use padding
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            pad_h = (num_patches_h + 1) * self.patch_size - H
            pad_w = (num_patches_w + 1) * self.patch_size - W
            x = F.pad(x, (0, pad_w, 0, pad_h))
            num_patches_h += 1 if pad_h > 0 else 0
            num_patches_w += 1 if pad_w > 0 else 0
        
        # Extract patches
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_size
                end_h = start_h + self.patch_size
                start_w = j * self.patch_size
                end_w = start_w + self.patch_size
                
                patch = x[:, :, start_h:end_h, start_w:end_w].contiguous()
                patches.append(patch)
        
        # Memory-efficient local attention using depthwise separable conv
        # Instead of applying attention to each patch separately
        attended_patches = []
        for patch in patches:
            # Simple local enhancement instead of full attention
            B_p, C_p, H_p, W_p = patch.shape
            
            # Local feature enhancement with efficient operations
            enhanced = F.adaptive_avg_pool2d(patch, 1)  # Global context per patch
            enhanced = F.interpolate(enhanced, size=(H_p, W_p), mode='bilinear', align_corners=False)
            
            # Channel attention
            channel_att = torch.sigmoid(enhanced)
            attended = patch * channel_att
            attended_patches.append(attended)
        
        # Reconstruct image from patches
        reconstructed = torch.zeros_like(x)
        patch_idx = 0
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_size
                end_h = start_h + self.patch_size
                start_w = j * self.patch_size
                end_w = start_w + self.patch_size
                
                reconstructed[:, :, start_h:end_h, start_w:end_w] = attended_patches[patch_idx]
                patch_idx += 1
        
        # Remove padding if applied
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            reconstructed = reconstructed[:, :, :H, :W]
        
        # Apply aggregation
        output = self.aggregation(reconstructed)
        
        return output + x[:, :, :H, :W]  # Residual connection


class HierarchicalAttentionFusion(nn.Module):
    """
    Module to combine Global, Regional and Local Attention
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(HierarchicalAttentionFusion, self).__init__()
        
        # Adaptive weights for fusion
        self.global_weight = nn.Parameter(torch.ones(1))
        self.regional_weight = nn.Parameter(torch.ones(1))
        self.local_weight = nn.Parameter(torch.ones(1))
        
        # Channel attention for intelligent fusion
        reduced_channels = max(in_channels // reduction_ratio, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 3, 1),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, global_features, regional_features, local_features):
        """
        Combines features from all attention levels
        """
        # Concatenate all features
        combined = torch.cat([global_features, regional_features, local_features], dim=1)
        
        # Calculate adaptive weights
        attention_weights = self.channel_attention(combined)  # [B, 3, 1, 1]
        
        # Apply weights
        weighted_global = global_features * attention_weights[:, 0:1, :, :]
        weighted_regional = regional_features * attention_weights[:, 1:2, :, :]
        weighted_local = local_features * attention_weights[:, 2:3, :, :]
        
        # Final fusion
        final_features = torch.cat([weighted_global, weighted_regional, weighted_local], dim=1)
        output = self.fusion_conv(final_features)
        
        return output
