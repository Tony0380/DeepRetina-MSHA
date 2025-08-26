"""
Multi-Scale Feature Pyramid per MSHA Network
Estrazione features a multiple scale (1x, 2x, 4x, 8x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeaturePyramid(nn.Module):
    """
    Feature Pyramid Network per estrazione multi-scala
    Estrae features a 4 scale: 1x, 2x, 4x, 8x
    """
    
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: Lista dei canali di input per ogni scala
            out_channels: Numero di canali output per ogni scala
        """
        super(MultiScaleFeaturePyramid, self).__init__()
        
        self.out_channels = out_channels
        self.num_scales = len(in_channels_list)
        
        # Lateral connections per ridurre i canali
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Output convolutions per smoothing
        self.output_convs = nn.ModuleList()
        for _ in range(self.num_scales):
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1)
                )
            )
        
        # Extra scale per maggiore dettaglio
        self.extra_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Inizializzazione dei pesi"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: Lista di tensori con features a diverse scale
                     Ordinati dal più grande al più piccolo
        Returns:
            pyramid_features: Lista di tensori con features normalizzate
        """
        # Applica lateral connections
        lateral_features = []
        for i, (feature, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            lateral = lateral_conv(feature)
            lateral_features.append(lateral)
        
        # Top-down pathway con feature fusion
        pyramid_features = []
        
        # Inizia dall'ultima feature (più piccola)
        prev_feature = lateral_features[-1]
        pyramid_features.append(self.output_convs[-1](prev_feature))
        
        # Processa dal penultimo al primo
        for i in range(len(lateral_features) - 2, -1, -1):
            # Upsample previous feature
            upsampled = F.interpolate(
                prev_feature, 
                size=lateral_features[i].shape[-2:],
                mode='bilinear', 
                align_corners=False
            )
            
            # Fused feature
            fused = lateral_features[i] + upsampled
            output = self.output_convs[i](fused)
            pyramid_features.insert(0, output)
            prev_feature = fused
        
        # Aggiungi una scala extra per dettagli fini
        extra_feature = self.extra_conv(pyramid_features[-1])
        pyramid_features.append(extra_feature)
        
        return pyramid_features


class FeaturePyramidAttention(nn.Module):
    """
    Attention Module specifico per Feature Pyramid
    Combina features da diverse scale con attention pesata
    """
    
    def __init__(self, channels, num_scales=5):
        super(FeaturePyramidAttention, self).__init__()
        
        self.channels = channels
        self.num_scales = num_scales
        
        # Scale attention weights
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_scales, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_scales, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement per ogni scala
        self.refinement_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_scales)
        ])
    
    def forward(self, pyramid_features, target_size):
        """
        Args:
            pyramid_features: Lista di features a diverse scale
            target_size: Dimensione target per output
        Returns:
            attended_feature: Feature combinata con attention
        """
        # Resize tutte le features alla dimensione target
        resized_features = []
        for i, feature in enumerate(pyramid_features):
            if feature.shape[-2:] != target_size:
                resized = F.interpolate(
                    feature, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                resized = feature
            
            # Applica refinement
            refined = self.refinement_convs[i](resized)
            resized_features.append(refined)
        
        # Concatena per calcolare attention weights
        concatenated = torch.cat(resized_features, dim=1)
        attention_weights = self.scale_attention(concatenated)  # [B, num_scales, 1, 1]
        
        # Applica weights e combina
        attended_feature = torch.zeros_like(resized_features[0])
        for i, (feature, weight) in enumerate(zip(resized_features, attention_weights.unbind(1))):
            attended_feature += feature * weight.unsqueeze(1)
        
        return attended_feature


class AdaptiveFeatureFusion(nn.Module):
    """
    Modulo per fusion adattiva delle features multi-scala
    """
    
    def __init__(self, channels, num_scales):
        super(AdaptiveFeatureFusion, self).__init__()
        
        self.channels = channels
        self.num_scales = num_scales
        
        # Spatial attention per ogni scala
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 8, 1),
                nn.Conv2d(channels // 8, 1, 7, padding=3),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])
        
        # Channel attention globale
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_scales, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, pyramid_features, target_size):
        """
        Fusion adattiva con spatial e channel attention
        """
        # Resize e applica spatial attention
        attended_features = []
        for i, feature in enumerate(pyramid_features):
            # Resize
            if feature.shape[-2:] != target_size:
                resized = F.interpolate(
                    feature, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                resized = feature
            
            # Spatial attention
            spatial_att = self.spatial_attentions[i](resized)
            attended = resized * spatial_att
            attended_features.append(attended)
        
        # Concatena e applica channel attention
        concatenated = torch.cat(attended_features, dim=1)
        channel_att = self.channel_attention(concatenated)
        
        # Final fusion
        fused = self.fusion_conv(concatenated)
        output = fused * channel_att
        
        return output
