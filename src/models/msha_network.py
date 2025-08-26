"""
MSHA Network: Multi-Scale Hierarchical Attention Networks
Architettura principale per classificazione retinopatia diabetica
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, Tuple, List

from .attention_modules import (
    GlobalAttention, 
    RegionalAttention, 
    LocalAttention, 
    HierarchicalAttentionFusion
)
from .feature_pyramid import (
    MultiScaleFeaturePyramid, 
    FeaturePyramidAttention,
    AdaptiveFeatureFusion
)
from .uncertainty import UncertaintyQuantification


class MSHANetwork(nn.Module):
    """
    Multi-Scale Hierarchical Attention Network per Diabetic Retinopathy Classification
    
    Architettura:
    1. EfficientNet-B4 Backbone
    2. Multi-Scale Feature Pyramid (1x, 2x, 4x, 8x)
    3. Hierarchical Attention (Global, Regional, Local)
    4. Feature Fusion con attention pesata
    5. Uncertainty Quantification
    6. Classification Head (5 classi: 0-4)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = 'efficientnet_b4',
        attention_dim: int = 256,
        dropout_rate: float = 0.3,
        uncertainty_estimation: bool = True,
        pretrained: bool = True
    ):
        super(MSHANetwork, self).__init__()
        
        self.num_classes = num_classes
        self.attention_dim = attention_dim
        self.dropout_rate = dropout_rate
        self.uncertainty_estimation = uncertainty_estimation
        
        # 1. Backbone: EfficientNet-B4
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]  # Multi-scale features
        )
        
        # Get backbone feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 512)
            backbone_features = self.backbone(dummy_input)
            self.backbone_channels = [f.shape[1] for f in backbone_features]
        
        print(f"Backbone channels: {self.backbone_channels}")
        
        # 2. Multi-Scale Feature Pyramid
        self.feature_pyramid = MultiScaleFeaturePyramid(
            in_channels_list=self.backbone_channels,
            out_channels=attention_dim
        )
        
        # 3. Hierarchical Attention Modules
        self.global_attention = GlobalAttention(
            in_channels=attention_dim,
            attention_dim=attention_dim,
            num_heads=8
        )
        
        self.regional_attention = RegionalAttention(
            in_channels=attention_dim,
            attention_dim=attention_dim,
            num_heads=8
        )
        
        self.local_attention = LocalAttention(
            in_channels=attention_dim,
            attention_dim=attention_dim,
            patch_size=64,
            num_heads=8
        )
        
        # 4. Attention Fusion
        self.attention_fusion = HierarchicalAttentionFusion(
            in_channels=attention_dim,
            reduction_ratio=16
        )
        
        # 5. Feature Pyramid Attention
        self.pyramid_attention = FeaturePyramidAttention(
            channels=attention_dim,
            num_scales=5  # 4 backbone + 1 extra
        )
        
        # 6. Adaptive Feature Fusion
        self.adaptive_fusion = AdaptiveFeatureFusion(
            channels=attention_dim,
            num_scales=5
        )
        
        # 7. Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 8. Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(attention_dim // 2, attention_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 9. Output layer
        if uncertainty_estimation:
            self.uncertainty_head = UncertaintyQuantification(
                input_dim=attention_dim // 4,
                hidden_dim=512,
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
        else:
            self.output_layer = nn.Linear(attention_dim // 4, num_classes)
        
        # 10. Feature visualization hooks (opzionale)
        self.feature_maps = {}
        self.register_hooks()
        
        self._init_weights()
    
    def _init_weights(self):
        """Inizializzazione dei pesi per layers non pre-trained"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module not in self.backbone.modules():
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                if module not in self.backbone.modules():
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def register_hooks(self):
        """Registra hooks per visualizzazione features"""
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # Registra hooks per componenti principali
        self.global_attention.register_forward_hook(hook_fn('global_attention'))
        self.regional_attention.register_forward_hook(hook_fn('regional_attention'))
        self.local_attention.register_forward_hook(hook_fn('local_attention'))
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass del MSHA Network
        
        Args:
            x: Input tensor [B, 3, 512, 512]
            return_features: Se True, ritorna anche le feature maps
            
        Returns:
            output: Dictionary con predizioni e features
        """
        batch_size = x.size(0)
        
        # 1. Backbone feature extraction
        backbone_features = self.backbone(x)  # Lista di tensori a diverse scale
        
        # 2. Multi-Scale Feature Pyramid
        pyramid_features = self.feature_pyramid(backbone_features)
        
        # 3. Seleziona feature principale per attention (scala intermedia)
        main_features = pyramid_features[2]  # Scale intermedia
        
        # 4. Hierarchical Attention
        global_features = self.global_attention(main_features)
        regional_features = self.regional_attention(main_features)  
        local_features = self.local_attention(main_features)
        
        # 5. Attention Fusion
        fused_attention = self.attention_fusion(
            global_features, 
            regional_features, 
            local_features
        )
        
        # 6. Pyramid Attention su tutte le scale
        target_size = fused_attention.shape[-2:]
        pyramid_attended = self.pyramid_attention(pyramid_features, target_size)
        
        # 7. Adaptive Feature Fusion
        final_features = self.adaptive_fusion(pyramid_features, target_size)
        
        # 8. Combina attention e pyramid features
        combined_features = fused_attention + pyramid_attended + final_features
        
        # 9. Global Average Pooling
        pooled_features = self.global_pool(combined_features)  # [B, C, 1, 1]
        pooled_features = pooled_features.view(batch_size, -1)  # [B, C]
        
        # 10. Classification
        classifier_features = self.classifier(pooled_features)
        
        # 11. Output
        if self.uncertainty_estimation:
            mean_pred, variance_pred = self.uncertainty_head(classifier_features)
            output = {
                'logits': mean_pred,
                'variance': variance_pred,
                'predictions': torch.softmax(mean_pred, dim=-1)
            }
        else:
            logits = self.output_layer(classifier_features)
            output = {
                'logits': logits,
                'predictions': torch.softmax(logits, dim=-1)
            }
        
        # 12. Features aggiuntive se richieste
        if return_features:
            output.update({
                'backbone_features': backbone_features,
                'pyramid_features': pyramid_features,
                'global_attention': global_features,
                'regional_attention': regional_features,
                'local_attention': local_features,
                'fused_features': combined_features,
                'pooled_features': pooled_features,
                'classifier_features': classifier_features
            })
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estrae i pesi di attention per visualizzazione
        """
        with torch.no_grad():
            output = self.forward(x, return_features=True)
            
        attention_weights = {}
        for name, features in self.feature_maps.items():
            if 'attention' in name:
                # Calcola attention weights come media sui canali
                weights = features.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                weights = F.interpolate(weights, size=(512, 512), mode='bilinear', align_corners=False)
                attention_weights[name] = weights
        
        return attention_weights
    
    def monte_carlo_predict(
        self, 
        x: torch.Tensor, 
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Predizione con Monte Carlo Dropout per uncertainty estimation
        """
        if not self.uncertainty_estimation:
            raise ValueError("Il modello deve essere inizializzato con uncertainty_estimation=True")
        
        # Estrai features una volta sola
        with torch.no_grad():
            # Forward fino al classifier
            backbone_features = self.backbone(x)
            pyramid_features = self.feature_pyramid(backbone_features)
            main_features = pyramid_features[2]
            
            global_features = self.global_attention(main_features)
            regional_features = self.regional_attention(main_features)
            local_features = self.local_attention(main_features)
            
            fused_attention = self.attention_fusion(
                global_features, regional_features, local_features
            )
            
            target_size = fused_attention.shape[-2:]
            pyramid_attended = self.pyramid_attention(pyramid_features, target_size)
            final_features = self.adaptive_fusion(pyramid_features, target_size)
            combined_features = fused_attention + pyramid_attended + final_features
            
            pooled_features = self.global_pool(combined_features)
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            classifier_features = self.classifier(pooled_features)
        
        # Monte Carlo sampling su uncertainty head
        uncertainty_results = self.uncertainty_head.monte_carlo_inference(
            classifier_features, 
            num_samples=num_samples
        )
        
        return uncertainty_results
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Congela/scongela il backbone per fine-tuning
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def get_model_size(self) -> Dict[str, float]:
        """
        Calcola la dimensione del modello
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Memory estimation (in MB)
        param_size = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': param_size
        }


def create_msha_model(config: Dict) -> MSHANetwork:
    """
    Factory function to create MSHA model
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        model: Instance of MSHANetwork
    """
    model = MSHANetwork(
        num_classes=config.get('num_classes', 5),
        backbone=config.get('backbone', 'efficientnet_b4'),
        attention_dim=config.get('attention_dim', 256),
        dropout_rate=config.get('dropout_rate', 0.3),
        uncertainty_estimation=config.get('uncertainty_estimation', True),
        pretrained=True
    )
    
    return model


if __name__ == "__main__":
    # Test del modello
    config = {
        'num_classes': 5,
        'backbone': 'efficientnet_b4',
        'attention_dim': 256,
        'dropout_rate': 0.3,
        'uncertainty_estimation': True
    }
    
    model = create_msha_model(config)
    print(f"Modello creato: {model.get_model_size()}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output['logits'].shape}")
        print(f"Predictions shape: {output['predictions'].shape}")
        if 'variance' in output:
            print(f"Variance shape: {output['variance'].shape}")
