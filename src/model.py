import torch
import torch.nn as nn
import timm

class HybridRetinaNet(nn.Module):
    def __init__(self, num_classes=5):
        # 1. Corrected super call
        super().__init__() 
        
        # 2. CNN Branch: EfficientNetV2-M (Local Feature Extractor)
        # We use num_classes=0 to get the feature vector before the final FC layer
        self.cnn = timm.create_model(
            'tf_efficientnetv2_m', 
            pretrained=True, 
            num_classes=0, 
            global_pool='avg'
        )
        
        # 3. ViT Branch: vit_small_patch16_224 (Global Dependency Extractor)
        self.transformer = timm.create_model(
            'vit_small_patch16_224', 
            pretrained=True, 
            num_classes=0
        )
        
        # 4. Hybrid Fusion Head (Corrected 'Sequential' capitalization)
        # EfficientNetV2-M features (1280) + ViT-Small features (384) = 1664
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 384, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Local features from CNN
        cnn_features = self.cnn(x) 
        
        # Global features from Transformer
        vit_features = self.transformer(x)
        
        # Concatenate features (Dim 1 is the feature dimension)
        combined = torch.cat((cnn_features, vit_features), dim=1)
        
        # Final classification
        return self.fusion_head(combined)