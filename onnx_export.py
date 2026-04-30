import torch
import torch.nn as nn
import timm
import torch.onnx

# --- 1. THE SKELETON (Must match your training architecture) ---
class HybridModel(nn.Module):
    def __init__(self, num_classes=5):
        super(HybridModel, self).__init__()
        self.cnn = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=0, global_pool='avg')
        self.transformer = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + 384, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.transformer(x)
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        return self.fusion_head(combined)

# --- 2. PREPARATION ---
MODEL_PATH = r'L:\Projects\RetinaEdge-AI\weights\hybrid_retina_v2(S).pth'
OUTPUT_ONNX = "hybrid_retina_fp32.onnx"
DEVICE = torch.device("cpu")

# Load the model
model = HybridModel(num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 3. THE EXPORT (Tracing) ---
# Create a dummy image (1 image, 3 color channels, 224x224 pixels)
dummy_input = torch.randn(1, 3, 224, 224)

print("Starting ONNX Export...")
torch.onnx.export(
    model, 
    dummy_input, 
    OUTPUT_ONNX,
    export_params=True, 
    opset_version=14, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f" Success! File saved as: {OUTPUT_ONNX}")