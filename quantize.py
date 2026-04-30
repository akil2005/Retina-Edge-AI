import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "hybrid_retina_fp32.onnx"
model_int8 = "hybrid_retina_int8.onnx"

print("Shrinking model...")

quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QUInt8,
    per_channel=True,
    extra_options={'DisableShapeInference': True}# Good practice for CNNs to maintain accuracy
)

print(f" Quantization complete. INT8 model saved as: {model_int8}")