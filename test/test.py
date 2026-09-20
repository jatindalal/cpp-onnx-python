from cpp_onnx_python_ext import InferenceSession, InferenceSessionProvider
import numpy as np

session = InferenceSession(
    "/Users/jd/Downloads/4x-UltraSharpV2_fp32_op17.onnx",
    InferenceSessionProvider.CPU,
    0
)

for input in session.inputs:
    print(input.to_dict())
for output in session.outputs:
    print(output.to_dict())

input_data = np.random.random((1, 3, 60, 40)).astype(np.float32)
print(session.run([input_data]))
