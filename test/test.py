print("cuDNN explicitly loaded")
import os
from pathlib import Path
import ctypes




package_dir = Path(r"C:\Users\jdjat\venv\Lib\site-packages\cpp_onnx_python")
if os.name == "nt":
    os.add_dll_directory(str(package_dir))
ctypes.WinDLL(str(package_dir / "cudnn64_9.dll"))
from cpp_onnx_python import InferenceSession, InferenceSessionProvider
import numpy as np

session = InferenceSession(
    "C:/Users/jdjat/Downloads/efficientnet_es_Opset18.onnx",
    InferenceSessionProvider.GPU,
    0
)

for input in session.inputs:
    print(input.to_dict())
for output in session.outputs:
    print(output.to_dict())

input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
output_data = session.run([input_data])
print(output_data[0].shape)
