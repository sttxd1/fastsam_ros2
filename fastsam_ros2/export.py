import torch
from fastsam.model import FastSAM
from ultralytics import YOLO

# def export_to_onnx(model_path, onnx_path):
#     model = FastSAM(model_path)
#     model.model.eval() 
    
#     dummy_input = torch.randn(1, 3, 640, 640)
#     torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=11)

if __name__ == "__main__":
    model_path = '/home/st/scooter_ws/segmentation/weights/FastSAM-x.pt'
    onnx_path = '/home/st/scooter_ws/segmentation/weights/FastSAM-x.onnx'
    model  = YOLO(model_path)
    model.export(format="engine")
    # export_to_onnx(model_path, onnx_path)
