import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
# from fastsam import FastSAM, FastSAMPrompt 
from .fastsam.model import FastSAM
from .fastsam.prompt import FastSAMPrompt
import torch
from PIL import Image as PilImage
import numpy as np
import ast
from .utils.tools import convert_box_xywh_to_xyxy
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from .fastsam.utils import bbox_iou
from ultralytics import YOLO
import tensorrt as trt

np.bool = np.bool_

class FastSAMNodeTrt(Node):
    def __init__(self):
        super().__init__('fastsam_node')
        self.declare_parameter('model_path', '/home/st/scooter_ws/segmentation/weights/FastSAM-x.engine')
        # self.declare_parameter('model_path', './weights/FastSAM-x.pt')
        self.declare_parameter('input_topic','oak/rgb/image_raw')
        self.declare_parameter('imgsz', 1024)
        self.declare_parameter('iou', 0.9)
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('retina', True)
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # self.declare_parameter('text_prompt', 'None')
        # self.declare_parameter('output', './output/')
        # self.declare_parameter('randomcolor', True)
        # self.declare_parameter('point_prompt', '[[0,0]]')
        # self.declare_parameter('point_label', '[0]')
        # self.declare_parameter('box_prompt', '[[0,0,0,0]]')
        self.declare_parameter('better_quality', False)
        self.declare_parameter('withContours', False)
        self.declare_parameter('output_topic','segmented_image')

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.input = self.get_parameter('input_topic').get_parameter_value().string_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.iou = self.get_parameter('iou').get_parameter_value().double_value
        self.conf = self.get_parameter('conf').get_parameter_value().double_value
        self.retina = self.get_parameter('retina').get_parameter_value().bool_value
        self.device = torch.device(self.get_parameter('device').get_parameter_value().string_value)
        # self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value
        # self.output = self.get_parameter('output').get_parameter_value().string_value
        # self.randomcolor = self.get_parameter('randomcolor').get_parameter_value().bool_value
        # self.point_prompt = ast.literal_eval(self.get_parameter('point_prompt').get_parameter_value().string_value)
        # self.point_label = ast.literal_eval(self.get_parameter('point_label').get_parameter_value().string_value)
        # self.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(self.get_parameter('box_prompt').get_parameter_value().string_value))
        self.better_quality = self.get_parameter('better_quality').get_parameter_value().bool_value
        self.withContours = self.get_parameter('withContours').get_parameter_value().bool_value
        self.output = self.get_parameter('output_topic').get_parameter_value().string_value

        self.model = FastSAM(self.model_path)
        # self.engine = self.load_engine(self.model_path)
        # self.engine = YOLO(self.model_path)

        self.subscription = self.create_subscription(
            Image,
            self.input,
            self.segment_callback,
            10)
        self.publisher = self.create_publisher(Image, self.output, 10)
        self.bridge = CvBridge()

    def load_engine(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    def segment_callback(self, msg):
        self.get_logger().info('Received image')
        image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        pil_image = PilImage.fromarray(image_np[..., ::-1])
        self.get_logger().info(f'PIL image size: {pil_image.size}')


        
        # input_tensor = self.preprocess_image(pil_image)

        preds = self.engine(pil_image)

        everything_results = self.model(
            pil_image,
            device=self.device,
            retina_masks=self.retina,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
        )
        everything_results = self.postprocess(preds, pil_image, image_np)


        prompt_process = FastSAMPrompt(pil_image, everything_results, device=self.device)
        bboxes = None
        points = None
        point_label = None

        # if self.box_prompt[0][2] != 0 and self.box_prompt[0][3] != 0:
        #     ann = prompt_process.box_prompt(bboxes=self.box_prompt)
        #     bboxes = self.box_prompt
        # elif self.text_prompt != 'None':
        #     ann = prompt_process.text_prompt(text=self.text_prompt)
        #     self.get_logger().info(f'Annotations: {ann}')
        # elif self.point_prompt[0] != [0, 0]:
        #     ann = prompt_process.point_prompt(
        #         points=self.point_prompt, pointlabel=self.point_label
        #     )
        #     points = self.point_prompt
        #     point_label = self.point_label
        # else:
        ann = prompt_process.everything_prompt()

        result = prompt_process.plot_to_result(
            annotations=ann,
            bboxes=bboxes,
            points=points,
            point_label=point_label,
            withContours=self.withContours,
            better_quality=self.better_quality,
        )

        segmented_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
        self.publisher.publish(segmented_image_msg)

    def preprocess_image(self, image):
        # Preprocess the image as required by the TensorRT model
        image = image.resize(self.imgsz)
        image_np = np.array(image).astype(np.float32)
        image_np = image_np.transpose(2, 0, 1)  # Change to channel-first format
        image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
        image_np /= 255.0  # Normalize to [0, 1]
        return image_np
    
    def postprocess(self, preds, pil_image, orig_img):
        """Postprocess the predictions from the TensorRT model."""
        p = ops.non_max_suppression(preds[0],
                                    self.conf,
                                    self.iou,
                                    agnostic=False,
                                    max_det=300,
                                    nc=len(self.engine.names),
                                    classes=None)
        
        results = []
        if len(p) == 0 or len(p[0]) == 0:
            self.get_logger().info("No object detected.")
            return results

        full_box = torch.zeros_like(p[0][0])
        full_box[2], full_box[3], full_box[4], full_box[6:] = orig_img.shape[3], orig_img.shape[2], 1.0, 1.0
        full_box = full_box.view(1, -1)
        critical_iou_index = bbox_iou(full_box[0][:4], p[0][:, :4], iou_thres=0.9, image_shape=orig_img.shape[2:])
        if critical_iou_index.numel() != 0:
            full_box[0][4] = p[0][critical_iou_index][:, 4]
            full_box[0][6:] = p[0][critical_iou_index][:, 6:]
            p[0][critical_iou_index] = full_box
        
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
        for i, pred in enumerate(p):
            if not len(pred):
                results.append(Results(orig_img=pil_image, path=None, names=self.engine.names, boxes=pred[:, :6]))
                continue
            if self.retina:
                if not isinstance(orig_img, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(orig_img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[2:], upsample=True)
                if not isinstance(orig_img, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(orig_img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img=pil_image, path=None, names=self.engine.names, boxes=pred[:, :6], masks=masks))
        return results



def main(args=None):
    rclpy.init(args=args)
    node = FastSAMNodeTrt()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
