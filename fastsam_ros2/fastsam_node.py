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

class FastSAMNode(Node):
    def __init__(self):
        super().__init__('fastsam_node')
        self.declare_parameter('model_path', './weights/FastSAM-x.pt')
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

        self.subscription = self.create_subscription(
            Image,
            self.input,
            self.segment_callback,
            10)
        self.publisher = self.create_publisher(Image, self.output, 10)
        self.bridge = CvBridge()

    def segment_callback(self, msg):
        self.get_logger().info('Received image')
        image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        pil_image = PilImage.fromarray(image_np[..., ::-1])

        everything_results = self.model(
            pil_image,
            device=self.device,
            retina_masks=self.retina,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
        )

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





def main(args=None):
    rclpy.init(args=args)
    node = FastSAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
