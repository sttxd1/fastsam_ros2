import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
# from cv_bridge import CvBridge
# from fastsam import FastSAM, FastSAMPrompt 
from .fastsam.model import FastSAM
from .fastsam.prompt import FastSAMPrompt
import torch
from PIL import Image as PilImage
import numpy as np
import ast
from .utils.tools import convert_box_xywh_to_xyxy
from collections import deque

class FastSAMNode(Node):
    def __init__(self):
        super().__init__('fastsam_node')
        self.declare_parameter('model_path', '/home/st/scooter_ws/segmentation/FastSAM/weights/FastSAM-x.pt')
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
            20)
        self.publisher = self.create_publisher(Image, self.output, 20)
        # self.bridge = CvBridge()
        self.previous_anns = deque(maxlen=5)

    def segment_callback(self, msg):
        self.get_logger().info('Received image')
        image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        pil_image = PilImage.fromarray(image_np[..., ::-1])
        self.get_logger().info(f'PIL image size: {pil_image.size}')

        everything_results = self.model(
            pil_image,
            device=self.device,
            retina_masks=self.retina,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
        )

        prompt_process = FastSAMPrompt(pil_image, everything_results, device=self.device)
        # bboxes = None
        points = None
        point_label = None

        # if self.box_prompt[0][2] != 0 and self.box_prompt[0][3] != 0:

        ## box 
        ann, iou = prompt_process.box_prompt(bboxes=[[327, 350, 527, 480]])
        
        self.get_logger().info(f'current iou: {iou}')

        bboxes = [[350, 350, 450, 480]]


        if iou < 0.4:
            found_valid_ann = False
            for stored_ann, stored_iou in self.previous_anns:
                if stored_iou >= 0.4:
                    ann = stored_ann
                    found_valid_ann = True
                    self.get_logger().info('Using previous annotation due to low current IoU')
                    break
            if not found_valid_ann:
                self.get_logger().info('No previous annotation with sufficient IoU found, using current annotation')
        else:
            self.previous_anns.append((ann, iou))


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

        ## everything
        # ann = prompt_process.everything_prompt()

        # result = prompt_process.plot_to_result(
        #     annotations=ann,
        #     bboxes=bboxes,
        #     points=points,
        #     point_label=point_label,
        #     withContours=self.withContours,
        #     better_quality=self.better_quality,
        # )

        # Convert the annotation to a binary mask
        binary_mask = (ann[0] > 0).astype(np.uint8) * 255  # Assuming ann[0] is the mask
        binary_mask_pil = PilImage.fromarray(binary_mask)


        # Convert PIL image to ROS Image message manually (binary image)
        binary_mask_np = np.array(binary_mask_pil)
        img_msg = Image()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = msg.header.frame_id
        img_msg.height = binary_mask_np.shape[0]
        img_msg.width = binary_mask_np.shape[1]
        img_msg.encoding = "mono8"
        img_msg.is_bigendian = False
        img_msg.step = binary_mask_np.shape[1]
        img_msg.data = binary_mask_np.tobytes()


        # img_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")

        # Convert PIL image to ROS Image message manually (rgb image)
        # result_np = np.array(result)
        # img_msg = Image()
        # img_msg.header.stamp = self.get_clock().now().to_msg()
        # img_msg.header.frame_id = msg.header.frame_id
        # img_msg.height = result_np.shape[0]
        # img_msg.width = result_np.shape[1]
        # img_msg.encoding = "bgr8"
        # img_msg.is_bigendian = False
        # img_msg.step = result_np.shape[1] * 3
        # img_msg.data = result_np.tobytes()

        self.publisher.publish(img_msg)
        self.get_logger().info('Output published')

    # def segment_callback(self, msg):
    #     self.get_logger().info('Received image')
    #     image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    #     pil_image = PilImage.fromarray(image_np[..., ::-1])
    #     self.get_logger().info(f'PIL image size: {pil_image.size}')

    #     try:
    #         # Check if the model is None
    #         if self.model is None:
    #             self.get_logger().info('Model is None, reinitializing')
    #             self.model = FastSAM(self.model_path)

    #         self.get_logger().info('Running model inference...')
    #         everything_results = self.model(
    #             pil_image,
    #             device=self.device,
    #             retina_masks=self.retina,
    #             imgsz=self.imgsz,
    #             conf=self.conf,
    #             iou=self.iou
    #         )
    #         self.get_logger().info('Model inference completed')

    #         prompt_process = FastSAMPrompt(pil_image, everything_results, device=self.device)
    #         ann = prompt_process.everything_prompt()

    #         result = prompt_process.plot_to_result(
    #             annotations=ann,
    #             withContours=self.withContours,
    #             better_quality=self.better_quality,
    #         )

    #         segmented_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
    #         self.publisher.publish(segmented_image_msg)
    #         self.get_logger().info('Output published')
    #     except torch.cuda.OutOfMemoryError as e:
    #         self.get_logger().error(f'CUDA out of memory: {e}')
    #         torch.cuda.empty_cache()  # Clear GPU cache
    #         self.model = None  # Reset model to force reinitialization
    #     except AttributeError as e:
    #         self.get_logger().error(f'Attribute error: {e}')
    #         self.model = None  # Reset model to force reinitialization
    #     except Exception as e:
    #         self.get_logger().error(f'Error processing image: {e}')
    #         self.model = None  # Reset model to force reinitialization
    #     finally:
    #         torch.cuda.empty_cache()  # Clear GPU cache




def main(args=None):
    rclpy.init(args=args)
    node = FastSAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
