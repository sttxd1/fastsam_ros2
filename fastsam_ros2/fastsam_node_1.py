import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from .fastsam.model import FastSAM
from .fastsam.prompt import FastSAMPrompt
import torch
from PIL import Image as PilImage
import numpy as np
import ast
from .utils.tools import convert_box_xywh_to_xyxy
import time

class FastSAMNode(Node):
    def __init__(self):
        super().__init__('fastsam_node')
        self.declare_parameter('model_path', '/home/st/scooter_ws/segmentation/FastSAM/weights/FastSAM-x.engine')
        self.declare_parameter('input_topic','oak/rgb/image_raw')
        self.declare_parameter('imgsz', 1024)
        self.declare_parameter('iou', 0.9)
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('retina', True)
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
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
        self.better_quality = self.get_parameter('better_quality').get_parameter_value().bool_value
        self.withContours = self.get_parameter('withContours').get_parameter_value().bool_value
        self.output = self.get_parameter('output_topic').get_parameter_value().string_value

        self.model = self.load_model()

        self.subscription = self.create_subscription(
            Image,
            self.input,
            self.segment_callback,
            10)
        self.publisher = self.create_publisher(Image, self.output, 10)
        self.bridge = CvBridge()

        self.last_time = time.time()
        self.rate_limit = 0.1  # Limit to 10 FPS

    def load_model(self):
        self.get_logger().info(f'Loading model from: {self.model_path}')
        try:
            model = FastSAM(self.model_path)
            if model is not None and hasattr(model, 'create_execution_context'):
                self.get_logger().info('Model loaded and context created successfully')
                return model
            else:
                self.get_logger().error('Failed to load model or create context')
                return None
        except Exception as e:
            self.get_logger().error(f'Error loading model: {e}')
            return None

    def segment_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_time < self.rate_limit:
            self.get_logger().info('Skipping frame to enforce rate limit')
            return
        self.last_time = current_time

        self.get_logger().info('Received image')
        image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        pil_image = PilImage.fromarray(image_np[..., ::-1])
        self.get_logger().info(f'PIL image size: {pil_image.size}')

        try:
            if self.model is None:
                self.get_logger().info('Model is None, reinitializing')
                self.model = self.load_model()

            self.get_logger().info('Running model inference...')
            everything_results = self.model(
                pil_image,
                device=self.device,
                retina_masks=self.retina,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou
            )
            self.get_logger().info('Model inference completed')

            prompt_process = FastSAMPrompt(pil_image, everything_results, device=self.device)
            ann = prompt_process.everything_prompt()

            result = prompt_process.plot_to_result(
                annotations=ann,
                withContours=self.withContours,
                better_quality=self.better_quality,
            )

            segmented_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            self.publisher.publish(segmented_image_msg)
            self.get_logger().info('Output published')
        except torch.cuda.OutOfMemoryError as e:
            self.get_logger().error(f'CUDA out of memory: {e}')
            torch.cuda.empty_cache()  # Clear GPU cache
            self.model = None  # Reset model to force reinitialization
        except AttributeError as e:
            self.get_logger().error(f'Attribute error: {e}')
            self.model = None  # Reset model to force reinitialization
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            self.model = None  # Reset model to force reinitialization
        finally:
            torch.cuda.empty_cache()  # Clear GPU cache

def main(args=None):
    rclpy.init(args=args)
    node = FastSAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
