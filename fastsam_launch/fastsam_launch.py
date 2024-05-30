from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import torch

def generate_launch_description():
    # Declare launch arguments
    model_path = LaunchConfiguration("model_path")
    model_path_cmd = DeclareLaunchArgument(
        "model_path",
        default_value="./weights/FastSAM-x.pt",
        description="Model path"
    )


    input_topic = LaunchConfiguration("input_topic")
    input_topic_cmd = DeclareLaunchArgument(
        "input_topic",
        default_value="oak/rgb/image_raw",
        description="Input image topic"
    )

    imgsz = LaunchConfiguration("imgsz")
    imgsz_cmd = DeclareLaunchArgument(
        "imgsz",
        default_value="1024",
        description="Image size"
    )

    iou = LaunchConfiguration("iou")
    iou_cmd = DeclareLaunchArgument(
        "iou",
        default_value="0.9",
        description="IoU threshold for filtering the annotations"
    )

    conf = LaunchConfiguration("conf")
    conf_cmd = DeclareLaunchArgument(
        "conf",
        default_value="0.4",
        description="Object confidence threshold"
    )

    retina = LaunchConfiguration("retina")
    retina_cmd = DeclareLaunchArgument(
        "retina",
        default_value="True",
        description="Draw high-resolution segmentation masks"
    )

    device = LaunchConfiguration("device")
    device_cmd = DeclareLaunchArgument(
        "device",
        default_value="cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use (GPU/CPU)"
    )

    # text_prompt = LaunchConfiguration("text_prompt")
    # text_prompt_cmd = DeclareLaunchArgument(
    #     "text_prompt",
    #     default_value="None",
    #     description="Text prompt for FastSAM"
    # )

    # output = LaunchConfiguration("output")
    # output_cmd = DeclareLaunchArgument(
    #     "output",
    #     default_value="./output/",
    #     description="Image save path"
    # )

    # randomcolor = LaunchConfiguration("randomcolor")
    # randomcolor_cmd = DeclareLaunchArgument(
    #     "randomcolor",
    #     default_value="True",
    #     description="Mask random color"
    # )

    # point_prompt = LaunchConfiguration("point_prompt")
    # point_prompt_cmd = DeclareLaunchArgument(
    #     "point_prompt",
    #     default_value="[[0,0]]",
    #     description="Point prompt for FastSAM"
    # )

    # point_label = LaunchConfiguration("point_label")
    # point_label_cmd = DeclareLaunchArgument(
    #     "point_label",
    #     default_value="[0]",
    #     description="Point label for FastSAM"
    # )

    # box_prompt = LaunchConfiguration("box_prompt")
    # box_prompt_cmd = DeclareLaunchArgument(
    #     "box_prompt",
    #     default_value="[[0,0,0,0]]",
    #     description="Box prompt for FastSAM"
    # )

    better_quality = LaunchConfiguration("better_quality")
    better_quality_cmd = DeclareLaunchArgument(
        "better_quality",
        default_value="False",
        description="Better quality using morphologyEx"
    )

    withContours = LaunchConfiguration("withContours")
    withContours_cmd = DeclareLaunchArgument(
        "withContours",
        default_value="False",
        description="Draw the edges of the masks"
    )

 

    output_topic = LaunchConfiguration("output_topic")
    output_topic_cmd = DeclareLaunchArgument(
        "output_topic",
        default_value="'segmented_image'",
        description="Output image topic"
    )

    # Node configuration
    fastsam_node_cmd = Node(
        package="fastsam_ros2",
        executable="fastsam_node",
        name="fastsam_node",
        parameters=[{
            "model_path": model_path,
            "input_topic": input_topic,
            "imgsz": imgsz,
            "iou": iou,
            "conf": conf,
            "retina": retina,
            "device": device,
            "better_quality": better_quality,
            "withContours": withContours,
            "output_topic": output_topic,
        }]
        # remappings=[("input_image", input_image_topic), ("segmented_image", "/output/segmented_image")]
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(model_path_cmd)
    ld.add_action(input_topic_cmd)
    ld.add_action(imgsz_cmd)
    ld.add_action(iou_cmd)
    ld.add_action(conf_cmd)
    ld.add_action(retina_cmd)
    ld.add_action(device_cmd)
    # ld.add_action(text_prompt_cmd)
    # ld.add_action(output_cmd)
    # ld.add_action(randomcolor_cmd)
    # ld.add_action(point_prompt_cmd)
    # ld.add_action(point_label_cmd)
    # ld.add_action(box_prompt_cmd)
    ld.add_action(better_quality_cmd)
    ld.add_action(withContours_cmd)
    ld.add_action(output_topic_cmd)

    # Add node action
    ld.add_action(fastsam_node_cmd)

    return ld


# "text_prompt": text_prompt,
#             "output": output,
#             "randomcolor": randomcolor,
#             "point_prompt": point_prompt,
#             "point_label": point_label,
#             "box_prompt": box_prompt,

