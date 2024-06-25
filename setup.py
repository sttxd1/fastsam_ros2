from setuptools import find_packages, setup

package_name = 'fastsam_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['fastsam_launch/fastsam_launch.py']),
        ('share/' + package_name + '/launch', ['fastsam_launch/fastsam_trt_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='st',
    maintainer_email='119010261@link.cuhk.edu.cn',
    description='ROS 2 package for FastSAM segmentation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fastsam_node = fastsam_ros2.fastsam_node:main',
            'fastsam_node_trt = fastsam_ros2.fastsam_node_trt:main',
        ],
    },
)
