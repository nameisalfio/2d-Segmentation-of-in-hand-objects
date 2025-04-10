from setuptools import setup, find_packages

setup(
    name="hot3d_mask_rcnn",
    version="0.1.0",
    description="Mask R-CNN per la segmentazione di oggetti in mano nel dataset HOT3D",
    author="Alfio Spoto",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0",
        "Pillow>=8.0.0",
        "tensorboard>=2.6.0"
    ],
)