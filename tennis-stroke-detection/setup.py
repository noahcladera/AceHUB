from setuptools import setup, find_packages

setup(
    name="tennis-stroke-detection",
    version="0.1.0",
    description="An AI-powered pipeline for detecting and analyzing tennis strokes from video.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Noah Cladera",
    author_email="noahcladera@gmail.com",
    url="https://github.com/noahcladera/tennis-stroke-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["lkxc"],
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.5.0",
        "mediapipe>=0.8.10",
        "opencv-python>=4.5.0",
        "ffmpeg-python>=0.2.0",
        "yt-dlp>=2021.12.1",
        "dtaidistance>=2.0.5",
        "pyyaml>=6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.7',
)