from setuptools import setup, find_packages

setup(
    name="people-tracking-system",
    version="0.1.0",
    description="A comprehensive people tracking system",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pillow>=8.3.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "fastapi>=0.68.0",
        "uvicorn[standard]>=0.15.0",
        "pydantic>=1.8.0",
        "streamlit>=1.0.0",
        "pyyaml>=5.4.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "tracking-api=src.api.main:main",
            "tracking-ui=src.ui.main:main",
            "tracking-inference=src.inference.main:main",
        ],
    },
)

