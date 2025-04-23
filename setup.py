from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="cybi-smartshoe-ai",
    version="1.0.0",
    author="CYBI AI Team",
    author_email="info@cybi-smartshoe.com",
    description="AI system for weight prediction and health monitoring from CYBI smartshoe data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cybi-ai/cybi-smartshoe-ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cybi-train=cybi.main:main",
            "cybi-api=cybi.deploy_api:main",
        ],
    },
) 