from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mujoco==3.6",
    "warp-lang==1.12",
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
    "scipy==1.17.1",
    "onnxruntime==1.27.0"
]

# Installation operation
setup(
    name="mimic_mjlab",
    packages=find_packages(include=["src", "src.*", "rsl_rl", "rsl_rl.*"]),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
