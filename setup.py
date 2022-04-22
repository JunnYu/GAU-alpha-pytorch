from setuptools import find_packages, setup

setup(
    name="gau_alpha",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.0.2",
    license="Apache 2.0",
    description="gau_alpha_pytorch",
    author="Jun Yu",
    author_email="573009727@qq.com",
    url="https://github.com/JunnYu/GAU-alpha-pytorch",
    keywords=["gau_alpha", "pytorch"],
    install_requires=["transformers>=4.13.0"],
)
