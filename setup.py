from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nluz-cleaning-toolbox",
    version="0.3.0",
    author="Tengku Irfan",
    author_email="tengku.irfan0278@student.unri.ac.id",
    description="A modular, Lego-like toolbox for data cleaning and image preprocessing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tengkuirfan/nluz-cleaning-toolbox",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="data-cleaning, image-preprocessing, data-science, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/tengkuirfan/nluz-cleaning-toolbox/issues",
        "Source": "https://github.com/tengkuirfan/nluz-cleaning-toolbox",
    },
)
