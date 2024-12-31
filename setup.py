from setuptools import setup, find_packages
from gitin.__version__ import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gitin",
    version=__version__,
    author="unclecode",
    author_email="",  # Add your email if you want
    description="Extract and format GitHub repository content for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/unclecode/gitin",
    project_urls={
        "Bug Tracker": "https://github.com/unclecode/gitin/issues",
        "Documentation": "https://github.com/unclecode/gitin#readme",
        "Source Code": "https://github.com/unclecode/gitin",
        "Changelog": "https://github.com/unclecode/gitin/blob/main/CHANGELOG.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Version Control :: Git",
        "Topic :: Text Processing :: Markup :: Markdown",
    ],
    keywords="github, llm, content-extraction, markdown, repository-analysis",
    python_requires=">=3.6",
    install_requires=[
        "click>=8.1.7",
        "requests>=2.31.0",
        "tqdm>=4.66.1",
    ],
    entry_points={
        "console_scripts": [
            "gitin=gitin.gitin:main",
        ],
    },
)
