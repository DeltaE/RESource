from setuptools import setup, find_packages
from pathlib import Path

HERE   = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")
REQS   = (HERE / "requirements.txt").read_text().splitlines()

setup(
    name='RES',
    version='1.0',
    author='Md Eliasinul Islam',
    author_email='eliasinul@gmail.com',
    url="https://github.com/DeltaE/RESource",
    description='This tool standardizes the resource assessment framework for Renewable Energy resources. Currently support Solar and On-shore Wind resource assessments. Developed by Delta E+ Research Lab, Simon Fraser University, BC, CA',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'RES=run:main',
        ],
    },
    install_requires= REQS,  # Use requirements from requirements.txt
    python_requires='>=3.11',
    long_description=README,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',

    ],
)
