from setuptools import setup, find_packages

# Read in requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='RES',
    version='1.0',
    author='Md Eliasinul Islam',
    author_email='eliasinul@gmail.com',
    description='This tool standardizes the resource assessment framework for Renewable Energy resources. Currently support Solar and On-shore Wind resource assessments. Developed by Delta E+ Research Lab, Simon Fraser University, BC, CA',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'RES=RES.cli:main',
        ],
    },
    install_requires= required,  # Use requirements from requirements.txt
    python_requires='>=3.11',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
