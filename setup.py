from setuptools import setup, find_packages

# Read in requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='linkingtool',
    version='2.0',
    author='Md Eliasinul Islam',
    author_email='eliasinul@gmail.com',
    description='This tool translates and connects the results from BC-NEXUS to BC-PyPSA models. Developed by Delta E+ Research Lab, Simon Fraser University, BC, CA',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'linkingtool=linkingtool.cli:main',
        ],
    },
    install_requires= required,  # Use requirements from requirements.txt
    python_requires='>=3.6',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
