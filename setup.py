from setuptools import setup, find_packages

setup(
    name='linkingtool',
    version='2.0',
    author='Md Eliasinul Islam',
    author_email='eliasinul@gmail.com',
    description='This tool translates and connects the results from BC-NEXUS to BC-PyPSA models. Developed by Delta E+ Research Lab, Simon Fraser University, BC, CA',
    packages=find_packages(),  # Corrected typo
    entry_points={
        'console_scripts': [
            'linkingtool=linkingtool.cli:main',  # Ensure 'main' is the function in 'linkingtool.cli'
        ],
    },
    install_requires=[],  # Uncomment and specify requirements if needed
    python_requires='>=3.6',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',  # Fixed typo
        'Operating System :: OS Independent',  # Changed from Linux to OS Independent for broader compatibility
    ],
)
