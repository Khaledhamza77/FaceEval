import os
import setuptools

def read_requirements():
    try:
        with open(os.path.dirname(__file__) + '/requirements.txt', 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("requirements.txt not found. Proceeding without it.")
        return []

setuptools.setup(
    name='mkr',
    version='0.0.1',
    author='Khaled Ibrahim',
    author_email='khaledhamza@aucegypt.edu',
    description='A Python package for face evaluation and quality assessment',
    url='https://github.com/Khaledhamza77/FaceEval',
    packages=setuptools.find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.8'
)