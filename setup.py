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
    version='0.0.8',
    author='Khaled Ibrahim',
    author_email='khaled.ibrahim@wfp.org',
    description='Enterprise Deduplication Data Science Package',
    url='https://github.com/Khaledhamza77/mkr',
    packages=setuptools.find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.8'
)