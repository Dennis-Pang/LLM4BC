from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()

setup(
    name="NLP4BC",  
    version="0.1.0",  
    description="A NLP base BC analysis tool",  
    author="Junyong Cao, Zixian Pang",  
    author_email="junyong.cao@uzh.ch, zixian.pang@uzh.ch", 
    url="https://github.com/JCTaylor666/MP_LLMBC", 
    packages=find_packages(), 
    install_requires=read_requirements(), 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  
)
