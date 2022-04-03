from setuptools import setup

with open("README.md") as f:
    desc = f.read()

setup(
    name='PlaceNL',
    author='Lucas van Dijk',
    author_email='info@lucasvandijk.nl',
    description='PlaceNL: bot to automatically place pixels on /r/place 2022',
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/PlaceNL/rPlace2022",

    py_modules=['PlaceNL'],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],

    version="1.0",

    # Dependencies
    setup_requires=[
        'aiohttp'
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'aiohttp',
        'rich',
    ],
    python_requires=">=3.8",

    # CLI endpoints
    entry_points={
        'console_scripts': [
            'PlaceNL=PlaceNL:run',
        ]
    }
)
