import setuptools

with open('README.md', 'r') as readme:
    long_description = readme.read()

reqs = []
with open('requirements.txt', 'r') as requirements:
    for row in requirements:
        reqs.append(row)

setuptools.setup(
    name='joey',
    version='pre1',
    author='DevitoCodes',
    description='A machine learning framework running on top of Devito',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/devitocodes/joey',
    packages=setuptools.find_packages(),
    install_requires=reqs,
    classifiers=[
        'Programming Language :: Python :: 3'
    ]
)
