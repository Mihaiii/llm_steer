from setuptools import setup

setup(
    name='llm_steer',
    version='0.0.1',
    description='Steer LLM responses towards a certain topic/subject and enhance response capabilities using activation engineering by adding steer vectors',
    author='Mihai Chirculescu',
    author_email='apropodemine@gmail.com',
    py_modules=['llm_steer'],
    url="https://github.com/Mihaiii/llm_steer",
    install_requires=[
        'transformers'
    ],
)