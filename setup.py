from setuptools import setup

setup(
    name='llm-steer',
    version='0.0.4',
    description='Steer LLM responses towards a certain topic/subject and enhance response capabilities using activation engineering by adding steer vectors',
    author='Mihai Chirculescu',
    author_email='apropodemine@gmail.com',
    py_modules=['llm_steer'],
    url="https://github.com/Mihaiii/LLMSteer",
    install_requires=[
        'transformers'
    ],
)