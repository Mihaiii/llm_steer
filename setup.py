from setuptools import setup

setup(
    name='SteerLLM',
    version='1.0.0',
    description='Steer LLM responses towards a certain topic/subject or to enhance rule-play characteristics (ex: personas which are more funny) or to enhance response capabilities (ex: make it provide correct responses to tricky puzzles), using steering vectors by doing activation engineering',
    author='Mihai Chirculescu',
    author_email='apropodemine@gmail.com',
    py_modules=['steer_llm'],
    url="https://github.com/Mihaiii/SteerLLM",
    install_requires=[
        transformers
    ],
)