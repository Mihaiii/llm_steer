# LLM Steer
A Python module to steer LLM responses towards a certain topic/subject and to enhance capabilities (e.g., making it provide correct responses to tricky logical puzzles more often).
A practical tool for using activation engineering by adding steer vectors to different layers of a Large Language Model (LLM).
It should be used along with the transformers library.
## Demo
Google Colab demo: https://colab.research.google.com/github/Mihaiii/llm_steer/blob/main/demo/llm_steer_demo.ipynb

## Basic usage
Install it: `pip install llm_steer`
Then use:
```python
from llm_steer import Steer
steered_model = Steer(model, tokenizer)
```
Add a steering vector on a particular layer of the model with a given coefficient and text.
The coefficient can also be negative.
```python
steered_model.add(layer_idx=20, coeff=0.4, text="logical")
```
Get all the applied steering vectors:
```python
steered_model.get_all()
```
Remove all steering vectors to revert to initial model:
```python
steered_model.reset_all()
```

## Q / A
Q: What's the difference between llm_steer and mentioning what you want in the system prompt?

A: I see llm_steer as an enhancer. It can be used together with the system prompt.

<br/>
Q: How to determine the best parameters to be used?

A: I don't have a method; it's all trial and error. I recommend starting middle layers and with a small coefficient and then slowly increase it.

<br/>
Q: What models are supported?

A: I tested it on multiple architectures, including LLaMa, Mistral, Phi, StableLM.
Keep in mind that llm_steer is meant to be used together with HuggingFace's transformers library, so it won't work on GGUF, for example.

<br/>
Q: I applied steering vectors, but the LLM outputs gibberish. What should I do?

A: Try a lower coeff value or another layer.

<br/>
Q: Can I add multiple steering vectors on the same layer? Can I add the same steering vector on multiple layers? Can I add steering vectors with negative coefficients?

A: Yes, and please do. llm_steer is built for experimenting.
See the Colab for examples: https://colab.research.google.com/github/Mihaiii/llm_steer/blob/main/demo/llm_steer_demo.ipynb

<br/>
Q: Can I use steer vectors to enhance role-play characteristics (e.g., personas that are more funny or cocky)?

A: Yes.

<br/>
Q: Can I use negative steering vectors to force it not to say "As an AI language model"?

A: Yes.

## Credits / Thanks
- [DL Explorers](https://www.youtube.com/@DLExplorers-lg7dt) for his video on [activation engineer](https://www.youtube.com/watch?v=J2Gx6FFEaRY&t=29s) which goes over [an article](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector) and [a colab he made](https://colab.research.google.com/github/githubpradeep/notebooks/blob/main/activation_engineering.ipynb). The resources mentioned in that video were the starting point of llm_steer.
- Gary Bernhardt for his excellent [Python for programmers](https://www.executeprogram.com/courses/python-for-programmers) course. I needed a course that could help me go through the basics of Python without treating me like a dev noob (like most basic level tutorials treat their audience).
- Andrej Karpathy for his [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) video. I always wanted to make an open-source project, but there already was a repo for every idea I had. Not when it comes to tools for LLMs, though!
