# import packages

import torch
from transformers import AutoTokenizer
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# check available computation device
# if you have a local GPU or if you are using a GPU on Colab, the following code should return "CUDA"
# if you are on Mac and have an > M1 chip, the following code should return "MPS"
# otherwise, it should return "CPU"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: {device}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Device: {device}")
else:
    device = torch.device("cpu")
    print(f"Device: {device}")

# test PyTorch

# randomly initialize a tensor of shape (5, 3)
x = torch.rand(5, 3).to(device)
print(x)
print("Shape of tensor x:", x.shape)
print("Device of tensor x:", x.device)

# initialize a tensor of shape (5, 3) with ones
y = torch.ones(5, 3).to(device)
print(y)

# multiply x and y in a pointwise manner
z = x * y
print(z)

# note that to do a matrix multiplication between x and y, we need to transpose
# either matrix x or y because the inner dimensions must match
z1 = x @ y.T
print(z1)

# or:
z2 = x.T @ y
print(z2)
# and note that the result of z1 and z2 are not the same and have different shapes

# testing LangChain

# run a Wikipedia query, searching for the article "Attention is all you need"
# NB: requires an internet connection
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia.run("Attention is all you need")

# testing the package transformers which provides pre-trained language models
# and excellent infrastructure around them

# download (if not available yet) and load GPT-2 tokenizer
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
text = "Attention is all you need"
# tokenize the text (i.e., convert the string into a tensor of token IDs)
input_ids = tokenizer_gpt2(
    text,
    return_tensors="pt",
).to(device)

print("Input IDs:", input_ids)
