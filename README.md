# Summary-with-pre-trained-model
In this project yoy will find a program that make a summaries with a pre trained model from hugging face. 

---
## Requirements 

I recomend to install the following libraries in the following order:
- python = 3.11
- pip install git+https://github.com/huggingface/transformers
- conda install pytorch:pytorch
If you are using a GPU use this command to install pytorch in your eviroment
- conda install pythorch==2.2.2 torchvision==0.17.2 torchaudio=2.2.2 putorch-cuda=11.8 -c pytorch -c nvidia

---
## Code
### Importing libraries 
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# Assign available GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
```
### Importing the pre-trained model 
```python
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
```

### Input the text that you want and send it to the model
```python
text = "Monkeys and gorillas are fascinating members of the primate family, each with distinct characteristics and behaviors that highlight their evolutionary significance. Monkeys, which include both Old World monkeys like baboons and macaques, and New World monkeys like capuchins and howler monkeys, are known for their diverse adaptations to various habitats. They exhibit a wide range of social structures, from complex troop hierarchies to cooperative foraging behaviors. Gorillas, on the other hand, are the largest of the great apes and are known for their impressive size and strength, as well as their gentle and social nature. Living primarily in the dense forests of Africa, gorillas are divided into two species: the Eastern gorillas and the Western gorillas. Both species face significant threats from habitat loss and poaching, making conservation efforts crucial for their survival. Studying these primates not only provides insights into their lives but also helps us understand the evolutionary connections shared with humans."
inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True).to(device)
#If you are using a GPU 
inputs = inputs.to(device)
model = model.to(device)
```
### Generate your summary
```python
summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=250, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```
### Result will be something like 
Monkeys and gorillas are fascinating members of the primate family. Each with distinct characteristics and behaviors that highlight their evolutionary significance. Studying these primates not only provides insights into their lives but also helps us understand the evolutionary connections shared with humans. Both species face significant threats from habitat loss and poaching.
