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

### Enter the texts you want and send them to the model
```python

#you can summarize different texts
text1 = "Monkeys and gorillas are fascinating members of the primate family, each with distinct characteristics and behaviors that highlight their evolutionary significance. Monkeys, which include both Old World monkeys like baboons and macaques, and New World monkeys like capuchins and howler monkeys, are known for their diverse adaptations to various habitats. They exhibit a wide range of social structures, from complex troop hierarchies to cooperative foraging behaviors. Gorillas, on the other hand, are the largest of the great apes and are known for their impressive size and strength, as well as their gentle and social nature. Living primarily in the dense forests of Africa, gorillas are divided into two species: the Eastern gorillas and the Western gorillas. Both species face significant threats from habitat loss and poaching, making conservation efforts crucial for their survival. Studying these primates not only provides insights into their lives but also helps us understand the evolutionary connections shared with humans."
text2 = "Ancient cultures offer a rich tapestry of history, traditions, and achievements that continue to influence the world today. From the grand pyramids of Egypt, which stand as a testament to the architectural prowess and spiritual beliefs of the ancient Egyptians, to the sophisticated city-planning and governance of the Romans, whose legal and political systems laid the foundation for many modern societies, these civilizations left an indelible mark on human history. The Maya and Aztecs of Mesoamerica developed intricate calendars and monumental architecture, reflecting their deep understanding of astronomy and cosmology. In Asia, the ancient Chinese civilization contributed innovations like papermaking, gunpowder, and the compass, which have had lasting global impacts. Each of these cultures, with their unique languages, art, and philosophies, shaped the course of history and enriched the human experience in profound ways."

#here you create an array of the texts
texts = [text1, text2]

# Tokenizar el texto
inputs = tokenizer(texts, max_length=1024, return_tensors='pt', truncation=False, padding=True).to(device)

# If you have an available GPU

inputs = inputs.to(device)
model = model.to(device)
```
### Generate your summary
```python
summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=250, early_stopping=True)

#With summary_ids[#] you can choose what text are you summarizing
summary1 = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
summary2 = tokenizer.decode(summary_ids[1], skip_special_tokens=True)
```
### Result will be something like 
Summary 1: Monkeys and gorillas are fascinating members of the primate family. Each with distinct characteristics and behaviors that highlight their evolutionary significance. Studying these primates not only provides insights into their lives but also helps us understand the evolutionary connections shared with humans. Both species face significant threats from habitat loss and poaching.

Summary 2: Ancient cultures offer a rich tapestry of history, traditions, and achievements that continue to influence the world today. From the grand pyramids of Egypt to the sophisticated city-planning and governance of the Romans, these civilizations left an indelible mark on human history. Each of these cultures, with their unique languages, art, and philosophies, shaped the course of history.
