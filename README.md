# japaneseparticle-llm-testing
This is the code that accompanies the paper, "Mono- and Multilingual Language Models and Japanese Particle Prediction"

# Installation 
For use of this code, install the following packages:

- pandas
- torch torchvision
- transformers accelerate

If using Google Colab, you can install packages and import torch and pandas like such:
```
!pip3 install nltk pandas
!pip3 install torch torchvision
!pip3 install transformers accelerate

import pandas as pd 
import torch 
```
For the device, you can use either a GPU or a CPU:
```
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
# Description 
This code allows you to import a language model of your choosing, 
input synthetic sentence data sets, and specify particles of interest. 
The results generated are the log probabilites of the chosen particles 
for index positions marked by masked tokens. For particles of interest,
make sure that input sentences have [MASK] instead of the particles. 

The results generated will create a table in the form of a DataFrame. 
This can be exported and used to generate graphs or other tables,
using either seaborn or R. 

# Running 
Make sure packages are installed prior to running the code. 

Make sure your LLM does not tokenize particles as two separate tokens.

```
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

# test model with particles of interest
particles = ['で']
particle_ids = tokenizer.convert_tokens_to_ids(particles)
particle = tokenizer.convert_ids_to_tokens(particle_ids)

# print particle as a single token
particle
```

To import the model you would like to use, as well as tokenizers:
```
# alter code for appropriate models you wish to use
# LLM in this model comes from the model used in the original paper

from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
model_id = "FacebookAI/xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
model.eval()
```

To identify particles of interest and input data set:
```
# input particles to be focused on to calculate the log probs
# input sentence data set

particles = ['で', 'に']
particle_ids = tokenizer.convert_tokens_to_ids(particles)
sentences = ["図書館[MASK]勉強します。", "公園[MASK]休みます。",
             "学校[MASK]習います。", "バス[MASK]待ちます。",
             "図書館[MASK]行きます。", "公園[MASK]散歩します。",
             "学校[MASK]走ります。", "バス[MASK]出ます。"]
```

To generate results:

```
# Calculate log probs
# Create a DataFrame that organizes by sentence, particle,
# masked index position, and the log probs.

results = []
for sentence in sentences:
  sentence = sentence.replace('[MASK]', tokenizer.mask_token)
  inputs = tokenizer(sentence, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)
  masked_indices = (inputs['input_ids'][0] == tokenizer.mask_token_id).nonzero()
  logprobs = torch.log_softmax(outputs.logits, dim=-1)
  for masked_index in masked_indices:
      for particle in particle_ids:
        results.append({
        'sentence': sentence,
        'particle': particle,
        'masked_index': masked_index[0].item(),
        'logprobs': logprobs[0, masked_index, particle][0].item()
        })

results = pd.DataFrame(results)
results["particle"] = [tokenizer.convert_ids_to_tokens(particle)
                      for particle in results["particle"]]
results
```

# Citation
Hiwatari-Brown. (2025). Mono- and Multilingual Language Models and Japanese Particle Prediction.
