""" 29-08 - claude 16:30h
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Classe para carregar os dados 
class QuestionDataset(Dataset):

  def __init__(self, df):
    self.questions = df['Question'].values
  
  def __len__(self):
    return len(self.questions)

  def __getitem__(self, idx):
    return self.questions[idx]


# Carrega o conjunto de dados
df = pd.read_csv("dados.csv")

# Divide em train/val
df_train = df.sample(frac=0.8, random_state=42)
df_val = df.drop(df_train.index)

# Transforma em dataset
train_set = QuestionDataset(df_train)
val_set = QuestionDataset(df_val)

# Carrega em batches
train_loader = DataLoader(train_set, batch_size=32, shuffle=True) 
val_loader = DataLoader(val_set, batch_size=32)

# Carrega o modelo
modelo = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-deduped")

# Otimizador e Loss
otimizador = torch.optim.Adam(modelo.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss() 

# Loop de treinamento
for epoca in range(10):

  for batch in train_loader:
  
    # Forward pass
    saidas = modelo(input_ids, attention_mask)
    loss = loss_fn(saidas, labels)

    # Backpropagation
    loss.backward()  
    otimizador.step()
    otimizador.zero_grad()

  # Código de validação omitido

# Salva o modelo
torch.save(modelo.state_dict(), 'medquad_pythia.pth') 

"""
""" treinamento funciona!
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd





# Classe para carregar os dados 
class QuestionDataset(Dataset):

  def __init__(self, df, tokenizer):
    self.questions = df['Question'].values
    self.tokenizer = tokenizer
  
  def __len__(self):
    return len(self.questions)

  def __getitem__(self, idx):
    question = self.questions[idx]
    encoding = self.tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    return {
        "input_ids": encoding["input_ids"].squeeze(),
        "attention_mask": encoding["attention_mask"].squeeze()
    }

# Carrega o conjunto de dados
df = pd.read_csv("dados.csv")

# Divide em train/val
df_train = df.sample(frac=0.8, random_state=42)
df_val = df.drop(df_train.index)

# Carrega o tokenizer e o modelo
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token

modelo = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-deduped")

# Transforma em dataset
train_set = QuestionDataset(df_train, tokenizer)
val_set = QuestionDataset(df_val, tokenizer)

# Carrega em batches
train_loader = DataLoader(train_set, batch_size=32, shuffle=True) 
val_loader = DataLoader(val_set, batch_size=32)

# Otimizador e Loss
otimizador = torch.optim.Adam(modelo.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss() 

# Loop de treinamento
for epoca in range(10):

  for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    # Forward pass
    outputs = modelo(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Backpropagation
    loss = loss_fn(logits.view(-1, logits.shape[-1]), input_ids.view(-1))  
    loss.backward()  
    otimizador.step()
    otimizador.zero_grad()

  # Código de validação omitido

# Salva o modelo
torch.save(modelo.state_dict(), 'medquad_pythia.pth')
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Classe para carregar os dados 
class QuestionDataset(Dataset):

    def __init__(self, df, tokenizer, max_length):
        self.questions = df['Question'].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        encoding = self.tokenizer(question, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

# Carrega o conjunto de dados
df = pd.read_csv("dados.csv")

# Divide em train/val
df_train = df.sample(frac=0.8, random_state=42)
df_val = df.drop(df_train.index)

# Carrega o tokenizer e o modelo
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
modelo = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-deduped")

# Transforma em dataset
train_set = QuestionDataset(df_train, tokenizer, max_length=45)  # Adjust max_length as needed
val_set = QuestionDataset(df_val, tokenizer, max_length=45)      # Adjust max_length as needed

# Carrega em batches
train_loader = DataLoader(train_set, batch_size=32, shuffle=True) 
val_loader = DataLoader(val_set, batch_size=32)

# Otimizador e Loss
otimizador = torch.optim.Adam(modelo.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss() 

# Loop de treinamento
for epoca in range(10):

    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Forward pass
        outputs = modelo(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Backpropagation
        loss = loss_fn(logits.view(-1, logits.shape[-1]), input_ids.view(-1))  
        loss.backward()  
        otimizador.step()
        otimizador.zero_grad()

# Código de validação omitido

# Salva o modelo
torch.save(modelo.state_dict(), 'medquad_pythia.pth')
