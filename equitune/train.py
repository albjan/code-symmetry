import os
import pickle
import itertools
import time

import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, RandomSampler, random_split
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM

lr = 1e-5
num_epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STORAGE_DIR = '/proj/rcs-hdd/aj3051/symmetry'

run = wandb.init(
    reinit=True,
    project="equitune",
    name="Train 2",
    config={
        "learning_rate": lr,
        "epochs": num_epochs,
    },
)

with open(os.path.join(STORAGE_DIR, 'data_tokenized_4.pkl'), 'rb') as f:
    dataset = pickle.load(f)

def permute_lines(input, line_permutation_order, tokenized_line_inds):
    input_len = len(input)
    permute_indices = torch.zeros(input_len).to(input.device).detach()
    
    curr_ind = 0
    for new_line_num in line_permutation_order: 
        line_beg, line_end = tokenized_line_inds[new_line_num], tokenized_line_inds[new_line_num+1]
        line_len = line_end - line_beg
        permute_indices[curr_ind:curr_ind+line_len] = torch.arange(line_beg, line_end)
        curr_ind += line_len
        
    permuted_input = torch.index_select(input, 0, permute_indices.to(torch.long))
    return permuted_input

class CodePermutationDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

permutation_dataset = CodePermutationDataset(dataset=dataset)

sampler = RandomSampler(dataset)
shuffled_indices = list(sampler)
shuffled_dataset = torch.utils.data.Subset(dataset, shuffled_indices)
train_length = int(0.8 * len(shuffled_dataset))
validation_length = int(0.1 * len(shuffled_dataset))
test_length = len(shuffled_dataset) - train_length - validation_length

train_set, validation_set, test_set = random_split(
    shuffled_dataset, [train_length, validation_length, test_length]
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=None, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=None, shuffle=False)

class CodePermutationEquitune(nn.Module): 

    def __init__(self, base_model="deepseek-ai/deepseek-coder-1.3b-base", num_permutations=4):
        super(CodePermutationEquitune, self).__init__() 
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir="/proj/rcs-hdd/aj3051/hf_transformers") #torch_dtype=torch.float16)
        self.num_permutations = num_permutations
        module_dict = dict(self.base_model.named_modules())

        second_to_last_attn_layer = module_dict['model.layers.22']
        second_to_last_attn_layer.register_forward_hook(self.permute_output_hook)

        last_attn_layer = module_dict['model.layers.23']
        last_attn_layer.register_forward_hook(self.average_output_hook)

    def permute_output_hook(self, module, input, output):
        hidden_state = output[0]
        # print(f'thing being permuted shape: {hidden_state.shape}')
        num_permutations, num_tokens, embedding_dim = hidden_state.shape
        hidden_state = hidden_state.reshape(hidden_state.shape[0], -1)
        # print(f'reshaped hidden state: {hidden_state.shape}')
        embedding_orig = hidden_state[0, :]
        for i in range(1, num_permutations):
    # permute the input, fill in next row of data
            hidden_state[i, :] = permute_lines(
                input=embedding_orig,
                line_permutation_order=self.metadata['line_permutation_orders'][i],
                tokenized_line_inds=self.metadata['tokenized_line_inds'] * embedding_dim,
            )
        hidden_state = hidden_state.reshape(num_permutations, num_tokens, embedding_dim)
        # print(f'reshaped hidden state again: {hidden_state.shape}')
        return (hidden_state,) + output[1:]

    def average_output_hook(self, module, input, output):
        hidden_state = output[0]
        hidden_state = hidden_state.mean(axis=0).unsqueeze(0)
        return (hidden_state,) + output[1:]

    def forward(self, input, metadata):
        x = input
        self.metadata = metadata
        x = self.base_model(x)
        logits = x.logits[:, -1, :]
        return logits
    
equitune_model = CodePermutationEquitune().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = bnb.optim.Adam8bit(equitune_model.parameters(), lr=lr)

eval_every = 2000
checkpoint_every = 10000

MODEL_DIR = f'{STORAGE_DIR}/results/'

equitune_model.train()
step = 0
mx_len = 0

@torch.no_grad()
def get_val():
    val_loss = 0
    for batch in validation_loader:
        input_ids, label, metadata = batch
        input_ids = input_ids.to(device)
        label = label.unsqueeze(dim=0).to(device)
        with torch.autocast(device_type="cuda"):
            output = equitune_model(input_ids, metadata)
            loss = loss_fn(output, label)

        val_loss += loss.item()
    val_loss = val_loss / len(validation_set)
    wandb.log({"validation loss": val_loss})

pbar = tqdm(range(num_epochs * len(train_set)))

for epoch in range(num_epochs):  # Train for 5 epochs
    total_loss = 0
    for epoch_i, batch in enumerate(train_loader):
        input_ids, label, metadata = batch
        input_ids = input_ids.to(device)
        label = label.unsqueeze(dim=0).to(device)
        pbar.update(1)
        if input_ids.shape[1] > 900:
            continue 
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            output = equitune_model(input_ids, metadata)
            loss = loss_fn(output, label)
        wandb.log({"training loss": loss})
        # wandb.log({"sequence length": input_ids.shape[1]})
        step += 1
        total_loss += loss.item()
        if step % eval_every == 0:
            get_val()
        if step % checkpoint_every == 0:
            save_dir = f"{MODEL_DIR}/model_{step}"
            torch.save({
                'epoch': epoch, 
                'model_state_dict': equitune_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_dir)
        loss.backward()
        optimizer.step()
    wandb.log({"epoch training loss": total_loss})



save_dir = f"{MODEL_DIR}/final_model"
torch.save({
    'epoch': epoch, 
    'model_state_dict': equitune_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, save_dir)