import os
import pickle
import itertools
import time
import argparse

import wandb
import gc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--wandb', action='store_true') 
    parser.add_argument('--data_path', type=str, default="/home/albertjan/equitune/data/dataset_4perms_function_insertion_preprocessed.pkl")
    parser.add_argument('--save_dir', type=str, default="/home/albertjan/equitune/code-symmetry/results/equitune/")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()   
    return args 


def main(args):
    @torch.no_grad()
    def evaluate_model(dataset, dataloader):
        total_loss = 0
        correct_samples = 0
        total_samples = 0
        for batch in dataloader:
            input_ids, label, metadata = batch["input_ids"], batch["label"], batch["metadata"]
            input_ids = input_ids.to(device)
            label = label.to(device)
            with torch.autocast(device_type="cuda"):
                output = equitune_model(input_ids, metadata)
                loss = loss_fn(output, label)

            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct_samples += (predicted == label).sum().item()
            total_samples += label.shape[0]
        total_loss = total_loss / len(dataloader)
        accuracy = correct_samples / total_samples
        return total_loss, accuracy 

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)

    # Initialize wandb
    if args.wandb:
        run = wandb.init(
            reinit=True,
            project="equitune",
            name="Equitune Train 3",
            config={
                "learning_rate": args.lr,
                "epochs": args.num_epochs,
            },
        )

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

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

    data =  CodePermutationDataset(dataset=data)

    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size
    train_set, validation_set, test_set = random_split(
        data, [train_size, val_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=None, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=None, shuffle=True)

    class CodePermutationEquitune(nn.Module): 

        def __init__(self, base_model="deepseek-ai/deepseek-coder-1.3b-base", num_permutations=4):
            super(CodePermutationEquitune, self).__init__() 
            self.base_model = AutoModelForCausalLM.from_pretrained(base_model) #torch_dtype=torch.float16)
            self.num_permutations = num_permutations
            module_dict = dict(self.base_model.named_modules())

            # second_to_last_attn_layer = module_dict['model.layers.22']
            # second_to_last_attn_layer.register_forward_hook(self.permute_output_hook)

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
    optimizer = torch.optim.AdamW(equitune_model.parameters(), lr=args.lr)

    step = 0
    equitune_model.train()

    pbar = tqdm(range(args.num_epochs * len(train_set)))

    for epoch in range(args.num_epochs):  
        total_loss = 0
        correct_samples = 0
        total_samples = 0 

        for batch in train_loader:
        # for epoch_i, batch in enumerate(sorted(train_loader, key=lambda x: x["input_ids"].shape[1], reverse=True)):
            input_ids, label, metadata = batch["input_ids"], batch["label"], batch["metadata"]
            input_ids = input_ids.to(device)
            label = label.to(device)

            pbar.update(1)
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda"):
                output = equitune_model(input_ids, metadata)
                loss = loss_fn(output, label)

            _, predicted = torch.max(output, 1)
            correct_samples += (predicted == label).sum().item()
            total_samples += label.shape[0]
            
            if args.wandb: wandb.log({"training loss": loss.item()}, step=step) 
            step += 1
            total_loss += loss.item()
            
            if step % args.eval_every == 0:
                val_loss, val_accuracy = evaluate_model(validation_set, validation_loader)
                if args.wandb: wandb.log({"validation loss": val_loss, "validation accuracy": val_accuracy}, step=step)
            
            if step % args.save_every == 0:
                save_path = os.path.join(args.save_dir, f"model_{step}")
                torch.save({
                    'epoch': epoch, 
                    'model_state_dict': equitune_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_path)
                        
            loss.backward()
            optimizer.step()

        total_loss = total_loss / len(train_loader)
        accuracy = correct_samples / total_samples
        if args.wandb: wandb.log({"epoch training loss": total_loss, "epoch accuracy": accuracy}, step=step)

    test_loss, test_accuracy = evaluate_model(test_set, test_loader)
    save_path = os.path.join(args.save_dir, "final_model")
    if args.wandb: wandb.log({"final_test_loss": test_loss, "final_test_accuracy": test_accuracy}, step=step)
    torch.save({
        'epoch': epoch, 
        'model_state_dict': equitune_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss
    }, save_path)

if __name__ == "__main__":
    args = parse_args(None)
    main(args)

