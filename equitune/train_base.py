import os
import pickle
import argparse

import wandb
from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--save_every', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--wandb', action='store_true') 
    parser.add_argument('--data_path', type=str, default="/home/albertjan/equitune/data/dataset_4perms_function_insertion_preprocessed.pkl")
    parser.add_argument('--save_dir', type=str, default="/home/albertjan/equitune/code-symmetry/results/baseline/")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()   
    return args 

def main(args):
    @torch.no_grad()
    def evaluate_model(dataset, dataloader):
        total_loss = 0
        correct_samples = 0
        total_samples = 0
        for batch in dataloader:
            with torch.autocast(device_type="cuda"):
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
                output = model(input_ids, attention_mask=attention_mask)
                logits = output.logits # (batch_size, seq_len, vocab_size)
                last_non_padded_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(logits.shape[0])
                last_token_logits = logits[batch_indices, last_non_padded_indices] # (batch_size, vocab_size)
                loss = loss_fn(last_token_logits, labels)

            _, predicted = torch.max(last_token_logits, 1)
            correct_samples += (predicted == labels).sum().item()
            total_samples += labels.shape[0] 

            total_loss += loss.item()
        total_loss = total_loss / len(dataloader)
        accuracy = correct_samples / total_samples
        return total_loss, accuracy 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_token_id = tokenizer.pad_token_id


    # Initialize wandb
    if args.wandb:
        run = wandb.init(
            reinit=True,
            project="equitune",
            name="Baseline Train 2",
            config={
                "learning_rate": args.lr,
                "epochs": args.num_epochs,
            },
        )

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    # discard permutations for training the baseline
    # data = [{"input_ids": d["input_ids"][0], "label": d["label"]} for d in data]  

    data = [{"input_ids": d["input_ids"][i], "label": d["label"]} for d in data for i in range(4)]

    class CodeDataset(Dataset):

        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            return item["input_ids"], item["label"]

    def collate_fn(batch, pad_token_id):
        inputs, labels = zip(*batch)
        # print("Inputs:", inputs)
        # print("Labels:", labels)
        # print("Input lengths:", [len(input) for input in inputs])
        # print("Label lengths:", [len(label) for label in labels])

        input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
        attention_mask = (input_ids != pad_token_id).long()
        
        labels = torch.stack(labels).squeeze() if len(labels) > 1 else torch.tensor(labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    code_dataset = CodeDataset(dataset=data)

    train_size = int(0.8 * len(code_dataset))
    val_size = int(0.1 * len(code_dataset))
    test_size = len(code_dataset) - train_size - val_size
    train_set, validation_set, test_set = random_split(
        code_dataset, [train_size, val_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=lambda x: collate_fn(x, pad_token_id), shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, collate_fn=lambda x: collate_fn(x, pad_token_id), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=lambda x: collate_fn(x, pad_token_id), shuffle=True)

    model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    model.train()

    pbar = tqdm(range(args.num_epochs * len(train_loader)))
    for epoch in range(args.num_epochs):  
        total_loss = 0
        correct_samples = 0
        total_samples = 0 
        for batch in train_loader:
        # for epoch_i, batch in enumerate(sorted(train_loader, key=lambda x: x["input_ids"].shape[1], reverse=True)):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            pbar.update(1)
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda"):
                output = model(input_ids, attention_mask=attention_mask)
                logits = output.logits # (batch_size, seq_len, vocab_size)
                last_non_padded_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(logits.shape[0])
                last_token_logits = logits[batch_indices, last_non_padded_indices]
                loss = loss_fn(last_token_logits, labels)

            _, predicted = torch.max(last_token_logits, 1)
            correct_samples += (predicted == labels).sum().item()
            total_samples += labels.shape[0] 

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
                    'model_state_dict': model.state_dict(),
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
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss
    }, save_path)


if __name__ == "__main__":
    args = parse_args(None)
    main(args)