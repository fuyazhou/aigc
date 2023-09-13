from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, GPT2Config
from transformers import DataCollatorForLanguageModeling
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_scheduler

# from tqdm.notebook import tqdm

# accelerator = Accelerator(fp16=True)
accelerator = Accelerator()

context_length = 1280
batch_size = 8
weight_decay = 0.1
lr = 5e-4
num_train_epochs = 1000
output_dir = "scGPT-gpt"
gradient_accumulation_steps = 2
eval_steps = 5

import sys
from datasets import load_dataset

sys.path.append('/home/lushi02/scGPT2/scGPT')

import scgpt
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab

dataset = load_dataset("parquet", data_files={'train': '/home/lushi02/scGPT2/data/binned/databanks_binned.parquet'})
# vocab = scgpt.tokenizer.GeneVocab.from_file('/home/lushi02/scGPT2/singlecell_gpt/data/default_census_vocab.json')
vocab = scgpt.tokenizer.GeneVocab.from_file('/home/lushi02/scGPT2/data/8.scb/gene_vocab.json')

cls_id = dataset['train'][0]['gene_ids'][0]
pad_id = dataset['train'][0]['gene_ids'][-1]

vocab_size = len(vocab)

seq_length = len(dataset['train'][0]['gene_ids'])
dataset = dataset.remove_columns(['target_values'])
tokenized_dataset = dataset['train'].rename_column("gene_ids", "position_ids").rename_column("target_values",
                                                                                             "input_ids")
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

tokenized_dataset.set_format("torch")
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=batch_size)

# config = AutoConfig.from_pretrained(
#     "gpt2",
#     vocab_size=vocab_size,
#     n_ctx=context_length,
#     bos_token_id=cls_id,
#     eos_token_id=pad_id,
# )
config = GPT2Config(
    vocab_size=vocab_size + 3,
    n_ctx=context_length,
    bos_token_id=cls_id,
    eos_token_id=pad_id,
)

config.n_positions = context_length

model = GPT2LMHeadModel(config)

# embed_dim = 768
# model.transformer.wpe = nn.Embedding(len(vocab_size) + 3, embed_dim)


model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")


def keytoken_weighted_loss(inputs, logits, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    # weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
    #     axis=[0, 2]
    # )
    # weights = alpha * (1.0 + weights)
    # Calculate weighted average
    # weighted_loss = (loss_per_sample * weights).mean()
    weighted_loss = (loss_per_sample).mean()
    return weighted_loss


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    # loss = torch.mean(torch.cat(losses))
    loss = torch.mean(torch.tensor(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


optimizer = AdamW(get_grouped_params(model), lr=lr)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)

model.train()
completed_steps = 0
for epoch in range(num_train_epochs):
    print(f"num_train_epochs:  {epoch}")
    for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        # for step, batch in (enumerate(train_dataloader, start=1)):
        logits = model(batch["input_ids"]).logits
        # logits = model(input_ids = batch["input_ids"],position_ids=something).logits

        loss = keytoken_weighted_loss(batch["input_ids"], logits)
        # print(f"step {step}")
        # print(f"train_dataloader {len(train_dataloader)}")
        if step % 15 == 0:
            accelerator.print(
                {
                    # "lr": get_lr(),
                    # "samples": step * samples_per_step,
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                }
            )
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            # print(f"gradient_accumulation_steps {step}")
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            # if accelerator.is_main_process:
            #     tokenizer.save_pretrained(output_dir)
            #     repo.push_to_hub(
            #         commit_message=f"Training in progress step {step}", blocking=False
            #     )
