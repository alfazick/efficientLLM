import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# pad on the left, new tokens on the right
tokenizer.padding_side = "left"
tokenizer.truncate_side = "left"

# multiple prompts of varying lengths to send to the model at once
prompts = [
    "The quick brown fox jumped over the",
    "The rain in Spain falls",
    "What comes up must",
]

# note: padding=True ensures the padding token will be inserted into the tokenized tensors
inputs = tokenizer(prompts, padding=True, return_tensors="pt")


# init_batch(requests):
# Initializes the batch with all required initial states based on user input requests.
# Outputs a batch setup with tokenized input IDs, position IDs, attention masks,
# and the initial prompts stored in responses.

def init_batch(requests):
    prompts = [r[0] for r in requests]
    inputs = tokenizer(prompts, padding=True,return_tensors="pt")

    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    return {
        "position_ids":position_ids,
        "responses":copy.copy(prompts),
        "tokens_remaining": [r[1] for r in requests],
        **inputs
    }



# generate_batch_tokens_with_past(inputs):
# Directly interacts with the model under a no gradient tracking context for inference:
# Model Call: Passes the inputs through the model to get logits.
# Token Prediction: Extracts the last logits to determine the most probable next token using argmax.
# State Management: Retrieves updated past key values which are crucial 
# for sequential token generation in transformer-based models.

def generate_batch_tokens_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits 
    last_logits = logits[:,-1,:]
    next_token_ids = last_logits.argmax(dim=1)
    return next_token_ids, outputs.past_key_values


# generate_next_token(batch):
# Central function in the generation loop:
# Copies the batch to modify it without affecting the original during operations.
# Excludes the responses and tokens_remaining as these are not needed for the token generation process.

def generate_next_token(batch):
    inputs = copy.copy(batch)
    inputs.pop("responses")
    inputs.pop("tokens_remaining")

    next_token_ids,past_key_values = generate_batch_tokens_with_past(inputs)

    next_tokens = tokenizer.batch_decode(next_token_ids)

    return get_next_inputs(batch,next_token_ids, past_key_values, next_tokens)

# get_next_inputs(...):
# Receives updated tokens and past key values:
# Updates input_ids: By incorporating new token IDs.
# Updates position_ids and attention_mask: To reflect the addition of new tokens.
# Appends New Tokens: To the ongoing responses for each prompt.
# Decrements tokens_remaining: Reflects one less token needed per prompt.

def get_next_inputs(batch,next_token_ids, past_key_values, next_tokens):
    return {
        "inputs_ids":next_token_ids.reshape((-1,1)),
        "position_ids":batch["position_ids"][:,-1].unsqueeze(-1) + 1,
        "attention_mask":torch.cat([
            batch["attention_mask"],
            torch.ones((next_token_ids.shape[0],1)),
        ],dim=1),
        "past_key_values":past_key_values,
        "responses": [
            r1 + r2 for r1,r2 in zip(batch["responses"],next_tokens)
        ],
        "tokens_remaining": [v-1 for v in batch["tokens_remaining"]],
    }

