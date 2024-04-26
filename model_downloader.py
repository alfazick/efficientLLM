from transformers import GPT2LMHeadModel, GPT2Tokenizer

def download_and_save_model(model_name, save_directory):
    # Load model and tokenizer from Hugging Face Hub
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Save the model and the tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

# Usage
model_name = "gpt2"  # You can choose other models like gpt2-medium, gpt2-large, etc.
save_directory = "./models/gpt2"
download_and_save_model(model_name, save_directory)
