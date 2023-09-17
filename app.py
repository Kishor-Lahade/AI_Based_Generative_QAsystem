import gradio as gr

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer



model_path = r'outputs'
model1 = load_model(model_path)
tokenizer1 = load_tokenizer(model_path)

def generate_text(sequence, max_new_tokens):
    ids = tokenizer1.encode(f'{sequence}', return_tensors='pt')
    input_length = ids.size(1)
    max_length = input_length + max_new_tokens
    final_outputs = model1.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model1.config.eos_token_id
    )
    return tokenizer1.decode(final_outputs[0], skip_special_tokens=True)


def root(prompt: str):
    print(prompt)
    return generate_text("Question: " + prompt + "Answer: ", 35).split('Answer: ')[1]
def greet(name):
    return "Hello " + name + "!!"

iface = gr.Interface(fn=root, inputs="text", outputs="text")
iface.launch()