

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

app = FastAPI()
# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

model_path = r"C:\Users\Kishor\QASystem\outputs"
model = load_model(model_path)
tokenizer = load_tokenizer(model_path)


def generate_text(sequence, max_new_tokens):
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    input_length = ids.size(1)
    max_length = input_length + max_new_tokens
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)



@app.get("/answer/{prompt}")
async def root(prompt: str):
    print(prompt)
    result = {"Answer": generate_text("Question: "+prompt+" answer: ", 30).split('answer: ')[1]}
    print(result)
    return result