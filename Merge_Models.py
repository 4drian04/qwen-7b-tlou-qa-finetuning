from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModel
import torch

print("Obtaining the base model of Qwen...")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",torch_dtype=torch.bfloat16) # Obtenemos el modelo base 

print("Obtaining the tokenizer from the base model...")
tokenizer =  AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct") # Obtenemos el tokenizer del modelo base

# Esto nos permite cargar un modelo ajustado sobre un modelo base 

model = PeftModel.from_pretrained(base_model, "qwen-finetuning")

print("Merged models...")
merged_model = model.merge_and_unload() # Hacemos merge para unir los adaptadores del modelo ajustado al modelo base

# Guardamos tanto el modelo como el tokenizer del modelo ajustado
merged_model.save_pretrained("./qwen-merged")

tokenizer.save_pretrained("./qwen-merged")

print("Merged model saved no adapter needed at inference time")