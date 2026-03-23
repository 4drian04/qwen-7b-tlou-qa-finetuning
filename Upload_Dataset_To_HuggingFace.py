from datasets import load_dataset
from huggingface_hub import login

login() # Nos logeamos en Hugging Face para subir el dataset
dataset = load_dataset("json", data_files="./tlou_dataset.jsonl") # Cargamos el dataset de nuestro equipo

dataset = dataset["train"].train_test_split(test_size=0.1) # Lo dividmos en train y test para subirlo a Hugging Face

dataset.push_to_hub("adriangg04/the-last-of-us-instruction-dataset") # Lo subimos a nuestro repositorio