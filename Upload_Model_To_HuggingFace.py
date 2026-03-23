from huggingface_hub import login, upload_folder

# Esto nos permite logearnos en Hugging Face para poder subir el modelo
login()

# Subimos los archivos del modelo
upload_folder(folder_path="./qwen-merged", repo_id="adriangg04/TheLastOfUs-QA", repo_type="model")