import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
import os
import evaluate

os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs" # Escribimos está variable de entorno para poder hacer uso de TensorBoard

# Configuración inicial
model_name = "Qwen/Qwen2.5-7B-Instruct" # Escribimos el nombre del modelo base
OUTPUT_DIR = "./qwen-finetuning"


def load_quantized_model(model_name):
    """Carga el modelo en 4 bits (QLoRA)."""
    # Cargar el modelo en 4-bits para hacer QLoRa
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, # Nos permite cargar el modelo en 4 bits
        bnb_4bit_quant_type="nf4", # Tipo de cuantización
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True # Cuantización de las constantes de escalado
    )

    # Cargamos el modelo base
    model = AutoModelForCausalLM.from_pretrained(
        model_name, # Indicamos el nombre del modelo
        device_map="auto",
        quantization_config=quant_config, # Le aplicamos la cuantización
        trust_remote_code=True
    )

    return model


def load_tokenizer(model_name):
    """Carga y configura el tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name) # Obtenemos el tokenizer del modelo en cuestión
    tokenizer.pad_token = tokenizer.eos_token # se define qué token se usa para hacer padding, en este caso, fin de secuencia ("End of Sequence")
    tokenizer.padding_side = "right" # Esto define en qué lado se añade el padding. "right" significa que se añade al final de la secuencia.
    return tokenizer


def format_chat(example, tokenizer):
    """
    Formatea una conversación en formato chat al template esperado por el modelo.

    Esta función toma un ejemplo de un dataset que contiene una lista de mensajes
    en el campo "messages" (con roles como "system", "user" y "assistant") y utiliza
    el tokenizer del modelo para convertir dicha conversación al formato de texto
    específico que el modelo espera para su entrenamiento.

    Parámetros
    ----------
    example : dict
        Diccionario que representa una fila del dataset y que contiene la clave
        "messages" con la conversación estructurada.

    Returns
    -------
    dict
        Diccionario con una nueva clave "text" que contiene la conversación
        formateada según el chat template del modelo.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


def get_lora_config():
    """Define la configuración LoRA."""
    # LoRA config
    lora_config = LoraConfig(
        r=16, # rank, que representa la dimensión de las matrices de bajo rango Un $r$ más alto permite capturar patrones más complejos pero aumenta los parámetros entrenables y el uso de memoria.
        lora_alpha=32, # Determina cuánto peso tienen las actualizaciones de LoRA sobre los pesos originales del modelo base, normalmente es rx2
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #  indica que capas queremos adaptar, en este caso se ha elegido la capa de atención y las capas del perceptrón multicapa
        lora_dropout=0.05, # permite aplicar regularización durante el entrenamiento, reduciendo el riesgo de que el modelo se sobreajuste
        bias="none", # indica si se adaptan los ‘bias’ del modelo base
        task_type="CAUSAL_LM" #  indica el tipo de modelo al que le vamos a hacer Fine-Tuning
    )
    return lora_config


def prepare_model(model, lora_config):
    """Prepara el modelo para entrenamiento QLoRA."""
    # Prepara el modelo para entrenamiento en baja precisión (k-bit), ajustando ciertos
    # parámetros y capas para que el entrenamiento con cuantización sea estable y eficiente
    # en memoria.
    model = prepare_model_for_kbit_training(model)

    # Aplica PEFT (Parameter-Efficient Fine-Tuning) al modelo usando la configuración LoRA.
    # Esto inserta adaptadores LoRA en capas específicas del modelo para entrenarlo
    # modificando solo un pequeño número de parámetros, reduciendo el coste computacional.
    model = get_peft_model(model, lora_config)

    return model


def print_trainable_parameters(model):
    """Muestra estadísticas de parámetros entrenables."""
    # Veamos cuantos parámetros vamos a entrenar con respecto al total
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params,}/{all_params,} = {100 * trainable_params / all_params:.2f}%")


def load_and_prepare_datasets(tokenizer):
    """Carga y formatea el dataset."""
    dataset = load_dataset("adriangg04/the-last-of-us-instruction-dataset") # Cargamos el dataset que hemos subido a Hugging Face

    # Formateamos tanto el train como el test con el formato chat
    train_dataset = dataset["train"].map(lambda x: format_chat(x, tokenizer))
    eval_dataset = dataset["test"].map(lambda x: format_chat(x, tokenizer))

    return train_dataset, eval_dataset


def get_training_args():
    """Define los argumentos de entrenamiento."""
    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, # indica el número de ejemplos procesados simultáneamente por cada GPU en un paso de entrenamiento
        gradient_accumulation_steps=4, # esto permite simular un batch más grande antes de actualizar los pesos sin necesidad de una GPU con muchísima memoria
        learning_rate=1e-4, # indica la tasa de aprendizaje
        lr_scheduler_type="cosine", # indica cómo cambia la tasa de aprendizaje según las circunstancias
        num_train_epochs=3, # número de veces que recorre el ldataset
        bf16=True, # Se usa bf16 como hemos indicado en la cuantización
        gradient_checkpointing=True, # Guarda checkpoint por cada 'epoch'
        logging_steps=10, # útil para ver progreso
        save_strategy="epoch", # guarda checkpoint al final de cada epoch
        eval_strategy="epoch", # evalúa al final de cada epoch
        load_best_model_at_end=True, # Guarda el mejor modelo de los checkpoints guardados
        metric_for_best_model="eval_loss", # indica la métrica que debe de medir para quedarse con el mejor modelo
        optim="paged_adamw_8bit", # optimizador
        warmup_ratio=0.03, # los primeros 3% de los pasos totales, el learning rate cambia muy lentamente, de esta manera se estabiliza el entrenamiento y no hay cambios bruscos
        report_to="tensorboard" # permite ver gráficamente algunos valores durante el entrenamiento gracias a TensorBoard
    )
    return training_args


def train():
    """Pipeline principal de entrenamiento."""

    # Obtenemos en primer lugar el modelo cuantizado y el tokenizer correspondiente
    model = load_quantized_model(model_name)
    tokenizer = load_tokenizer(model_name)

    # Configuramos LORA y preparamos el modelo con dicha configuración
    lora_config = get_lora_config()
    model = prepare_model(model, lora_config)

    print_trainable_parameters(model)

    # Obtenemos el dataset de entrenamiento y test
    train_dataset, eval_dataset = load_and_prepare_datasets(tokenizer)
    training_args = get_training_args() # Configuramos los distintos hiperparámetros

    # Se define la clase para hacer Fine-Tuning
    trainer = SFTTrainer(
        model=model, # Indicamos el modelo
        args=training_args, # Indicamos todos los argumentos anteriores
        train_dataset=train_dataset, # Se indica el dataset de entrenamiento
        eval_dataset=eval_dataset # Se indica el dataset de evaluación
    )

    trainer.train() # Se empieza el entrenamiento
    trainer.save_model(OUTPUT_DIR) # Una vez entrenado, se guarda el mdoelo en el directorio
    tokenizer.save_pretrained(OUTPUT_DIR) # Se guarda también el tokenizador
    print(f"LoRa adapter save to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()