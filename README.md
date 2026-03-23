# 🎮 The Last of Us QA – Fine-Tuning with QLoRA

Este proyecto implementa un pipeline completo de **Fine-Tuning de un modelo de lenguaje** usando **QLoRA (4-bit quantization + LoRA)** sobre el modelo base:

👉 `Qwen/Qwen2.5-7B-Instruct`

El objetivo es especializar el modelo para responder preguntas sobre el universo de **The Last of Us**, evaluando su rendimiento **antes y después del entrenamiento**.

---

# 📌 Tabla de contenidos

- [📖 Descripción](#-descripción)
- [⚙️ Pipeline del proyecto](#️-pipeline-del-proyecto)
- [🧪 Evaluación del modelo](#-evaluación-del-modelo)
- [🧠 Fine-Tuning (QLoRA)](#-fine-tuning-qlora)
- [🔀 Merge del modelo](#-merge-del-modelo)
- [📊 Resultados](#-resultados)
- [📁 Estructura del proyecto](#-estructura-del-proyecto)
- [🚀 Cómo usar](#-cómo-usar)
- [🛠️ Tecnologías utilizadas](#️-tecnologías-utilizadas)

---

# 📖 Descripción

Este repositorio muestra un flujo completo de trabajo en Fine-Tuning:

1. Evaluación del modelo base  
2. Entrenamiento con dataset específico  
3. Integración de adaptadores LoRA  
4. Merge del modelo final  
5. Re-evaluación con los mismos tests  

El dataset utilizado ha sido creado específicamente para este proyecto y contiene instrucciones en formato conversacional sobre *The Last of Us*.

---

# ⚙️ Pipeline del proyecto

Modelo base → Evaluación inicial → Fine-Tuning (QLoRA) → Merge → Evaluación final → Comparación


---

# 🧪 Evaluación del modelo

Antes del Fine-Tuning, se ejecuta una batería de 15 preguntas sobre el universo de *The Last of Us*.

### Características:

- Prompts en formato conversación (system + user)  
- Generación con `transformers.pipeline`  
- Resultados guardados en JSON  

### Ejemplo:

```python
messages = [
    {"role": "system", "content": "You are an expert on The Last of Us"},
    {"role": "user", "content": prompt}
]
```

## 🧠 Fine-Tuning (QLoRA)

Se aplica Fine-Tuning eficiente usando:

- Cuantización en 4 bits (NF4)
- LoRA (Low-Rank Adaptation)

### 🔧 Configuración LoRA

```python
r = 16
lora_alpha = 32
lora_dropout = 0.05
```

### 🎯 Capas adaptadas

```python
["q_proj", "k_proj", "v_proj", "o_proj",
 "gate_proj", "up_proj", "down_proj"]
```

### ⚡ Características clave

- Uso de bitsandbytes para reducir memoria
- gradient_accumulation para simular batches grandes
- Scheduler: cosine
- Logging con TensorBoard

---

## 🔀 Merge del modelo

Tras el entrenamiento, se realiza:

```python
model.merge_and_unload()
```

## 🔧 Esto permite

- Integrar los adaptadores LoRA en el modelo base  
- Usar el modelo final sin dependencias de PEFT  
- Simplificar la inferencia  

---

## 🧪 Evaluación final

Se ejecuta la misma batería de tests sobre el modelo ajustado:

- Mismo conjunto de preguntas  
- Misma configuración de generación  
- Resultados guardados en JSON  

---

## 📊 Resultados

| Modelo             | Aciertos | Precisión |
|--------------------|----------|----------|
| Base model         | 5 / 15   | 33.3%    |
| Fine-Tuned model   | 13 / 15  | 86.7%    |

---

## 📈 Mejoras observadas

- Reducción significativa de alucinaciones  
- Respuestas más precisas  
- Mejor comprensión del contexto narrativo  
- Mayor consistencia en respuestas  

## 📁 Estructura del proyecto


---

## 🚀 Cómo usar

1. Clonar repositorio

```bash
git clone https://github.com/tu-usuario/https://github.com/4drian04/tlou-qa-finetuning.git
cd tlou-qa-finetuning
```

2. Instalar dependencias

```bash
pip install transformers datasets peft trl bitsandbytes accelerate evaluate
```

3. Ejecutar baseline

```bash
python Test_Modelo_Base.py
```

4. Entrenar modelo

```bash
python Fine-Tuning.py
```

5. Hacer merge

```bash
python Merge_Models.py
```

6. Evaluar modelo ajustado

```bash
python Test_FineTuning_Model.py
```

## 🛠️ Tecnologías utilizadas

- Transformers  
- Datasets  
- PEFT (LoRA)  
- TRL (SFTTrainer)  
- BitsAndBytes  
- PyTorch  
- TensorBoard  

---

## 👨‍💻 Autor

**Adrián García García** - [LinkedIn](https://www.linkedin.com/in/adri%C3%A1n-garc%C3%ADa-garc%C3%ADa-6ab399333/)

---

## ⭐ Contribuciones

Si te interesa el proyecto, ¡no dudes en darle una estrella o contribuir!
