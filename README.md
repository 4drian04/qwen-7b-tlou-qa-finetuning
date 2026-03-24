# 🎮 The Last of Us QA – Fine-Tuning with QLoRA
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-TheLastOfUs--QA-blue)](https://huggingface.co/adriangg04/TheLastOfUs-QA)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-The%20Last%20of%20Us-yellow)](https://huggingface.co/datasets/adriangg04/the-last-of-us-instruction-dataset)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Demo-Space-green)](https://huggingface.co/spaces/adriangg04/Qwen-Finetuned-TheLastofUs)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-LLM-orange)](https://huggingface.co/docs/transformers/index)

This project implements a complete **fine-tuning pipeline for a language model** using **QLoRA (4-bit quantization + LoRA)** on the base model:

👉 `Qwen/Qwen2.5-7B-Instruct`

The goal is to specialize the model to answer questions about the universe of **The Last of Us**, evaluating its performance **before and after training**.


---

# 📌 Table of Contents

- [📖 Description](#-description)
- [⚙️ Project Pipeline](#-project-pipeline)
- [🧪 Model Evaluation](#-model-evaluation)
- [🧠 Fine-Tuning (QLoRA)](#-fine-tuning-qlora)
- [🔀 Model Merge](#-model-merge)
- [📊 Results](#-results)
- [📁 Project Structure](#-project-structure)
- [🚀 How to Use](#-how-to-use)
- [🛠️ Technologies Used](#-technologies-used)
- [📂 Dataset](#-dataset)
- [🤖 Fine-Tuned Model](#-fine-tuned-model)
- [🚀 Demo (Hugging Face Space)](#-demo-hugging-face-space)

---

# 📖 Description

This repository shows a complete Fine-Tuning workflow:

1. Baseline model evaluation
2. Training with a specific dataset
3. Integration of LoRA adapters
4. Merging the final model
5. Re-evaluation with the same tests

The dataset used was created specifically for this project and contains conversational instructions about *The Last of Us*.


---

# ⚙️ Project Pipeline

Baseline Model → Initial Evaluation → Fine-Tuning (QLoRA) → Merge → Final Evaluation → Comparison

---

# 🧪 Model Evaluation

Before fine-tuning, a battery of 15 questions about the universe of *The Last of Us* is run.

### Features:

- Prompts in conversational format (system + user)
- Generation with `transformers.pipeline`
- Results saved in JSON

### Example:

```python
messages = [
{"role": "system", "content": "You are an expert on The Last of Us"},
{"role": "user", "content": prompt}
]
```

## 🧠 Fine-Tuning (QLoRA)

Efficient fine-tuning is applied using:

- 4-bit quantization (NF4)
- LoRA (Low-Rank Adaptation)

### 🔧 LoRA Configuration

```python

r = 16
lora_alpha = 32
lora_dropout = 0.05
```

### 🎯 Layers Adapted

```python
["q_proj", "k_proj", "v_proj", "o_proj",
"gate_proj", "up_proj", "down_proj"]
```

### ⚡ Key Features

- Use of bits and bytes to reduce memory
- gradient_accumulation to simulate large batches
- Scheduler: cosine
- Logging with TensorBoard

---

## 🔀 Model Merge

After training, the following is performed:

```python
model.merge_and_unload()
```

## 🔧 This allows:

- Integrating LoRA adapters into the base model
- Using the final model without PEFT dependencies
- Simplifying inference

---

## 🧪 Final Evaluation

The same test suite is run on the adjusted model:

- Same Question set
- Same generation configuration
- Results saved in JSON

---

## 📊 Results

| Model | Correct Answers | Accuracy |

|--------------------|----------|----------|

| Base model | 5 / 15 | 33.3% |

| Fine-Tuned model | 13 / 15 | 86.7% |


---

## 📈 Observed Improvements

- Significant reduction in hallucinations

- More accurate responses

- Better understanding of the narrative context

- Greater consistency in responses

## 📁 Project Structure

```
├── Test_Modelo_Base.py
├── Fine-Tuning.py
├── Merge_Models.py
├── Test_FineTuning_Model.py
├── tlou_dataset.json
├── Upload_Dataset_To_HuggingFace.py
├── Upload_Model_To_HuggingFace.py
├── Memoria_Tecnica_Fine_Tuning_TLOU.pdf
└── README.md

```

---

## 🚀 How to Use

1. Clone repository

```bash
git clone https://github.com/your-username/https://github.com/4drian04/tlou-qa-finetuning.git
cd tlou-qa-finetuning
```

2. Install dependencies

```bash
pip install transformers datasets peft trl bitsandbytes accelerate evaluate
```

3. Run baseline

```bash
python Test_Base_Model.py
```

4. Train model

```bash
python Fine-Tuning.py
```

5. Merge

```bash
python Merge_Models.py
```

6. Evaluate fitted model

```bash
python Test_FineTuning_Model.py
```

## 🛠️ Technologies Used

- Transformers
- Datasets
- PEFT (LoRA)
- TRL (SFTTrainer)
- BitsAndBytes
- PyTorch
- TensorBoard

---

## 📂 Dataset

The dataset used for Fine-Tuning is available on Hugging Face:

👉 https://huggingface.co/datasets/adriangg04/the-last-of-us-instruction-dataset

This dataset was created specifically for this project and contains conversational examples (system, user, assistant) focused on the universe of *The Last of Us*.

Furthermore, the dataset was generated using web scraping techniques with the following script:

```bash
scrappingTLOU.py
```
This script collects relevant information from the universe of *The Last of Us*, which is then processed and transformed into the instruction format used in Fine-Tuning.

---

## 🤖 Fine-Tuned Model

The fine-tuned model is available on Hugging Face:

👉 https://huggingface.co/adriangg04/TheLastOfUs-QA

This model is a fine-tuned version of Qwen2.5-7B-Instruct, optimized to answer questions about *The Last of Us*.

---

## 🚀 Demo (Hugging Face Space)

You can try the model directly from the following Space:

👉 https://huggingface.co/spaces/adriangg04/Qwen-Finetuned-TheLastofUs

This Space allows you to interact with the model in real time without needing to install it locally.

---

## 👨‍💻 Author

**Adrián García García** - [LinkedIn](https://www.linkedin.com/in/adri%C3%A1n-garc%C3%ADa-garc%C3%ADa-6ab399333/)

---

## ⭐ Contributions

If you're interested in the project, feel free to give it a star or contribute!
