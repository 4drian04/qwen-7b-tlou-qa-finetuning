from transformers import pipeline
import json
import csv
from datetime import datetime

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def create_pipeline(model_name):
    """Crea el pipeline de generación de texto."""
    # Crear pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto"
    )
    return pipe


def get_tests():
    """Define la batería de pruebas."""
    # Batería de pruebas
    tests = [
        {
            "id": "test_001",
            "prompt": "What does Ellie find out when she returns to her house on The Last of Us?"
        },
        {
            "id": "test_002",
            "prompt": "Why does Tommy go to Santa Barbara ahead of Ellie in The Last of Us?"
        },
        {
            "id": "test_003",
            "prompt": "What happened in the event 'Ellie goes to Seattle' in The Last of Us?"
        },
        {
            "id": "test_004",
            "prompt": "Why does Joel kill Marlene in The Last of Us?"
        },
        {
            "id": "test_005",
            "prompt": "How does the rat king differ from other infected in The Last of Us?"
        },
        {
            "id": "test_006",
            "prompt": "How do stalkers compare to runners in terms of strength in The Last of Us?"
        },
        {
            "id": "test_007",
            "prompt": "What is the main reason for Ellie's journey to Seattle in The Last of Us?"
        },
        {
            "id": "test_008",
            "prompt": "What happens to Sarah in Texas on The Last of Us?"
        },
        {
            "id": "test_009",
            "prompt": "What happens to Sam after he is scratched by an infected in The Last of Us?"
        },
        {
            "id": "test_010",
            "prompt": "What happened to Ellie and Joel after they left the University on The Last of Us?"
        },
        {
            "id": "test_011",
            "prompt": "What is the relationship between Jerry and Abby in The Last of Us?"
        },
        {
            "id": "test_012",
            "prompt": "How did Abby react to Owen's death in The Last of Us?"
        },
        {
            "id": "test_013",
            "prompt": "What motivated Abby to spare Ellie's life in The Last of Us?"
        },
        {
            "id": "test_014",
            "prompt": "Why do Clickers develop a form of echolocation in The Last of Us?"
        },
        {
            "id": "test_015",
            "prompt": "What are the four categories of collectibles in The Last of Us?"
        }
    ]
    return tests


def generate_response(pipe, test):
    """Genera la respuesta del modelo para un test."""
    # Montamos el mensaje con los roles correspondientes (system y user)
    messages = [
        {"role": "system", "content": "You are an expert on The Last of Us"},
        {"role": "user", "content": test["prompt"]}
    ]

    # Le pasamos el prompt al modelo para que responda
    response = pipe(
        messages,
        max_new_tokens=200,
        temperature=0.7
    )

    # Obtenemos la respuesta
    answer = response[0]["generated_text"][-1]["content"]

    return answer


def run_tests(pipe, tests):
    """Ejecuta todos los tests."""
    results = [] # Guardamos los resultados en una lista de diccionarios

    for test in tests: # Recorremos las distintas preguntas
        answer = generate_response(pipe, test)

        # Creamos un diccionario con el prompt, id y la respuesta
        result = {
            "test_id": test["id"],
            "model": MODEL_NAME,
            "prompt": test["prompt"],
            "response": answer
        }

        results.append(result) # Lo añadimos a la lista

        print(f"{test['id']}:")
        print("Prompt:", test["prompt"])
        print("Response:", answer)
        print()

    return results


def save_results(results, filename="baseline_results.json"):
    """Guarda los resultados en un archivo JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nBaseline guardado en {filename}")


if __name__ == "__main__":
    pipe = create_pipeline(MODEL_NAME)
    tests = get_tests()
    print("The tests on the base model will then be carried out...")
    results = run_tests(pipe, tests)
    save_results(results)