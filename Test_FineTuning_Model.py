from transformers import pipeline, AutoTokenizer
import json

MODEL_NAME = "adriangg04/TheLastOfUs-QA"


def load_tokenizer(model_name):
    """Carga el tokenizer del modelo."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def create_pipeline(model_name, tokenizer):
    """Crea el pipeline de generación."""
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="auto"
    )
    return pipe


def get_tests():
    """Define la batería de pruebas."""
    # Bateria de pruebas
    tests = [
        {"id": "test_001","prompt": "What does Ellie find out when she returns to her house on The Last of Us?"},
        {"id": "test_002","prompt": "Why does Tommy go to Santa Barbara ahead of Ellie in The Last of Us?"},
        {"id": "test_003","prompt": "What happened in the event 'Ellie goes to Seattle' in The Last of Us?"},
        {"id": "test_004","prompt": "Why does Joel kill Marlene in The Last of Us?"},
        {"id": "test_005","prompt": "How does the rat king differ from other infected in The Last of Us?"},
        {"id": "test_006","prompt": "How do stalkers compare to runners in terms of strength in The Last of Us?"},
        {"id": "test_007","prompt": "What is the main reason for Ellie's journey to Seattle in The Last of Us?"},
        {"id": "test_008","prompt": "What happens to Sarah in Texas on The Last of Us?"},
        {"id": "test_009","prompt": "What happens to Sam after he is scratched by an infected in The Last of Us?"},
        {"id": "test_010","prompt": "What happened to Ellie and Joel after they left the University on The Last of Us?"},
        {"id": "test_011","prompt": "What is the relationship between Jerry and Abby in The Last of Us?"},
        {"id": "test_012","prompt": "How did Abby react to Owen's death in The Last of Us?"},
        {"id": "test_013","prompt": "What motivated Abby to spare Ellie's life in The Last of Us?"},
        {"id": "test_014","prompt": "Why do Clickers develop a form of echolocation in The Last of Us?"},
        {"id": "test_015","prompt": "What are the four categories of collectibles in The Last of Us?"}
    ]
    return tests


def build_prompt(tokenizer, test):
    """Construye el prompt en formato chat."""
    # Montamos el mensaje con los roles correspondientes (system y user)
    messages = [
        {"role": "system", "content": "You are an expert on The Last of Us"},
        {"role": "user", "content": test["prompt"]}
    ]

    # Convierte el mensaje en tipo chat para que el modelo lo entienda (en el otro código no se hacia porque el pipeline lo hacia solo)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


def generate_response(pipe, prompt):
    """Genera la respuesta del modelo."""
    # Le pasamos el prompt para que responda
    response = pipe(
        prompt,
        max_new_tokens=200,
        temperature=0.7
    )

    # Devolvemos la respuesta del modelo ajustado
    answer = response[0]["generated_text"]
    return answer


def run_tests(pipe, tokenizer, tests):
    """Ejecuta todos los tests."""
    results = [] # Lista con los resultados obtenidos

    for test in tests: # Recorremos las distintas preguntas
        prompt = build_prompt(tokenizer, test)
        answer = generate_response(pipe, prompt)

        # Lo guardamos en un diccionario
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


def save_results(results, filename="baseline_results_merge_model.json"):
    """Guarda los resultados en JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Baseline guardado en {filename}")


if __name__ == "__main__":
    tokenizer = load_tokenizer(MODEL_NAME)
    pipe = create_pipeline(MODEL_NAME, tokenizer)
    tests = get_tests()
    print("Next, the adapted model will be tested...")
    results = run_tests(pipe, tokenizer, tests)
    save_results(results)