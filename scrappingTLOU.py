import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ======================
# CONFIG
# ======================

WIKI_API = "https://thelastofus.fandom.com/api.php"
OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "llama3:instruct"

CATEGORIES = [

    # Lore
    "Category:Characters",
    "Category:Locations",
    "Category:Groups",
    "Category:Infected types",

    # Gameplay
    "Category:Weapons",
    "Category:Items",
    "Category:Crafting",
    "Category:Supplies",

    # Lore profundo
    "Category:Artifacts",
    "Category:Collectibles",

    # Facciones
    "Category:Members of the Fireflies",
    "Category:Members of the Seraphites",
    "Category:Members of the Washington Liberation Front",

    # Mundo
    "Category:Quarantine zones",
    "Category:Vehicles",

    # Narrativa
    "Category:The Last of Us transcripts",
    "Category:The Last of Us Part II transcripts",
    "Category:The Last of Us Part I transcripts",
]

CATEGORY_TO_TYPE = {
    "Category:Characters": "Character",
    "Category:Locations": "Location",
    "Category:Groups": "Faction",
    "Category:Infected types": "Infected",
    "Category:Weapons": "Weapon",
    "Category:Items": "Item",
    "Category:Crafting": "Item",
    "Category:Supplies": "Item",
    "Category:Artifacts": "Artifact",
    "Category:Collectibles": "Collectible",
    "Category:Members of the Fireflies": "Character",
    "Category:Members of the Seraphites": "Character",
    "Category:Members of the Washington Liberation Front": "Character",
    "Category:Quarantine zones": "Location",
    "Category:Vehicles": "Vehicle",
    "Category:The Last of Us transcripts": "Transcript",
    "Category:The Last of Us Part II transcripts": "Transcript",
    "Category:The Last of Us Part I transcripts": "Transcript",
}

CHUNK_SIZE = 900
MIN_CHUNK_LENGTH = 200

SCRAPE_WORKERS = 8
GEN_WORKERS = 4

GENERATIONS_PER_CHUNK = 2

OUTPUT_FILE = "tlou_dataset_part_four.json"

seen_questions = set()
seen_lock = threading.Lock()  # para multithread seguro

# ======================
# ENTITY TYPE DETECTION
# ======================

def detect_type(title, category=None):
    """Detecta el tipo basado en la categoría si está disponible, sino por palabras clave."""
    if category and category in CATEGORY_TO_TYPE:
        return CATEGORY_TO_TYPE[category]

    t = title.lower()
    if "weapon" in t:
        return "Weapon"
    if "infected" in t:
        return "Infected"
    if "location" in t or "quarantine zone" in t:
        return "Location"
    if "vehicle" in t:
        return "Vehicle"
    if "artifact" in t:
        return "Artifact"
    if "collectible" in t:
        return "Collectible"
    if "item" in t or "crafting" in t or "supply" in t:
        return "Item"
    if "transcript" in t:
        return "Transcript"
    if "character" in t or "member" in t:
        return "Character"
    if "organization" in t or "group" in t or "faction" in t:
        return "Faction"
    return "Entity"

# ======================
# GET CATEGORY PAGES
# ======================

def get_category_pages(category):
    pages = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "format": "json"
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        r = requests.get(WIKI_API, params=params).json()
        members = r.get("query", {}).get("categorymembers", [])
        for m in members:
            title = m.get("title")
            if title and not title.startswith(("File:", "Category:", "Template:", "User:")):
                pages.append(title)
        if "continue" in r:
            cmcontinue = r["continue"]["cmcontinue"]
        else:
            break
    return pages

# ======================
# GET PAGE HTML
# ======================

def get_page_html(title):
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json"
    }
    r_json = requests.get(WIKI_API, params=params).json()
    if "parse" not in r_json:
        return ""
    return r_json["parse"]["text"]["*"]

# ======================
# CLEAN HTML
# ======================

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","sup","table","nav","aside"]):
        tag.decompose()
    text = soup.get_text()
    return " ".join(text.split())

# ======================
# EXTRACT INFOBOX
# ======================

def extract_infobox(html):
    soup = BeautifulSoup(html, "html.parser")
    infobox = soup.find("table", class_="infobox")
    if not infobox:
        return None
    info = {}
    rows = infobox.find_all("tr")
    for row in rows:
        header = row.find("th")
        data = row.find("td")
        if header and data:
            key = header.get_text(strip=True)
            val = data.get_text(" ", strip=True)
            info[key] = val
    return info

# ======================
# CHUNK TEXT
# ======================

def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i:i+CHUNK_SIZE])
        if len(chunk) > MIN_CHUNK_LENGTH:
            chunks.append(chunk)
    return chunks

# ======================
# PROMPT
# ======================

def build_prompt(title, chunk, category=None):
    entity_type = detect_type(title, category)
    return f"""
You are generating high quality training data for an AI about The Last of Us.

Page Title: {title}
Entity Type: {entity_type}

Create diverse examples.

Types required:

1. factual questions
2. reasoning questions
3. character comparison
4. timeline events
5. natural conversations

Rules:

- Use ONLY information from the text
- Do not invent lore
- If information is missing, skip the question
- Avoid repeating questions
- Make answers detailed

Return ONLY JSON.

Format:

{{
"qa":[{{"question":"","answer":""}}],
"reasoning":[{{"question":"","answer":""}}],
"comparison":[{{"question":"","answer":""}}],
"conversation":[{{"user":"","assistant":""}}],
"timeline":[{{"event":"","description":""}}]
}}

Text:
{chunk}
"""

# ======================
# CALL OLLAMA
# ======================

def generate_examples(title, chunk, category=None):
    payload = {
        "model": MODEL,
        "prompt": build_prompt(title, chunk, category),
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.95
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload)
        return json.loads(r.json()["response"])
    except Exception as e:
        print(f"Error generating examples for {title}: {e}")
        return None

# ======================
# DATASET CONVERSION
# ======================

def convert_examples(example, title, category=None):
    data = []
    if not example:
        return data

    entity_type = detect_type(title, category)
    system_msg = f"You are an expert on The Last of Us.\nPage: {title}\nEntity Type: {entity_type}"

    for section in ["qa","reasoning","comparison"]:
        for qa in example.get(section, []):
            q = qa["question"]
            with seen_lock:
                if q not in seen_questions:
                    seen_questions.add(q)
                    data.append({
                        "messages":[
                            {"role":"system","content":system_msg},
                            {"role":"user","content":q},
                            {"role":"assistant","content":qa["answer"]}
                        ]
                    })

    for conv in example.get("conversation", []):
        data.append({
            "messages":[
                {"role":"system","content":system_msg},
                {"role":"user","content":conv["user"]},
                {"role":"assistant","content":conv["assistant"]}
            ]
        })

    for t in example.get("timeline", []):
        q = f"What happened in the event '{t['event']}'?"
        with seen_lock:
            if q not in seen_questions:
                seen_questions.add(q)
                data.append({
                    "messages":[
                        {"role":"system","content":system_msg},
                        {"role":"user","content":q},
                        {"role":"assistant","content":t["description"]}
                    ]
                })

    return data

# ======================
# INFOBOX DATASET
# ======================

def infobox_dataset(info):
    data = []
    name = info.get("Name") or info.get("Character Name") or ""
    if name:
        bio = f"{name} is a character in The Last of Us."
        if "Affiliation" in info:
            bio += f" They are affiliated with {info['Affiliation']}."
        if "Status" in info:
            bio += f" Their status is {info['Status']}."
        data.append({
            "messages":[
                {"role":"system","content":"You are an expert on The Last of Us."},
                {"role":"user","content":f"Who is {name}?"},
                {"role":"assistant","content":bio}
            ]
        })
    return data

# ======================
# PAGE SCRAPER
# ======================

def scrape_page(page):
    try:
        html = get_page_html(page)
        text = clean_html(html)
        chunks = chunk_text(text)
        info = extract_infobox(html)
        return page, chunks, info
    except Exception as e:
        print(f"Error scraping page {page}: {e}")
        return page, [], None

# ======================
# MAIN
# ======================

def main():
    pages = []
    print("Fetching categories...")
    for cat in CATEGORIES:
        pages.extend(get_category_pages(cat))
    pages = list(set(pages))
    print("Total pages:", len(pages))

    dataset = []
    chunks = []

    print("Scraping pages...")
    with ThreadPoolExecutor(SCRAPE_WORKERS) as executor:
        futures = [executor.submit(scrape_page, p) for p in pages]
        for f in tqdm(as_completed(futures), total=len(futures)):
            title, ch, info = f.result()
            for chunk in ch:
                chunks.append((title, chunk))
            if info:
                dataset.extend(infobox_dataset(info))

    print("Total chunks:", len(chunks))
    print("Generating examples with LLM...")

    with ThreadPoolExecutor(GEN_WORKERS) as executor:
        futures = []
        for title, chunk in chunks:
            for _ in range(GENERATIONS_PER_CHUNK):
                futures.append(executor.submit(generate_examples, title, chunk))
        for i, f in enumerate(tqdm(as_completed(futures), total=len(futures))):
            try:
                result = f.result()
                # Encontramos el título correspondiente del chunk
                title, _ = chunks[i % len(chunks)]
                dataset.extend(convert_examples(result, title))
            except Exception as e:
                print(f"Error converting examples: {e}")

    print("Final dataset size:", len(dataset))
    with open(OUTPUT_FILE,"w",encoding="utf-8") as f:
        json.dump(dataset,f,indent=2,ensure_ascii=False)
    print("Dataset saved:", OUTPUT_FILE)

# ======================

if __name__ == "__main__":
    main()