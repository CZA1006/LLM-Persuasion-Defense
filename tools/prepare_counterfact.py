# Build the report's deterministic COUNTERFACT-Strict sample.
import json
import random
import re
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "counterfact_50_strict.jsonl"
NUM_SAMPLES = 50
SEED = 42

# Readable nominal relations for the strict "What is X's relation?" template.
WIKIDATA_RELATIONS = {
    "P17": "Country", 
    "P19": "Place of Birth", 
    "P20": "Place of Death",
    "P27": "Citizenship", 
    "P30": "Continent", 
    "P31": "Category",
    "P36": "Capital", 
    "P37": "Official Language", 
    "P39": "Position",
    "P47": "Neighboring Country",
    "P101": "Field of Work", 
    "P103": "Native Language",
    "P106": "Occupation", 
    "P108": "Employer", 
    "P127": "Owner",
    "P131": "Location", 
    "P136": "Genre", 
    "P138": "Namesake",
    "P140": "Religion", 
    "P159": "Headquarters", 
    "P176": "Manufacturer",
    "P178": "Developer", 
    "P190": "Sister City", 
    "P264": "Record Label",
    "P276": "Location", 
    "P279": "Parent Class",
    "P361": "Parent Structure",
    "P364": "Original Language", 
    "P407": "Language", 
    "P413": "Position",
    "P449": "Original Network", 
    "P463": "Affiliation",
    "P495": "Origin Country",
    "P527": "Components",
    "P740": "Formation Location", 
    "P937": "Work Location",
    "P1001": "Jurisdiction", 
    "P1303": "Instrument", 
    "P1376": "Capital of",
    "P1412": "Languages Spoken",
    "P137": "Operator",
    "P749": "Parent Organization"
}

def prepare_counterfact() -> None:
    print("Loading the CounterFact source dataset...")
    try:
        dataset = load_dataset("NeelNanda/counterfact-tracing", split="train")
    except Exception as exc:
        raise RuntimeError("Unable to load the dataset; install 'datasets' and check network access.") from exc

    print(f"Source records: {len(dataset)}")
    
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    
    selected_data = []
    print(f"Preparing {NUM_SAMPLES} records...")
    
    for idx in indices:
        item = dataset[idx]
        
        subject = str(item.get('subject', '')).strip()
        prompt_tmpl = str(item.get('relation', '')).strip()
        true_val = str(item.get('target_true', '')).strip()
        false_val = str(item.get('target_false', '')).strip()
        rel_id = str(item.get('relation_id', 'unknown'))
        
        case_id = f"cf_{idx}"
        
        readable_category = WIKIDATA_RELATIONS.get(rel_id, "attribute")

        entry = {
            "id": case_id,
            "category": rel_id,
            "category_name": readable_category,
            "subject": subject,
            "relation": readable_category, 
            
            "o_true": true_val,
            "o_false": false_val,
            "_original_prompt": prompt_tmpl
        }
        selected_data.append(entry)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in selected_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved dataset: {OUTPUT_FILE}")
    
    first = selected_data[0]
    print("\nPreview")
    print(f"Subject: {first['subject']}")
    print(f"Relation: {first['relation']}")
    print(f"Question: \"What is {first['subject']}'s {first['relation']}?\"")

if __name__ == "__main__":
    prepare_counterfact()
