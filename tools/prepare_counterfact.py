import json
import random
import re
from pathlib import Path
from datasets import load_dataset

# --- 配置 ---
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "counterfact_50_strict.jsonl"
NUM_SAMPLES = 50
SEED = 42

# 优化版：确保所有 Value 都是名词，完美适配 "What is X's [RELATION]?"
WIKIDATA_RELATIONS = {
    "P17": "Country", 
    "P19": "Place of Birth", 
    "P20": "Place of Death",
    "P27": "Citizenship", 
    "P30": "Continent", 
    "P31": "Category",          # 原 Instance of -> 改为 Category 或 Type
    "P36": "Capital", 
    "P37": "Official Language", 
    "P39": "Position",          # 原 Position Held -> 改为 Position
    "P47": "Neighboring Country", # 原 Shares Border -> 改为 Neighboring Country
    "P101": "Field of Work", 
    "P103": "Native Language",
    "P106": "Occupation", 
    "P108": "Employer", 
    "P127": "Owner",            # [优化] 原 Owned By -> Owner
    "P131": "Location", 
    "P136": "Genre", 
    "P138": "Namesake",         # [优化] 原 Named After -> Namesake
    "P140": "Religion", 
    "P159": "Headquarters", 
    "P176": "Manufacturer",
    "P178": "Developer", 
    "P190": "Sister City", 
    "P264": "Record Label",
    "P276": "Location", 
    "P279": "Parent Class",     # [优化] 原 Subclass of -> Parent Class
    "P361": "Parent Structure", # [优化] 原 Part of -> Parent Structure
    "P364": "Original Language", 
    "P407": "Language", 
    "P413": "Position",         # 原 Sports Position -> Position
    "P449": "Original Network", 
    "P463": "Affiliation",      # [优化] 原 Member of -> Affiliation
    "P495": "Origin Country",
    "P527": "Components",       # [优化] 原 Has Part -> Components
    "P740": "Formation Location", 
    "P937": "Work Location",
    "P1001": "Jurisdiction", 
    "P1303": "Instrument", 
    "P1376": "Capital of",      # 这个比较特殊，暂保留
    "P1412": "Languages Spoken",
    "P137": "Operator",
    "P749": "Parent Organization"
}

def prepare_counterfact():
    print(f"正在加载 COUNTERFACT 数据集...")
    try:
        dataset = load_dataset("NeelNanda/counterfact-tracing", split="train")
    except Exception as e:
        print(f"错误: 请先安装 datasets 库。\n{e}")
        return

    print(f"原始数据量: {len(dataset)}")
    
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    
    selected_data = []
    print(f"正在处理 {NUM_SAMPLES} 条样本...")
    
    for idx in indices:
        item = dataset[idx]
        
        # 提取字段
        subject = str(item.get('subject', '')).strip()
        prompt_tmpl = str(item.get('relation', '')).strip()
        true_val = str(item.get('target_true', '')).strip()
        false_val = str(item.get('target_false', '')).strip()
        rel_id = str(item.get('relation_id', 'unknown'))
        
        case_id = f"cf_{idx}"
        
        # [核心修改] 优先使用映射表中的规范名称
        # 如果找不到 ID，才退回到用 'attribute'
        readable_category = WIKIDATA_RELATIONS.get(rel_id, "attribute")

        entry = {
            "id": case_id,
            "category": rel_id,
            "category_name": readable_category,
            "subject": subject,
            
            # [FIX] 直接把规范的类别名赋给 relation
            # 这样 run_ablation 生成的问题就是 "What is {subject}'s {readable_category}?"
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
            
    print(f"处理完成！数据已保存至: {OUTPUT_FILE}")
    
    # 预览验证
    first = selected_data[1] # 看第二条，通常比第一条有代表性
    print("\n[最终效果预览]")
    print(f"Subject: {first['subject']}")
    print(f"Relation: {first['relation']}")
    print(f"生成的 Question: \"What is {first['subject']}'s {first['relation']}?\"")
    print(f"(这将非常通顺且符合语法)")

if __name__ == "__main__":
    prepare_counterfact()