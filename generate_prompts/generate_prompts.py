import pandas as pd
from tqdm import tqdm
import pickle
import sys
import json

VALID_PROMPT_COMPONENTS = {
    "question",
    "link_standard",
    "link_dialect",
    "summary_standard",
    "summary_dialect",
    "summary_d2s_translated",
    "summary_original",
    "summary_translated",
}

try:
    with open("yue2cmn_GoogleCloudTranslate.pkl", "rb") as f:
        yue2cmn_MT = pickle.load(f)
except FileNotFoundError:
    yue2cmn_MT = {}

def load_from_excel(path: str, lang: str) -> list[dict]:
    df = pd.read_excel(path, sheet_name="v2")
    df = df[df["Q0: Should we annotate this dialect-standard Wikipedia page pair? If no, specify why."] == "Yes"]
    records = []
    for idx, row in df.iterrows():
        for qcol in ["Question1", "Question2", "Question3", "Question4"]:
            question_text = row.get(qcol)
            if pd.isna(question_text) or not str(question_text).strip():
                continue
            record = {
                "lang": lang,
                "row_index": idx,
                "question_column": qcol,
                "question_text": question_text,
                "ground_truth_answer": row.get(qcol.replace("Question", "Translation")),
                "standard_url": row.get("standard_url", ""),
                "dialect_url": row.get("dialect_url", ""),
                "standard_title": row.get("standard_title", ""),
                "dialect_title": row.get("dialect_title", ""),
                "summary_standard": row.get("standard_summary", ""),
                "summary_dialect": row.get("dialect_summary", ""),
                "summary_d2s_translated": yue2cmn_MT.get(row.get("dialect_summary", ""), ""),
            }
            records.append(record)
    return records

def load_from_jsonl(path: str, lang: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            summ_o = d.get("content")
            row_index = d.get("q_id", -1)
            if d.get("original_lang") in ("zh", "de"):
                continue
            if lang == "zho":
                q, a = d.get("zh_q"), d.get("zh_a")
                summ_t = d.get("zh_c")
            elif lang == "deu":
                q, a = d.get("de_q"), d.get("de_a")
                summ_t = d.get("de_c")
            records.append({
                "lang": lang,
                "row_index": row_index,
                "question_text": q,
                "ground_truth_answer": a,
                "summary_original": summ_o or "",
                "summary_translated": summ_t or "",
            })
    return records

def generate_prompt(lang, row, prompt_parts: list):
    for part in prompt_parts:
        if part not in VALID_PROMPT_COMPONENTS:
            raise ValueError(f"Invalid prompt part: {part}")
    sections = []
    if lang == 'zho':
        if "question" in prompt_parts:
            sections.append(f"请回答以下问题：\n{row.get('question_text','')}")
        if "link_standard" in prompt_parts and row.get("standard_url"):
            sections.append(f"你可以参考关于{row.get('standard_title','')}的维基百科链接：{row['standard_url']}")
        if "link_dialect" in prompt_parts and row.get("dialect_url"):
            sections.append(f"你可以参考关于{row.get('dialect_title','')}的维基百科链接：{row['dialect_url']}")
        for key in ["summary_standard", "summary_dialect", "summary_d2s_translated", "summary_original", "summary_translated"]:
            if key in prompt_parts and row.get(key):
                sections.append(f"这里有一些相关的信息：\n{row[key]}")
        sections.append("请用中文回答，并且将你的最终答案放在 <Answer> 和 </Answer> 标签之间。\n在标签内只保留最终答案内容，不要包含任何多余解释或其他文字。")
    elif lang == 'deu':
        if "question" in prompt_parts:
            sections.append(f"Bitte beantworten Sie diese Frage:\n{row.get('question_text','')}")
        if "link_standard" in prompt_parts and row.get("standard_url"):
            sections.append(f"Sie können die folgende Wikipedia-Seite über {row.get('standard_title','')} konsultieren: {row['standard_url']}")
        if "link_dialect" in prompt_parts and row.get("dialect_url"):
            sections.append(f"Sie können die folgende Wikipedia-Seite über {row.get('dialect_title','')} konsultieren: {row['dialect_url']}")
        for key in ["summary_standard", "summary_dialect", "summary_d2s_translated", "summary_original", "summary_translated"]:
            if key in prompt_parts and row.get(key):
                sections.append(f"Hier sind einige relevante Informationen:\n{row[key]}")
        sections.append("Bitte schließen Sie Ihre endgültige Antwort in <Answer>...</Answer>-Tags ein. Innerhalb der Tags soll nur der endgültige Antwortinhalt stehen, ohne zusätzliche Erklärungen.\nBitte stellen Sie sicher, dass Sie auf Deutsch antworten.")
    return "\n".join(sections)

def get_all_prompts(records: list[dict], lang: str, settings: list[list]) -> dict:
    all_setting_prompts = {}
    for setting in settings:
        key = "+".join(setting)
        prompts = []
        for row in tqdm(records):
            prompt_text = generate_prompt(lang, row, setting)
            prompts.append({
                'row_index': row.get("row_index", -1),
                'question_column': row.get("question_column", ""),
                "question": row.get("question_text", ""),
                "prompt": prompt_text,
                "ground_truth_answer": row.get("ground_truth_answer", ""),
            })
        all_setting_prompts[key] = prompts
    return all_setting_prompts

if __name__ == "__main__":
    lang = sys.argv[1]  # can be 'zho', 'deu', or 'zho_deu'
    source_type = sys.argv[2]  # 'dialectqa' or 'eclektic'
    if source_type == 'eclektic':
        input_path = 'eclektic_main.jsonl'
    elif lang == 'deu':
        input_path = 'de.bar-template-full.xlsx'
    elif lang == 'zho':
        input_path = 'zh.zh-yue-1000-template.xlsx'

    langs = lang.split('_')
    settings = [["question"],
                ["question","link_standard"],
                ["question","link_dialect"],
                ["question","link_standard", "link_dialect"],
                ["question","summary_standard"],
                ["question","summary_dialect"],["question","summary_d2s_translated"],
                ["question","summary_standard", "summary_dialect"]
                ]

    # settings = [["question"],
    #             ["question","summary_original"],
    #             ["question","summary_translated"],
    #             ]
    
    all_lang_prompts = {}
    for l in langs:
        if source_type == 'dialectqa':
            records = load_from_excel(input_path, l)
        elif source_type == 'eclektic':
            records = load_from_jsonl(input_path, l)
        else:
            raise ValueError("source_type must be 'dialectqa' or 'eclektic'")
        all_lang_prompts[l] = get_all_prompts(records, l, settings)

    out_path = f"all_setting_prompts_{lang}_{source_type}.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(all_lang_prompts, f)
    print(f"Prompts generated → {out_path}")
