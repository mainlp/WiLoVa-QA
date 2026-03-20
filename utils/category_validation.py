import copy
import io, re, os, json
from collections import Counter, defaultdict

de_bar_file = "../data/de.bar-MW_QualityMeta-VB.json"
zh_yue_file = "../data/zh.zh-yue-YM_QualityMeta-LP.json"

region_categories = {
"Guangdong":"local",
"Guangxi":"local",
"Hongkong":"local",
"Macau":"local",
"Bavaria":"local",
"Austria":"local",
"South Tyrol":"local",
"core-area":"local",
"China":"standard",
"Taiwan":"standard",
"Germany":"standard",
"Italy":"standard",
"Switzerland":"standard",
"majority-area":"standard",
"Other": "other",
"General": "other",
}

topic_classses = [
"history",
"sport",
"geography",
"entertainment-art",
"animals-plants",
"politics-government",
"food",
"transportation",
"science-math-technology",
"linguistics",
"culture-customs",
"education",
"business",
"other",
]

data_files = [
    de_bar_file,
    zh_yue_file
]

# get number of annotated wikipedia pages, summary QA pairs, whole page QA pairs
for data_file in data_files:
    corpus_topics = []
    corpus_regions = []
    with io.open(data_file, "r", encoding="utf8") as f:
        data_dict = json.load(f)
    for title_key in data_dict.keys():
        if "Article_Topic" in list(data_dict[title_key].keys()):
            if data_dict[title_key]["Article_Topic"] not in topic_classses:
                print(title_key, data_dict[title_key]["Article_Topic"])
            else:
                corpus_topics.append(data_dict[title_key]["Article_Topic"])
        if "Article_Region" in data_dict[title_key].keys():
            if data_dict[title_key]["Article_Region"] not in list(region_categories.keys()):
                print(title_key, data_dict[title_key]["Article_Region"])
            else:
                corpus_regions.append(data_dict[title_key]["Article_Region"])
                summary_length_ratio = 1.0 * data_dict[title_key]['dialect_len_sum'] / data_dict[title_key]['standard_len_sum']
                text_length_ratio = 1.0 * data_dict[title_key]['dialect_len_text'] / data_dict[title_key][
                    'standard_len_text']
                if summary_length_ratio > 2 and text_length_ratio > 1 and region_categories[data_dict[title_key]["Article_Region"]]=='local':
                    print(title_key, summary_length_ratio, text_length_ratio)

    print(
        data_file,
        Counter(corpus_topics),
        Counter(corpus_regions),
        Counter(Counter([region_categories[x] for x in corpus_regions])),
        sep="\n"
    )

print("o Done!")



