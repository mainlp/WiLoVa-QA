import copy
import io, re, os, json
from collections import Counter, defaultdict

import wikipediaapi
USER_AGENT = "AcademicResearchBot/1.0 (mailto:siyaopeng@cis.lmu.de)"
WIKI_EN = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")
WIKI_BAR = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="bar")
WIKI_YUE = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="zh-yue")
WIKI_DE = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="de")
WIKI_ZH = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="zh")

def parent_categories_of_category(category_title: str,
                                  wiki: wikipediaapi.Wikipedia,
                                  category_header: str = "Kategorie:",
                                  max_depth: int = 3,
                                  visited=None) -> set[str]:
    """
    Return a set of all ancestor categories of a given category up to max_depth.
    category_title should be 'Category:Something' (we'll normalize if not).
    """
    if visited is None:
        visited = set()

    # # normalize title
    if not category_title.startswith(category_header):
        category_title = category_header + category_title

    if max_depth <= 0:
        return set()

    # avoid cycles
    if category_title in visited:
        return set()
    visited.add(category_title)

    cat_page = wiki.page(category_title)
    if not cat_page.exists():
        return set()

    # Direct parents of this category :
    direct_parents = {t for t in cat_page.categories.keys() if t.startswith(category_header)}

    # Recurse upward
    all_ancestors = set(direct_parents)
    for parent in direct_parents:
        all_ancestors |= parent_categories_of_category(parent, wiki, category_header, max_depth - 1, visited)

    return all_ancestors


def page_categories_with_parents(title: str,
                                 wiki: wikipediaapi.Wikipedia = WIKI_EN,
                                category_header: str = "Kategorie:",
                                 max_depth: int = 3) -> dict:
    """
    For a normal article page:
      - 'page_categories': direct categories of the page
      - 'all_parent_categories': all ancestor categories of those categories (up to max_depth)
    """
    p = wiki.page(title)
    if not p.exists():
        return {"page_categories": [], "all_parent_categories": []}

    # Direct categories of the article page:
    page_cats = list(sorted([t for t in p.categories.keys() if t.startswith(category_header)]))

    # Collect ancestors
    all_parents: set[str] = set()
    for c in page_cats:
        all_parents |= parent_categories_of_category(c, wiki, category_header, max_depth=max_depth)

    page_and_parent_cats = list(all_parents.union(set(page_cats)))

    return page_and_parent_cats




# Example code to get page ancester categories
# res = page_categories_with_parents("PyCharm", wiki=WIKI_EN, category_header="Category:", max_depth=3)
# print("Direct page categories:", res["page_categories"])
# print("Ancestor categories (up to 3 levels):", res["all_parent_categories"])


print()

de_bar_file = "../data/de.bar-MW.json"
zh_yue_file = "../data/zh.zh-yue-YM.json"


data_files = [
    # de_bar_file,
    zh_yue_file
]

# get number of annotated wikipedia pages, summary QA pairs, whole page QA pairs
for data_file in data_files:
    with io.open(data_file, "r", encoding="utf8") as f:
        data_dict = json.load(f)
    for title_key in data_dict.keys():
        print("o processing: ", title_key)
        tmp_title_dict = copy.deepcopy(data_dict[title_key])
        for column_key in data_dict[title_key].keys():
            if re.match(r"^Question\d$", column_key) and isinstance(data_dict[title_key][column_key], str):
                tmp_title_dict[column_key+"_qualitychecked"] = data_dict[title_key][column_key]
                tmp_title_dict[column_key + "_QtypeYahoo"] = ""
                tmp_title_dict[column_key + "_QtypeEntity"] = ""
                tmp_title_dict[column_key + "_PageRegion"] = ""
            if re.match(r"^Translation\d$", column_key) and isinstance(data_dict[title_key][column_key], str):
                tmp_title_dict[column_key + "_qualitychecked"] = data_dict[title_key][column_key]

        # get the number of language pages
        if "de.wikipedia.org" in tmp_title_dict["standard_url"]:
            page = WIKI_DE.page(title_key)
        elif "zh.wikipedia.org" in tmp_title_dict["standard_url"]:
            page = WIKI_ZH.page(title_key)
        # ✅ Count how many language versions exist
        num_langs = len(page.langlinks)
        tmp_title_dict["num_langs"] = num_langs

        # get the Wikipedia category ancesters (up to max)
        if "de.wikipedia.org" in tmp_title_dict["standard_url"]:
            standard_res = page_categories_with_parents(title_key, wiki=WIKI_DE, category_header="Kategorie:", max_depth=3)
        elif "zh.wikipedia.org" in tmp_title_dict["standard_url"]:
            standard_res = page_categories_with_parents(title_key, wiki=WIKI_ZH, category_header="Category:", max_depth=3)
        if "bar.wikipedia.org" in tmp_title_dict["dialect_url"]:
            dialect_res = page_categories_with_parents(title_key, wiki=WIKI_BAR, category_header="Kategorie:",
                                                        max_depth=3)
        elif "zh-yue.wikipedia.org" in tmp_title_dict["dialect_url"]:
            dialect_res = page_categories_with_parents(title_key, wiki=WIKI_YUE, category_header="Category:",
                                                       max_depth=3)
            # Kategoariner
        tmp_title_dict["standard_wiki_categories"] = standard_res
        tmp_title_dict["dialect_wiki_categories"] = dialect_res
        print(tmp_title_dict)

        # rebuild dict with desired order
        # tmp_title_dict = {k: tmp_title_dict[k] for k in sorted(tmp_title_dict.keys())}
        data_dict[title_key] = tmp_title_dict

    ordered_data_file = data_file.replace(".json", "_QualityMeta.json")
    with open(ordered_data_file, "w") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

