import jieba
from rouge import Rouge
from sacrebleu.metrics.chrf import CHRF
from bert_score import score as bertscore
import pickle
import sys
from tqdm import tqdm

# Rouge-L scorer
def rouge_l(candidate, reference, language):
    rouge = Rouge()
    if language == 'zho':
        # jieba tokenization, handling mixture of chinese and latin-script words
        cand_tokens = " ".join(list(jieba.cut(candidate))[:500]) # truncate to first 500 tokens to avoid maximum recursion error for rouge
        ref_tokens = " ".join(list(jieba.cut(reference)))
        scores = rouge.get_scores(cand_tokens, ref_tokens)
        #print(cand_tokens)
    else:
        scores = rouge.get_scores(candidate[:5000], reference) # truncate to first 5000 characters to avoid maximum recursion error for rouge
    #Note: "f" stands for f1_score, "p" stands for precision, "r" stands for recall.
    # rouge_l_f1 = scores[0]['rouge-l']['f']# F1 score
    # Ensure scores is a list of dicts, not a float
    if isinstance(scores, list) and len(scores) > 0 and 'rouge-l' in scores[0]:
        rouge_l_f1 = scores[0]['rouge-l']['f'] # F1 score
    else:
        rouge_l_f1 = 0.0
    return rouge_l_f1

## evaluating when prompt language is chinese or german
def print_evaluation(all_setting_results:dict[str,list[dict]],subset:str,language:str):
    evaluation_results = []
    # Instantiate the chrF scorer (use word_order=2 for chrF++)
    chrf_scorer = CHRF(word_order=2) #pass 2 for chrF++
    if language == 'zho':
        EMPTY_TOKEN = '[无]'
        MODEL_TYPE = "bert-base-chinese"
        LANG = "zh"
        #question4s = get_question4s_from_excel('zh.zh-yue-1000-template.xlsx')
    elif language == 'deu':
        EMPTY_TOKEN = '[Null]'
        MODEL_TYPE = "bert-base-multilingual-cased"
        LANG = "de"
        #question4s = get_question4s_from_excel('de.bar-template-full.xlsx')
    evaluation_results.append(f"{language}({subset}):\trouge-l(F1)\tchrf++\tBERTScore(F1)\tAcc.(LLMjudge)")
    for setting_key, one_setting_results in tqdm(all_setting_results.items()):
        #print(setting_key)
        rouge_l_score_acc = 0
        chrf_score_acc = 0
        yes_s = 0
        no_s = 0
        count = 0
        # Collect all answers/references for batch BERTScore
        cand_list, ref_list = [], []
        if subset == 'Q1-3':
            subset_results = [result for result in one_setting_results if result['question_column'] != 'Question4']
        elif subset == 'Q4':
            subset_results = [result for result in one_setting_results if result['question_column'] == 'Question4']
        else:
            subset_results = one_setting_results
        for result in subset_results:
            #ground_truth_answer = GoogleTranslator(source='zh-CN', target='en').translate(result['ground_truth_answer'])
            ground_truth_answer = str(result['ground_truth_answer'])
            if result['answer'] == None or result['answer'] in ['', '...']:
                answer = EMPTY_TOKEN
            else:
                answer = str(result['answer'])
            # llm as judge evals
            if result['eval'] == 'NO':
                no_s += 1
            elif result['eval'] == 'YES':
                yes_s += 1
            # rouge_l
            #rouge_l_score = rouge_l(answer, ground_truth_answer)
            rouge_l_score = rouge_l(answer, ground_truth_answer, language)
            rouge_l_score_acc += rouge_l_score

            # chrF score (sentence-level)
            # sacrebleu expects list of references, even for one reference
            chrf_score = chrf_scorer.sentence_score(answer, [ground_truth_answer]).score
            chrf_score_acc += chrf_score
            # print(answer)
            # print(ground_truth_answer)
            # print(rouge_l_score)
            # For BERTScore
            cand_list.append(answer)
            ref_list.append(ground_truth_answer)
            count += 1
        P, R, F = bertscore(cand_list, ref_list,
                            model_type = MODEL_TYPE,
                            device="cpu",
                            lang = LANG,                             # <-- REQUIRED when rescale_with_baseline=True
                            rescale_with_baseline=True,
                            batch_size=16,
                            idf=False)
        bertscore_score = round(float(F.mean().item())*100, 2)
        evaluation_results.append(f"{setting_key.replace('question+','+')}\t{round(rouge_l_score_acc*100/count,2)}\t{round(chrf_score_acc/count,2)}\t{bertscore_score}\t{round(100*yes_s/(yes_s+no_s),2)}")
    print("\n".join(evaluation_results))

if __name__ == '__main__':
    results_path = sys.argv[1]
    # load the pickle
    with open(results_path, "rb") as f:
        all_setting_results_all_lang = pickle.load(f)
    if 'eclektic' in results_path:
        print('evaluating eclektic results')
        for lang, all_setting_results in all_setting_results_all_lang.items():
            print(lang)
            print_evaluation(all_setting_results,'all',lang)
    else:
        print('evaluating dialectQA results')
        for lang, all_setting_results in all_setting_results_all_lang.items():
            # for new-version results with language key
            if lang in ['zho','deu']:
                print('new version results with language code as the top-level key')
                print(lang)
                print_evaluation(all_setting_results,'Q1-3',lang)
                print_evaluation(all_setting_results,'Q4',lang)
            else:# for old-version results without language key
                print('(old version results without language code as the top-level key)')
                if 'zho' in results_path:
                    lang = 'zho'
                elif 'deu' in results_path:
                    lang = 'deu'
                print_evaluation(all_setting_results_all_lang,'Q1-3',lang)
                print_evaluation(all_setting_results_all_lang,'Q4',lang)