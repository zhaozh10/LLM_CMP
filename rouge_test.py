import os.path as osp
import pandas as pd
import evaluate
from tqdm import tqdm


def RougeTest(csv_path: str):
    df_test_csv = pd.read_csv(csv_path)
    rouge_e = evaluate.load('rouge')

    rouge_scores = []
    for index, row in tqdm(df_test_csv.iterrows(),total=len(df_test_csv), desc='Processing'):
        label = row['impression']   # GT impression
        prediction = row['pseudo_impression'] # pred impression


        score = rouge_e.compute(predictions=[prediction], references=[label], rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores.append(score)


    # mean
    mean_rouge1 = sum([score['rouge1'] for score in rouge_scores]) / len(rouge_scores)
    mean_rouge2 = sum([score['rouge2'] for score in rouge_scores]) / len(rouge_scores)
    mean_rougeL = sum([score['rougeL'] for score in rouge_scores]) / len(rouge_scores)

    print(f'File {csv_path} R-1: {mean_rouge1:.4f}', f'R-2: {mean_rouge2:.4f}', f'R-L: {mean_rougeL:.4f}')




csv_list=[
    # './results/HuaTuoGPT-7B/MIMIC-EN.csv',
    # './results/HuaTuoGPT-7B/MIMIC-ZH.csv',
    # './results/HuaTuoGPT-7B/Open_I-EN.csv',
    # './results/HuaTuoGPT-7B/Open_I-ZH.csv',
    # './results/luotuo/MIMIC-EN.csv',
    # './results/luotuo/MIMIC-ZH.csv',
    # './results/luotuo/Open_I-EN.csv',
    # './results/luotuo/Open_I-ZH.csv',
    # './results/Ziya-13B/MIMIC-EN.csv',
    # './results/Ziya-13B/MIMIC-ZH.csv',
    # './results/Ziya-13B/Open_I-EN.csv',
    # './results/Ziya-13B/Open_I-ZH.csv',
]
for csv_path in csv_list:
    RougeTest(csv_path)