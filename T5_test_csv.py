import os
import pandas as pd
from tqdm import *
from tqdm.auto import tqdm
from fuzzywuzzy import fuzz
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
import sys
import csv
path = './output/randeng30/'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(path) 
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
model.to(device)

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt_text, completion_text = self.data[idx]
        input_encoding = self.tokenizer(prompt_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        target_encoding = self.tokenizer(completion_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        return {
            'input_ids': input_encoding.input_ids.squeeze(),
            'attention_mask': input_encoding.attention_mask.squeeze(),
            'labels': target_encoding.input_ids.squeeze()
        }

def postprocess(text):
    return text.replace(".", "").replace('</>','')

def answer_fn(text, top_k=50):
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=512,temperature=0.5,do_sample=True,repetition_penalty=1.4 ,top_k=top_k,top_p=0.95)
    result = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    # print(type(out["sequences"]))
    return postprocess(result[0]) 

def test_data(file_path):
    turns = int(input('请输入轮数: '))
    df = pd.read_excel(file_path, engine='openpyxl')
    if turns == -1 : turns = len(df)
    df = df.sample(turns)
    data = [(str(row['问题']), str(row['答案'])) for _, row in df.iterrows()]
    print("---------data loaded---------")
    with open('test_data_output.csv','w') as w:
        writer = csv.DictWriter(w, fieldnames = ['序号', '问题', '答案', '输出','正确'])
        writer.writeheader()
        test_dataset = CustomDataset(tokenizer,data)
        test_loader = DataLoader(test_dataset, batch_size=1)
        model.eval()
        correct_predictions = 0
        cnt = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)

                # 假设模型生成的第一个输出就是预测答案
                result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
                # predicted_text = predicted_text.replace("<extra_id_0>",'')
                # print(predicted_text)
                
                a = tokenizer.decode(batch['labels'][0], skip_special_tokens=True).strip().lower()
                # print(actual_text)
                q = data[cnt][0]
                
                # print(a)
                # print(result)
                # sys.exit(0)
                
                # w.write(f"问题: {q}\n")
                # w.write(f"答案: {a}\n")
                # w.write(f"生成: {result}\n")
                # w.write("-"*100)
                # w.write('\n')
                
                
                similarity_ratio = fuzz.ratio(result, a)
                flag = 0
                if similarity_ratio>=80:
                    correct_predictions += 1
                    flag = 1
                else:
                    pass
                    # print(f"问题: {q}\n")
                    # print(f"答案: {a}\n")
                    # print(f"生成: {result}\n")
                    # w.write(f"{cnt+2}\n")
                if flag == 1:
                    flag = "是"
                else :
                    flag = "否"
                row = {'序号': cnt, '问题': q, '答案': a, '输出': result, '正确': flag}
                writer.writerow(row)
                
                cnt+=1
                print(correct_predictions / cnt)
                if cnt == turns:
                    break
        acc = correct_predictions / cnt
        # w.write(f"Test Accuracy: {(acc):.4f}")
        print(f"Test Accuracy: {(acc):.4f}")
        print("----ended----")
        # for q, a in tqdm(data, desc=f"Testing"):
        #     w.write(f"问题: {q}\n")
        #     # a = actual_text = tokenizer.decode(a, skip_special_tokens=True)
        #     w.write(f"答案: {a}\n")
        #     result=answer_fn(q, top_k=50)
        #     w.write(f"生成: {result}\n")
        #     w.write("*"*100)
        #     w.write('\n')
        #     if result.strip().lower() == a.strip().lower():
        #         acc += 1
        # w.write(f"Test Accuracy: {(acc / turns):.4f}")
def eval_data(file_path,turns = -1):
    df = pd.read_excel(file_path, engine='openpyxl')
    if turns == -1:
        turns = int(input(f'请输入测试数: {len(df)}: '))
        if turns > len(df):
            turns = len(df)
        print(f"测试数目：{turns}")
        print('*'*100)
    df = df.sample(turns)
    data = [(str(row['问题']), str(row['答案'])) for _, row in df.iterrows()]
    # data = data.iloc[:100]
    acc = 0
    for q, a in tqdm(data, desc=f"Evaluating ") :
        result=answer_fn(q, top_k=50)
        if result.strip().lower() == a.strip().lower():
            acc += 1
            
    print(f"Test Accuracy: {acc / len(data):.4f}")
    return acc



# test_path = './data/webQA/me_train.xlsx'
test_path = './data/数据全.xlsx'

while True:
    text = input('请输入问题:')
    # test_path = input('输入测试数据地址:')
    if text == 'q':
        break
    elif text == '2':
        test_data(test_path)
        break
    elif text == '3':
        eval_data(test_path)
        break
    elif text == '4':
        eval_data(test_path,turns=100)
        break
    result=answer_fn(text, top_k=50)
    print("模型生成:",result)
    print('*'*100)