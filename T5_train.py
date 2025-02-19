import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup##在这里删除了adamw
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW#根据提示用的
from tqdm import *
from tqdm.auto import tqdm  # 使用auto以确保在不同环境下都能正常显示
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter #tensorboard
import random
#import matplotlib.pyplot as plt

# import pandas as pd
# from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
# from torch.utils.data import Dataset, DataLoader
# import torch
# from tqdm.auto import tqdm  # 使用auto以确保在不同环境下都能正常显示
# from sklearn.model_selection import train_test_split

writer0 = SummaryWriter(log_dir = './output/randeng/output')

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

# 加载数据并分割为训练集和测试集
def load_and_split_data(tokenizer, file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    # df = df.iloc[:20]
    '''
    data = [(row['整合'], row['答案']) for _, row in df.iterrows()]
    data1 = [(row['问题'], row['答案']) for _, row in df.iterrows()]
    train_data, test_data1 = train_test_split(data, test_size=0.2, random_state=42)
    train_data1, test_data = train_test_split(data1, test_size=0.2, random_state=42)
    '''
    data = [(str(row['问题']), str(row['答案'])) for _, row in df.iterrows()]
    # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data = data
    test_data = random.sample(data,len(data)//5)
    print("---------data loaded---------")
    return CustomDataset(tokenizer, train_data), CustomDataset(tokenizer, test_data)

def train_and_evaluate(model, tokenizer, file_path, epochs=50, batch_size=1, lr=5e-5):
    train_dataset, test_dataset = load_and_split_data(tokenizer, file_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)  # 测试时batch_size设为1简化处理
    best_acc = 0
    
    cnt = 0
    num_dict = [str(i) for i in range(100)]

    #train_loss = []
    
    # model.train()
    # optimizer = AdamW(model.parameters(), lr=lr)
    # total_steps = len(train_loader) * epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        all_loss = 0.0
        # 训练部分
        # print(epoch)
        model.train()
        optimizer = AdamW(model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            all_loss += loss.item()

            # progress_bar.set_postfix(loss=loss.item())
            #train_loss.append(loss.item())
        writer0.add_scalar('train_loss', all_loss/len(train_loader), epoch)

        # 评估部分
        model.eval()
        correct_predictions = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
                
                _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = _.loss
                all_loss += loss.item()

                # 假设模型生成的第一个输出就是预测答案
                predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # predicted_text = predicted_text.replace("<extra_id_0>",'')
                # print(predicted_text)
                
                actual_text = tokenizer.decode(batch['labels'][0], skip_special_tokens=True)
                # print(actual_text)

                if predicted_text.strip().lower() == actual_text.strip().lower():
                    correct_predictions += 1

        acc = correct_predictions / len(test_loader)
        print(f"Test Accuracy: {acc:.4f}")
        writer0.add_scalar('test_acc', acc, epoch)
        writer0.add_scalar('test_loss', all_loss/len(test_loader), epoch)
        #return train_loss
        
        # 每次保存
        if best_acc<acc:
            best_acc = acc
            save_path = "./output/randeng" #+ num_dict[cnt]
            cnt += 1
            if cnt == 10:
                model.save_pretrained("./output/randeng20")
                tokenizer.save_pretrained("./output/randeng20")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print("模型和分词器已保存。")

'''if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("./model/flan-t5-base/")
    model = T5ForConditionalGeneration.from_pretrained("./model/flan-t5-base/")

    file_path = "./test.xlsx"
    train_and_evaluate(model, tokenizer, file_path)'''

if __name__ == "__main__":
    load_path = "./output/randeng30"
    # load_path = input('模型地址(模型保存在'./randeng'）'：)
    tokenizer = T5Tokenizer.from_pretrained(load_path)
    model = T5ForConditionalGeneration.from_pretrained(load_path).to('cuda')  # 将模型移动到GPU
    print("---------model loaded---------")
    file_path = "./data/数据全.xlsx"
    train_and_evaluate(model, tokenizer, file_path,20,1,1e-8)
    print("---------train finished---------")
    
    #plt.plot(loss)

    # # 训练和评估完成后保存模型
    # save_path = "./output/randeng"
    # model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)
    # print("模型和分词器已保存。")
    writer0.close()
