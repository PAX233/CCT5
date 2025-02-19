from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json


# 解析实体
def post_processing(outputs, text, labels_map):
    _, predicted_labels = torch.max(outputs.logits, dim=2)

    predicted_labels = predicted_labels.detach().cpu().numpy()

    predicted_tags = [labels_map[label_id] for label_id in predicted_labels[0]]

    result = {}
    entity = ""
    type = ""
    for index, word_token in enumerate(text):
        tag = predicted_tags[index]
        if tag.startswith("B-"):
            type = tag.split("-")[1]
            if entity:
                if type not in result:
                    result[type] = []
                result[type].append(entity)
            entity = word_token
        elif tag.startswith("I-"):
            type = tag.split("-")[1]
            if entity:
                entity += word_token
        else:
            if entity:
                if type not in result:
                    result[type] = []
                result[type].append(entity)
            entity = ""
    return result

def main():
    labels_path = "./data/NER/new/labels.json"
    model_name = './output/NER/'
    max_length = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载label
    labels_map = {}
    with open(labels_path, "r", encoding="utf-8") as r:
        labels = json.loads(r.read())
        for label in labels:
            label_id = labels[label]
            labels_map[label_id] = label

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels_map))
    model.to(device)

    
    while True:
        text = input("请输入：")
        if not text or text == '':
            continue
        if text == 'q':
            break

        encoded_input = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
        input_ids = torch.tensor([encoded_input['input_ids']]).to(device)
        attention_mask = torch.tensor([encoded_input['attention_mask']]).to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        result = post_processing(outputs, text, labels_map)
        #print(text)
        print(result)

if __name__ == '__main__':
    main()

