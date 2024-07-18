import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from statistics import mode

import re

def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text



class VQAModel(nn.Module):
    def __init__(self, n_answer):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # ResNetの最終FC層を削除して特徴量のみを取得
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(
            nn.Linear(2048 + 768, 512),  # ここを適切な次元に設定
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, n_answer)
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.resnet(image)  # [B, 2048, 1, 1]
        image_features = torch.flatten(image_features, start_dim=1)  # [B, 2048]
        
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        combined_features = torch.cat([image_features, text_features], dim=1)  # [B, 2048+768]
        x = self.fc(combined_features)
        
        return x

def load_model(model_path, n_answer):
    model = VQAModel(n_answer=n_answer)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer

        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self._prepare_dicts()

    def _prepare_dicts(self):
        # 質問と回答の辞書を準備
        for question in self.df["question"]:
            processed_question = process_text(question)
            for word in processed_question.split():
                if word not in self.question2idx:
                    idx = len(self.question2idx)
                    self.question2idx[word] = idx
                    self.idx2question[idx] = word  # 逆引き辞書の更新

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    processed_answer = process_text(answer["answer"])
                    if processed_answer not in self.answer2idx:
                        idx = len(self.answer2idx)
                        self.answer2idx[processed_answer] = idx
                        self.idx2answer[idx] = processed_answer  # 逆引き辞書の更新

    def update_dict(self, dataset):
        # 訓練データの辞書をテストデータに適用
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        # 画像の読み込みと変換
        image_path = f"{self.image_dir}/{self.df['image'][idx]}"
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 質問のトークナイズ
        question = self.df['question'][idx]
        encoded_question = self.tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        input_ids = encoded_question['input_ids'].to(image.device)
        attention_mask = encoded_question['attention_mask'].to(image.device)

        if self.answer:
            # 回答の処理
            answers = [self.answer2idx.get(process_text(answer['answer']), 0) for answer in self.df['answers'][idx]]
            mode_answer_idx = mode(answers)  # 最頻値の回答インデックス
            return image, input_ids.squeeze(0), attention_mask.squeeze(0), torch.tensor(answers, dtype=torch.int64), mode_answer_idx

        return image, input_ids.squeeze(0), attention_mask.squeeze(0)

    def __len__(self):
        return len(self.df)

# 予測用の関数
def create_submission(model, dataloader, device):
    model.to(device)
    model.eval()
    submission = []
    for image, input_ids, attention_mask in dataloader:
        image = image.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        pred = model(image, input_ids, attention_mask)
        predicted_index = pred.argmax(1).cpu().numpy()
        submission.extend(predicted_index)
    
    return submission

# メイン関数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)

    # データの変換設定
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # データセットのロード
    #test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=train_transform)

    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=test_transform, answer=False)
    test_dataset.update_dict(train_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # モデルのロード
    model = load_model("model.pth", n_answer=len(test_dataset.answer2idx))
    
    # 予測の実行
    predictions = create_submission(model, test_loader, device)
    
    # 予測結果の保存
    predictions = np.array([test_dataset.idx2answer[pred] for pred in predictions])
    np.save("submission.npy", predictions)

if __name__ == "__main__":
    main()
