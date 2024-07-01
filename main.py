import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
from torch.utils.data import random_split
import csv
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import os
import gc
from transformers import BertTokenizer, BertModel


def generate_unique_filename(extension=".pth"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}{extension}"


def read_params(file_path):
    params = {}
    with open(file_path, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header row
        for rows in reader:
            key = rows[0]
            value = rows[1]
            # 数値のキャスト処理を追加
            if key in ["seed", "batch_size", "num_epoch"]:
                params[key] = int(value)
            elif key in ["learning_rate", "weight_decay"]:
                params[key] = float(value)
            else:
                params[key] = value
    return params

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        question = self.df["question"][idx]
        # question_encoding = self.tokenizer(question, return_tensors='pt', padding=True, truncation=True)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, question, torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, question
    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

def ResNet50_trained():
    # 学習済みモデルの読み込み
    # Resnet50を重み付きで読み込む
    model_ft = models.resnet50(pretrained = True)
    model_ft.fc = nn.Linear(model_ft.fc.in_features, 512)

    return model_ft


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet50_trained()

        # BERTモデルのエンコーダーを追加
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # BERTのパラメータをフリーズ
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.bert_model(question).last_hidden_state[:, 0, :]  # テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        print(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0
    # loss_list = []
    # vqa_acc_list = []
    # simple_acc_list = []
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    start = time.time()
    for batch_idx, (image, question, answers, mode_answer) in enumerate(dataloader):
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_vqa_acc = VQA_criterion(pred.argmax(1), answers)
        batch_simple_acc = (pred.argmax(1) == mode_answer).float().mean().item()

        total_loss += batch_loss
        total_acc += batch_vqa_acc
        simple_acc += batch_simple_acc

        # # 損失と精度のリストに追加
        # loss_list.append(batch_loss)
        # vqa_acc_list.append(batch_vqa_acc)
        # simple_acc_list.append(batch_simple_acc)

        # # グラフの更新
        # clear_output(wait=True)
        # ax.clear()
        # ax.plot(loss_list, label='Loss')
        # ax.plot(vqa_acc_list, label='VQA Accuracy')
        # ax.plot(simple_acc_list, label='Simple Accuracy')
        # ax.set_xlabel('Batch')
        # ax.set_ylabel('Value')
        # ax.legend()
        # plt.show()


        # total_loss += loss.item()
        # total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        # simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
        # Print batch loss and accuracy
        print(f'Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}, '
              f'VQA Accuracy: { VQA_criterion(pred.argmax(1), answers):.4f}, Simple Accuracy: { (pred.argmax(1) == mode_answer).float().mean().item():.4f}')

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def create_directory_if_not_exists(directory_path):
    # ディレクトリが存在しない場合に作成する
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' has been created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def main():
    # パラメータの読み込み
    params = read_params('config.csv')

    # deviceの設定
    set_seed(params['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(60),  # ランダムに最大10度回転
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # カラーのランダム調整
        transforms.ToTensor()  # テンソルに変換
    ])

    train_datasets = VQADataset(df_path=params['train_df_path'], image_dir=params['train_image_dir'], transform=transform)
    test_dataset = VQADataset(df_path=params['valid_df_path'], image_dir=params['valid_image_dir'], transform=transform, answer=False)
    test_dataset.update_dict(train_datasets)
    print(len(train_datasets), len(test_dataset))


    total_size = len(train_datasets)
    val_size = int(total_size * 0.10)  # For example, 20% for validation
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(train_datasets, [train_size, val_size])



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(vocab_size=len(train_datasets.question2idx) + 1, n_answer=len(train_datasets.answer2idx)).to(device)

    # optimizer / criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # train model
    print("start training")
    fig_dir = os.path.join("fig", generate_unique_filename(extension=""))
    create_directory_if_not_exists(fig_dir)

    # 訓練データを保存するリスト
    train_losses = []
    train_accuracies = []
    train_simple_accuracies = []
    val_losses = []
    val_accuracies = []
    val_simple_accuracies = []
    for epoch in range(params['num_epoch']):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print("finish train")
        val_loss, val_acc, val_simple_acc,val_time = eval(model, val_loader, optimizer, criterion, device)
        # データをリストに保存
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_simple_accuracies.append(train_simple_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_simple_accuracies.append(val_simple_acc)
        print(f"【{epoch + 1}/{params['num_epoch']}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
        print(f"【{epoch + 1}/{params['num_epoch']}】\n"
              f"val time: {val_time:.2f} [s]\n"
              f"val loss: {val_loss:.4f}\n"
              f"val acc: {val_acc:.4f}\n"
              f"val simple acc: {val_simple_acc:.4f}")
        # グラフの更新
        clear_output(wait=True)
        epochs = range(1, epoch + 2)
        
        plt.figure(figsize=(12, 6))
        
        # 損失のプロット
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b', label='Training loss')
        plt.plot(epochs, val_losses, 'r', label='val loss')
        plt.title('loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 精度のプロット
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, 'r', label='Training accuracy')
        plt.plot(epochs, train_simple_accuracies, 'b', label='Training simple accuracy')
        plt.plot(epochs, val_accuracies, 'g', label='val accuracy')
        plt.plot(epochs, val_simple_accuracies, 'y', label='val simple accuracy')
        plt.title('accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        display(plt.gcf())
        plot_path = os.path.join(fig_dir, f'epoch_{epoch + 1}.png')
        plt.savefig(plot_path)
        plt.close()
        time.sleep(0.1)  # 更新の間隔を調整するために少し待つ

    model_save_file = generate_unique_filename()
    p = os.path.join("./result/model", model_save_file)
    torch.save(model.state_dict(), p)


    # 提出用ファイルの作成
    model.eval()
    submission = []
    print("start submission")
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_datasets.idx2answer[id] for id in submission]
    submission = np.array(submission)

    # タイムスタンプを利用して固有のファイル名を生成
    submission_save_path = generate_unique_filename(extension=".npy")
    p = os.path.join("./result/submission", submission_save_path)
    np.save(p, submission)

    # パラメータとファイルパスを一つの辞書にまとめる
    result_dict = params.copy()
    result_dict['model_save_path'] = model_save_file
    result_dict['submission_save_path'] = submission_save_path

    # CSVファイルに書き込む
    csv_file_path = 'training_results.csv'
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as csvfile:
        fieldnames = list(result_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(result_dict)

if __name__ == "__main__":
    main()