# encoding:utf-8
import torch
import torch.nn as nn
from torch.nn import TripletMarginLoss #BCEWithLogitsLoss CosineEmbeddingLoss
from transformers import BertTokenizer, BertModel

import numpy as np
import pandas as pd
from tqdm import tqdm



class TTModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.model_bert = BertModel.from_pretrained('bert_pretrain')
        self.model_bert = BertModel.from_pretrained("roberta/")

    def forward(self, text_q, text_pa, text_na):
        out_q = self.model_bert(text_q)[1]
        out_pa = self.model_bert(text_pa)[1]
        out_na = self.model_bert(text_na)[1]
        return out_q, out_pa, out_na

    def answer_vec(self, text_a):
        out_a = self.model_bert(text_a)[1]
        return out_a

    def query_vec(self, text_q):
        out_q = self.model_bert(text_q)[1]
        return out_q


def eu_n(a,b):
    dis=0
    for i in range(len(a)):
        dis+=(a[i]-b[i])**2
    distance=np.sqrt(dis)
    return distance


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, news_q, news_pa, news_na, tokenizer):
        self.news_q = news_q
        self.news_pa = news_pa
        self.news_na = news_na
        self.tokenizer = tokenizer
        self.max_len = 200
        self.set_size = len(news_pa)
        return

    def __len__(self):
        return len(self.news_q)

    def __getitem__(self, idx):
        tensor = {}
        tensor['news_q'] = self.tokenizer(self.news_q[idx],
                                          padding="max_length",
                                          truncation=True,
                                          max_length=self.max_len,
                                          return_tensors='pt')['input_ids'][0]
        tensor['news_pa'] = self.tokenizer(self.news_pa[idx],
                                          padding="max_length",
                                          truncation=True,
                                          max_length=self.max_len,
                                          return_tensors='pt')['input_ids'][0]
        tensor['news_na'] = self.tokenizer(self.news_na[idx],
                                          padding="max_length",
                                          truncation=True,
                                          max_length=self.max_len,
                                          return_tensors='pt')['input_ids'][0]
        return tensor


def load_data_from_file(file_path, tokenizer, batchsize):
    data = pd.read_csv(file_path, sep="\t")
    max_len = max(map(len, data['news_q']))
    news_q = list(data["news_q"])
    news_pa = list(data["news_pa"])
    news_na = list(data["news_pa"])
    print(len(news_q))
    t_dataset = NewsDataset(news_q, news_pa, news_na, tokenizer)
    t_loader = torch.utils.data.DataLoader(t_dataset, batchsize, shuffle=True, drop_last=True)
    return t_loader


def fit(epoch, device, model, loss_function, optimizer, train_loader, test_loader, batch_size):
    model.train()

    train_loss =0.0
    step = 0
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader) ):
    #for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader) // batch_size):
        optimizer.zero_grad()
        news_q = batch['news_q'].to(device)
        news_pa = batch['news_pa'].to(device)
        news_na = batch['news_na'].to(device)
        out_q, out_pa, out_na = model(news_q, news_pa, news_na)
        loss = loss_function(out_q, out_pa, out_na)  # 计算Loss
        train_loss += loss.cpu().detach().numpy()
        step += 1
        avg_loss = train_loss / (step*batch_size)
        print(f'Epoch: {epoch}, Step: {step}, Loss: {avg_loss}')
        loss.backward()
        optimizer.step()

    test_loss =0.0
    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader) // batch_size):
        news_q = batch['news_q'].to(device)
        news_pa = batch['news_pa'].to(device)
        news_na = batch['news_na'].to(device)
        out_q, out_pa, out_na = model(news_q, news_pa, news_na)
        loss = loss_function(out_q, out_pa, out_na)
        test_loss += loss.cpu().detach().numpy()
    print(f'Test Loss: {test_loss}')

    return train_loss, test_loss


def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained("roberta/")
    #model = BertModel.from_pretrained("roberta/")
    model = TTModel()
    model.to(device)
    model.load_state_dict(torch.load("roberta_model_weight01.pth"))
    print("load model")

    batchsize = 4
    train_datafile_path = "data/news for roberta train.csv"
    train_loader = load_data_from_file(train_datafile_path, tokenizer, batchsize)
    test_datafile_path = "data/news for roberta test.csv"
    test_loader = load_data_from_file(train_datafile_path, tokenizer, batchsize)
    print("load data")

    loss_function = TripletMarginLoss(margin=5) # loss_function = nn.SmoothL1Loss()
    learningrate = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), learningrate)
    print("load loss function and optimizer")

    train_loss = []
    test_loss = []

    print("start train")
    epochs = 4
    for epoch in range(epochs):
        epoch_loss, epoch_test_loss = fit(epoch, device, model, loss_function, optimizer,train_loader, test_loader, batchsize)
        train_loss.append(epoch_loss)
        test_loss.append(epoch_test_loss)
    print("Finished Training")
    torch.save(model.state_dict(), "roberta_model_weight01.pth")
    print("model saved")
    print(train_loss)
    print(test_loss)
    return


if __name__ == '__main__':
    print("start")

    train()

    print("over")

