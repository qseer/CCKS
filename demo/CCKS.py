import os
import json

from pip import main
import cv2 as cv
from PIL import Image
import numpy as np
import torch
from torch import dropout, embedding, nn
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms


def read_json(Q_path):
    with open(Q_path,'r') as f:
        for line in f:
            data_json = json.loads(line)
    MC_diagram_path_s = []
    MC_question_s = []
    MC_type_s = []
    MC_answer_s = []
    MC_correct_answer_s = []
    MC_cords_s = []
    
    TF_diagram_path_s = []
    TF_question_s = []
    TF_type_s = []
    TF_answer_s = []
    TF_correct_answer_s = []
    TF_cords_s = []
    
    for i in range(len(data_json)):
        #print('i', i, str(i))
        if data_json[str(i+1)]['question'] != '':
            if data_json[str(i+1)]['type'] == 'MC':
                MC_diagram_path_s.append(data_json[str(i+1)]['diagram_path'].split("\\")[-1])
                MC_question_s.append(data_json[str(i+1)]['question'])
                MC_type_s.append(data_json[str(i+1)]['type'])
                MC_answer_s.append(data_json[str(i+1)]['answer'])
                MC_correct_answer_s.append(data_json[str(i+1)]['correct_answer'])
                MC_cords_s.append(data_json[str(i+1)]['cords'])
            elif data_json[str(i+1)]['type'] == 'TF':
                # TF_diagram_path_s.append(int(data_json[str(i+1)]['diagram_path'].split("\\")[-1].split(".")[0]))
                TF_diagram_path_s.append(data_json[str(i+1)]['diagram_path'].split("\\")[-1])
                TF_question_s.append(data_json[str(i+1)]['question'])

                TF_type_s.append(data_json[str(i+1)]['type'])
                TF_answer_s.append(data_json[str(i+1)]['answer'])
                # TF_correct_answer_s.append('1' if data_json[str(i+1)]['correct_answer']=='a' else '0') 
                # "correct_answer": "a"--> "True"-->'1',"correct_answer": "b"--> "False"-->'0'
                # TF_correct_answer_s.append([1] if data_json[str(i+1)]['correct_answer']=='a' else [0])
                TF_correct_answer_s.append(int(1) if data_json[str(i+1)]['correct_answer']=='a' else int(0))
                TF_cords_s.append(data_json[str(i+1)]['cords'])
            else:
                print("the file Q_json error")
            
    MC = [MC_diagram_path_s, MC_question_s, MC_type_s, MC_answer_s, MC_correct_answer_s, MC_cords_s]
    TF = [TF_diagram_path_s, TF_question_s, TF_type_s, TF_answer_s, TF_correct_answer_s, TF_cords_s]
    return MC, TF
    
    
def read_image(img_data_path):
    #img = cv.imread(img_dta_path, cv.IMREAD_COLOR)
    print('imgpath', img_data_path)
    img = cv.imread(img_data_path)
    print(img.shape)
    img = np.resize(img, img_size) # h*c
    print(img.shape)
    # cv.imshow('img', img)


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
             nn.Conv2d(in_channels=3,
                       out_channels=16,
                       kernel_size=3,
                       stride=2,
                       padding=1),
             nn.BatchNorm2d(16),
             nn.ReLU(),
             )
             
        self.conv2 = nn.Sequential(
             nn.Conv2d(in_channels=16,
                       out_channels=1,
                       kernel_size=3,
                       stride=2,
                       padding=1),
             nn.BatchNorm2d(1),
             nn.ReLU(),
             )
        
        self.Flatten = nn.Flatten()
        
        self.lin1_img = nn.Linear(8192, 2048)
        self.lin2_img = nn.Linear(2048, 1024)
        self.lin3_img = nn.Linear(1024, 512)
        
        # self.lin1_x_Q = nn.Linear(2048, 1024)
        # self.lin2_x_Q = nn.Linear(1024, 512)
        
        self.lin1_x = nn.Linear(1024, 512)
        self.lin2_x = nn.Linear(512, 512)
        # self.lin3_x = nn.Linear(512, 2)

        self.relu = nn.ReLU()
        
        #utils.initialize_weight(self)
        
    def forward(self, x_img, x_Q):
        # img feature
        x_img = self.conv1(x_img) # 16*256*512-->16*128*256
        x_img = self.conv2(x_img) # 16*128*256-->1*64*128-->8192
        x_img = self.Flatten(x_img)
        x_img = self.lin1_img(x_img)
        x_img = self.relu(x_img)
        x_img = self.lin2_img(x_img)
        x_img = self.relu(x_img)
        x_img = self.lin3_img(x_img)
        # x_img = self.relu(x_img)
        # x_img = self.lin3_x(x_img)
        return x_img
    
        '''
        # word feature
        x_Q = self.lin1_x_Q(x_Q)
        x_Q = self.relu(x_Q)
        x_Q = self.lin2_x_Q(x_Q)
        x_Q = self.relu(x_Q)
        
        # fusion
        x = torch.cat((x_img, x_Q), dim=1) 
        x = self.lin1_x(x)
        x = self.relu(x)
        x = self.lin2_x(x)
        x = self.relu(x)
        x = self.lin3_x(x)
        '''


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout):
        super().__init__()
        # TODO: 基础模型待优化
        self.hid_dim = hidden_size
        self.n_layers = 1

        self.embedding = nn.Embedding(input_size, embedding_size)

        self.rnn = nn.GRU(embedding_size, hidden_size, self.n_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        
        # outputs are always from the top hidden layer
        # outputs (the top-layer hidden state for each time-step),
        return hidden.squeeze(0)


class MyData(Dataset):
    def __init__(self, img, text, labels1):
        self.img = img
        self.text = text
        self.labels1 = labels1
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

    def __getitem__(self, idx):
        imag_name = self.img[idx]
        imag_item_path = os.path.join(img_path,imag_name)
        img = cv.imread(imag_item_path)
        img = cv.resize(img, img_size)
        img = self.transform(img)
        label = self.labels1[idx]
        text  = self.text[idx]
        
        return img, text, label # 返回的第item项的图片以及对应的标签

    def __len__(self):
        return len(self.img)


def initialize_weight(m):
    for m in net.img_net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0,0.02)
            m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(0,0.02)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0,0.02)
            m.bias.data.zero_()

class Word2vec():
    def __init__(self, data) -> None:
        super().__init__()
        self.img = data[0]
        self.ori_text = data[1]
        self.label = data[4]
        self.word_list = ['PAD', 'SOS', 'EOS']
        self.word_dict = {'PAD': 0, 'SOS':1, 'EOS':2}
        self.word_count = 3
        self.build_word_list_and_dict()
        self.text, self.text_len = self.word2vec(data[1])
        
    
    def build_word_list_and_dict(self):
        for line in self.ori_text:
            word_list = line.split(' ')
            for word in word_list:
                if word not in self.word_list:
                    self.word_dict[word] = self.word_count
                    self.word_list.append(word)
                    self.word_count += 1
    
    def word2vec(self, sentence):
        text = []
        text_len = []
        for sen in sentence:
            word_list = sen.split(' ')
            word_list = ['SOS'] + word_list + ['EOS']
            word2idx = []
            for word in word_list:
                word2idx.append(self.word_dict[word])
            text.append(word2idx)
            text_len.append(len(word2idx))
        return text, text_len


class Model(nn.Module):
    def __init__(self, dataloader):
        super().__init__()
        self.img_net = Net().to(device)
        self.text_net = Encoder(dataloader.word_count, embedding_size, hidden_size, dropout).to(device)
        self.img_optimizer = torch.optim.SGD(self.img_net.parameters(), lr=lr)
        self.text_optimizer = torch.optim.SGD(self.text_net.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.out1 = nn.Linear(1024, 512)
        self.out2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def train(self, data):

        self.img_net.train()
        self.text_net.train()

        for epoch in range(epochs):
            count, total = 0, 0
            for batch_size in range(int(len(data.img)/batch_sizes)): # (998, 998/8=124.75, +1=125,using all data)

                img1 = data.img[batch_size:batch_sizes+batch_size]

                text_not_pad  = data.text[batch_size:batch_sizes+batch_size]
                text_len = data.text_len[batch_size:batch_sizes+batch_size]
                max_len = max(text_len)
                text = []
                for sen in text_not_pad:
                    sen = sen + (max_len - len(sen)) * [0]
                    text.append(sen)
                text = torch.tensor(text)
                
                labels1  = data.label[batch_size:batch_sizes+batch_size]

                all_data1 = MyData(img1, text, labels1)
                dataloader_try = DataLoader(all_data1, batch_size=batch_sizes, shuffle=True)
                for batch_idx, (img, text, target) in enumerate(dataloader_try):
                    #print(batch_idx,len(img))
                    #print(img.data.size())
                    
                    # 读入数据
                    img = img.to(device)
                    text = text.to(device)
                    target = torch.tensor(target)
                    target = target.to(device)
                    
                    # 计算模型预测结果和损失
                    output1 = self.img_net(img, img) # dim is the word2vec
                    output2 = self.text_net(text.transpose(0, 1))
                    output = torch.cat((output1, output2), 1)
                    pre = self.out1(output)
                    pre = self.relu(pre)
                    pre = self.out2(pre)

                    ls = self.loss(pre, target)
                    
                    self.img_optimizer.zero_grad()
                    self.text_optimizer.zero_grad()
                    ls.mean().backward()
                    self.img_optimizer.step()
                    self.text_optimizer.step()
                
                    for i in range(batch_sizes):
                        if torch.argmax(pre[i]) == labels1[i]:
                            count += 1
                        total += 1 
                # print("Epoch:", epoch+1, ' loss:', ls.mean().item())
            print("Epoch:", epoch+1, ' loss:', ls.mean().item(), "AccNum:", count, " TotNum:", total, " Acc:", count/total)
            # break # train the first batchsize


# TF_data = [TF_diagram_path_s, TF_question_s, TF_type_s, TF_answer_s, TF_correct_answer_s, TF_cords_s]
#loss = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
img_size = (256, 512)
batch_sizes = 64
epochs = 10
lr = 1e-2
hidden_size = 512
embedding_size = 128
dropout = 0.5
Q_path = './data/train/Q.json'
model_path = './TF.params'
img_path = './data/train/Diagrams/'

# read data and shuffle
MC_data,  TF_data= read_json(Q_path)
#print('TF_data:', len(TF_data), len(TF_data[0]), len(TF_data[5]),len(TF_data[5]))
#print('before shuffle TF_data:', TF_data[0][0],TF_data[1][0], TF_data[2][0])


for i in range(len(MC_data)):
    if len(MC_data[i]) == len(MC_data[0]):
        np.random.seed(len(MC_data[0]))
        np.random.shuffle(MC_data[i])
    else:
        print('this MC_data:', i, 'length is error')
        
for i in range(len(TF_data)):
    if len(TF_data[i]) == len(TF_data[0]):
        np.random.seed(len(TF_data[0]))
        np.random.shuffle(TF_data[i])
    else:
        print('this TF_data:', i, 'length is error')

# TODO: 目前只用了判断题，后面要加上选择题
dataloader = Word2vec(TF_data)
# transf = transforms.ToTensor()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Model(dataloader).to(device)

# 模型参数加载到新模型
# if os.path.exists(model_path):
#     state_dict=torch.load(model_path)
#     net.load_state_dict(state_dict)
#     print('加载...')
# else:
#     net.apply(initialize_weight)
#     print('params...')

#print(net)
#print(net.state_dict())

net.train(dataloader) 
torch.save(net.state_dict(), 'TF.params') #只保存加载模型参数
# TODO: 测试没有写

'''
import torch
import torch.nn as nn
import math

entroy=nn.CrossEntropyLoss()
input=torch.Tensor([[-0.7715, -0.6205,-0.2562], [-0.7715, -0.6205,-0.2562]])
target = torch.tensor([0,1])
print('target',target.shape)
output = entroy(input, target)
print(output)
'''