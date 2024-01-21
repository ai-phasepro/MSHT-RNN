import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
from torch.utils.data import dataset, dataloader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from torch import LongTensor, Tensor, from_numpy, max_pool1d, nn, unsqueeze,optim
import argparse
from torch.nn.utils import weight_norm
#from torchnlp.encoders.texts import StaticTokenizerEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import copy
# from torchsummaryX import summary

def readdata(root_dir, pos_protein_dir, neg_protein_dir, data_balance, seed):
    pos_protein_path = os.path.join(root_dir, pos_protein_dir)
    neg_protein_path = os.path.join(root_dir, neg_protein_dir)
    
    pos_data = pd.read_csv(pos_protein_path).iloc[:,0].to_numpy()
    neg_data = pd.read_csv(neg_protein_path).iloc[:,0].to_numpy()

    print("pos_size:", pos_data.size)
    print("neg_size:", neg_data.size)

    np.random.seed(seed)  
    np.random.shuffle(pos_data)  
    np.random.shuffle(neg_data)

    pos_size = pos_data.size
    neg_size = neg_data.size

    if(data_balance==True):
        print("数据正负样本均衡化")
        if(pos_size > neg_size):
            pos_data = pos_data[:neg_size]
        else:
            neg_data = neg_data[:pos_size]
        print("均衡化后pos_size:", pos_data.size)
        print("均衡化后neg_size:", neg_data.size)

    pos_label = np.ones(shape=(pos_data.size,))
    neg_label = np.zeros(shape=(neg_data.size,))


    pos_sequence = pos_data.tolist()
    pos_sequence = list(map(lambda x: x.lower(), pos_sequence))
    neg_sequence = neg_data.tolist()
    neg_sequence = list(map(lambda x: x.lower(), neg_sequence))

    return np.asarray(pos_sequence, dtype='O'), pos_label, np.asarray(neg_sequence, dtype='O'), neg_label
    
def word2Num(train, test, min=0, max=None, max_features=None):
    vocab = {}
    count = {}
    for sequence in train:
        sequence = sequence.replace(' ', '')
        for word in sequence:
            count[word] = count.get(word, 0) + 1
    
    for sequence in test:
        sequence = sequence.replace(' ', '')
        for word in sequence:
            count[word] = count.get(word, 0) + 1

    if min is not None:
        count = {word:value for word,value in count.items() if value>min}
    if max is not None:
        count = {word:value for word,value in count.items() if value<max}
    if  max_features is not None:
        temp = sorted(count.items(), key=lambda x:x[-1], reverse=True)[:max_features]
        count = dict(temp)

    for word in count:
        vocab[word] = len(vocab) + 1
    print(vocab)
    Num = []
    for sequence in train:
        sequence = sequence.replace(' ', '')
        num = []
        for word in sequence:
            num.append(vocab.get(word))
        Num.append(num)
    print(len(Num))
    Num2 = []
    for sequence in test:
        sequence = sequence.replace(' ', '')
        num2 = []
        for word in sequence:
            num2.append(vocab.get(word))
        Num2.append(num2)
    print(len(Num2))  
    return Num, Num2, vocab        
 

def head_padding(num, max_length):
    pad_index = 0
    head_padding_list = []
    for protein in num:
        protein_pad = protein.copy()
        if (len(protein) < max_length):
            padding = [pad_index for i in range(max_length - len(protein))]
            protein_pad.extend(padding)
            head_padding_list.append(protein_pad)
        else:
            protein_head = protein_pad[:max_length]
            head_padding_list.append(protein_head)
    return head_padding_list


def collate_fn(data): 
    max_length1, max_length2, max_length3 = 1000, 2000, 3000   
    data_num, data_label_ten = [], []
    for tuple in data:
        data_label_ten.append(tuple[1])
        data_num.append(tuple[0].tolist())

    head1 = head_padding(data_num, max_length1)
    head2 = head_padding(data_num, max_length2)
    head3 = head_padding(data_num, max_length3)
    head1 = torch.LongTensor(head1)
    head2 = torch.LongTensor(head2)
    head3 = torch.LongTensor(head3)

    data_label_ten = torch.LongTensor(data_label_ten)

    return (head1, head2, head3), data_label_ten   
    

class Mydata(dataset.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __getitem__(self, idx):
        protein = self.data[idx]
        label = self.label[idx]
        return protein, label
    def __len__(self):
        assert len(self.data)==len(self.label)
        return len(self.data)


class FConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(FConv, self).__init__()
        self.conv_layer=nn.Conv2d(in_channels, out_channels, 1, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.conv_layer(x)
        x = nn.ReLU()(x)
        # x = self.drop(x)
        x = torch.flatten(x, 1)
        return x


class Net_multi(nn.Module):
    def __init__(self, multi, vocab_size, embedding_num, hidden_dim, num_layers, biFlag, dropout=0.2):  
        super(Net_multi, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_num = embedding_num
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if biFlag:
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biFlag = biFlag
        self.device = torch.device("cuda")
        self.embedding = nn.Embedding(vocab_size, embedding_num, padding_idx=0)  
        self.lstm = nn.LSTM(input_size= embedding_num, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=biFlag)
        #self.globalaveragepool = nn.AdaptiveAvgPool1d(1)
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear((multi*self.bi_num*self.hidden_dim), 256),  #128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256,32),   #128,32
            nn.ReLU(),
            nn.Linear(32,2)
        )
        
    def forward(self, input1, input2, input3):
        output1 = self.embedding(input1)
        output1, (ht,ct) = self.lstm(output1)
        output1 = output1.permute(0, 2, 1)
        output1 = self.globalmaxpool(output1).squeeze()
        
        output2 = self.embedding(input2)
        output2, (ht,ct) = self.lstm(output2)
        output2 = output2.permute(0, 2, 1)
        output2 = self.globalmaxpool(output2).squeeze()
        
        output3 = self.embedding(input3)
        output3, (ht,ct) = self.lstm(output3)
        output3 = output3.permute(0, 2, 1)
        output3 = self.globalmaxpool(output3).squeeze()
        
        
        output = torch.cat([output1, output2, output3], dim=1)
        
        output = self.linear(output)
        return output


def set_seed(seed):
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True


if __name__== '__main__':
    device = torch.device("cuda")
    seed = 1
    set_seed(seed)
    root_dir = './processed_dataset'

    pos_protein_dir = 'nuclear.csv'   # 大小8421
    neg_protein_dir = 'nonnuclear.csv' # 大小18278

    pos_seed_list = [1]

    for time in range(len(pos_seed_list)):
        pos_seed = pos_seed_list[time]
        pos_sequence,pos_label,neg_sequence,neg_label  = readdata(root_dir, pos_protein_dir, neg_protein_dir, data_balance=True, seed=pos_seed)
        
        save_csv_path = './output_dir/Bilstm_MS_output/Bilstm_MS3_Head_em512_512_1_mydata_{}.csv'.format((time+1)) 

        df_test = pd.DataFrame(columns=['Fold','step','val_loss','val_correct','val_F1','val_sen','val_precision', 'val_spe','val_roc','test_loss','test_correct','test_F1','test_sen','test_precision', 'test_spe','test_roc'])
        df_test.to_csv(save_csv_path, index=False)    

        pos_num = pos_sequence.size
        neg_num = neg_sequence.size
        print('pos_num=',pos_num) 
        print('neg_num=',neg_num) 


        X = np.concatenate((pos_sequence, neg_sequence), axis=0)
        Y = np.concatenate((pos_label, neg_label), axis=0)
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, Y)):       # train:val:test=7:1:2 /6:2:2

            train_val_seq, train_val_y = X[train_val_idx].tolist(), Y[train_val_idx]
            test_seq, test_y = X[test_idx].tolist(), Y[test_idx]


            print ('-------第{}fold...-------'.format(fold+1))
            train_val_num, test_num, vocab  = word2Num(train_val_seq, test_seq)
            train_data_size = len(train_val_num)
            test_data_size = len(test_num)

            train_num, val_num, train_y, val_y = train_test_split(train_val_num, train_val_y, train_size=int(X.size * 0.7), random_state=1, shuffle=True, stratify=train_val_y)
        
                
            train_ten, val_ten,test_ten = [], [], []
            for sequence in train_num:
                train_ten.append(torch.LongTensor(sequence))
            for sequence in val_num:
                val_ten.append(torch.LongTensor(sequence))
            for sequence in test_num:
                test_ten.append(torch.LongTensor(sequence))
            
            
            # train_val_y = np.hstack((train_y, val_y))
            train_label_ten = from_numpy(train_y)
            val_label_ten = from_numpy(val_y)
            test_label_ten = from_numpy(test_y)
            train_label_ten = train_label_ten.type(torch.LongTensor)
            val_label_ten = val_label_ten.type(torch.LongTensor)
            test_label_ten = test_label_ten.type(torch.LongTensor)
            
            model = Net_multi(3, len(vocab)+1, 512, 512, 1, True)  # 512,256,1, hidden128表示256,128,1, em512:512,100,1 
            model = model.to(device)
            print(model)
    
            loss_fn = nn.CrossEntropyLoss()       
            loss_fn = loss_fn.to(device)
            
            learning_rate = 1e-4  # 之前1e-4
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.99))
            scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=20, verbose=True)
            
            train = Mydata(train_ten, train_label_ten)
            val = Mydata(val_ten, val_label_ten)
            test = Mydata(test_ten, test_label_ten)
            

            set_seed(seed)
            train_dataloader = dataloader.DataLoader(dataset=train, batch_size=32,shuffle=True, collate_fn=collate_fn)
            val_dataloader = dataloader.DataLoader(dataset=val, batch_size=32,shuffle=True, collate_fn=collate_fn)
            test_dataloader = dataloader.DataLoader(dataset=test, batch_size=32,shuffle=True, collate_fn=collate_fn)
                
                # 记录训练的次数
            total_train_step = 0
                # 记录测试的次数
            total_test_step = 0
                # 训练的轮数
            epochs = 81    
            
                
            for epoch in range(1, epochs):
                print("-------第 {} 轮训练开始-------".format(epoch))
                    
                model.train()
                total_labels = 0
                train_loss = 0.0
                y_true = []
                y_pre = []
                y_score = []
                for input, label in train_dataloader:
                    head1, head2, head3 = input
                    head1 = head1.to(device)
                    head2 = head2.to(device)
                    head3 = head3.to(device)
                    label = label.to(device)

                    logits1 = model(head1, head2, head3) 
                    loss = loss_fn(logits1, label)     
                        
                        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                        
                    train_loss += loss.item() * label.size(0)
                    _, predicted = torch.max(logits1, 1)
                    y_pre.extend(predicted.cpu())
                    y_true.extend(label.cpu())
                    y_score.extend(torch.softmax(logits1, dim=-1)[:,1].cpu().detach())
                    total_labels += label.size(0)
                        
                        
                    total_train_step = total_train_step + 1
                        
                    
                train_loss /= total_labels
                train_correct = metrics.accuracy_score(y_true, y_pre)
                train_F1 = metrics.f1_score(y_true, y_pre, average='macro')
                train_R = metrics.recall_score(y_true, y_pre)
                train_precision = metrics.precision_score(y_true, y_pre)
                train_auc = metrics.roc_auc_score(y_true, y_score)
                save_content = 'Train: Correct: %.5f, Precision: %.5f, R: %.5f, F1(macro): %.5f, AUC:%.5f, train_loss: %f' % \
                                (train_correct, train_precision, train_R, train_F1, train_auc, train_loss)
                print(save_content)
                    
                
                model.eval()
                max_val_correct = 0  # 最高验证集准确率
                save_model_list = []  # 保存验证集最好模型
                save_model_path_list = []  # 验证集最好模型的名称
                min_epoch = 10  # 训练至少需要的轮数
                val_loss = 0
                y_true_val = []
                y_pre_val = []
                y_score_val = []
                total_labels_val = 0
                with torch.no_grad():
                    for input, label in val_dataloader:
                        head1, head2, head3 = input
                        head1 = head1.to(device)
                        head2 = head2.to(device)
                        head3 = head3.to(device)
                        label = label.to(device)

                        logits1 = model(head1, head2, head3)  
                        loss = loss_fn(logits1, label)     
                            
                        val_loss += loss.item() * label.size(0)
                        _, predicted = torch.max(logits1,1)
                        y_pre_val.extend(predicted.cpu())
                        y_true_val.extend(label.cpu())
                        y_score_val.extend(torch.softmax(logits1, dim=-1)[:, 1].cpu().detach())
                        total_labels_val += label.size(0)
                    
                    val_loss /= total_labels_val
                    val_correct = metrics.accuracy_score(y_true_val, y_pre_val, normalize=True)
                    val_F1 = metrics.f1_score(y_true_val, y_pre_val, average='macro')
                    val_R = metrics.recall_score(y_true_val, y_pre_val)
                    val_precision = metrics.precision_score(y_true_val, y_pre_val)
                    val_auc = metrics.roc_auc_score(y_true_val, y_score_val)

                    save_content = 'val: Correct: %.5f, Precision: %.5f, R: %.5f, F1(macro): %.5f, AUC: %.5f, val_loss: %f' % \
                            (val_correct, val_precision, val_R, val_F1, val_auc, val_loss)
                    print(save_content)    
                                  
                    
                    p = np.array(y_pre_val)[np.array(y_true_val) == 1]
                    tp = p[p == 1]
                    n = np.array(y_pre_val)[np.array(y_true_val) == 0]
                    tn = n[n == 0]
                    
                    
                    list0 = [fold+1, epoch, val_loss, val_correct, val_F1, val_R, val_precision, (tn.shape[0] / n.shape[0] if n.shape[0] > 0 else 1), val_auc]
                    
                    
                    if epoch > 40:  
                        scheduler.step(val_correct)   
            
                model.eval()
                test_loss = 0
                y_true_test = []
                y_pre_test = []
                y_score_test = []
                total_labels_test = 0
                with torch.no_grad():
                    for input, label in test_dataloader:
                        head1, head2, head3 = input
                        head1 = head1.to(device)
                        head2 = head2.to(device)
                        head3 = head3.to(device)
                        label = label.to(device)

                        logits1 = model(head1, head2, head3)  
                        loss = loss_fn(logits1, label)    
                            
                        test_loss += loss.item() * label.size(0)
                        _, predicted = torch.max(logits1,1)
                        y_pre_test.extend(predicted.cpu())
                        y_true_test.extend(label.cpu())
                        y_score_test.extend(torch.softmax(logits1, dim=-1)[:, 1].cpu().detach())
                        total_labels_test += label.size(0)
                    
                    
                    test_loss /= total_labels_test
                    test_correct = metrics.accuracy_score(y_true_test, y_pre_test)
                    test_F1 = metrics.f1_score(y_true_test, y_pre_test, average='macro')
                    test_R = metrics.recall_score(y_true_test, y_pre_test)
                    test_precision = metrics.precision_score(y_true_test, y_pre_test)
                    test_auc = metrics.roc_auc_score(y_true_test, y_score_test)

                    save_content = 'Test: Correct: %.5f, Precision: %.5f, R: %.5f, F1(macro): %.5f, AUC:%.5f, test_loss: %f' % \
                            (test_correct, test_precision, test_R, test_F1, test_auc, test_loss)
                    print(save_content)
                    
                   
                    p = np.array(y_pre_test)[np.array(y_true_test) == 1]
                    tp = p[p == 1]
                    n = np.array(y_pre_test)[np.array(y_true_test) == 0]
                    tn = n[n == 0]
                    
                    list1 = [test_loss, test_correct, test_F1, test_R, test_precision, (tn.shape[0] / n.shape[0] if n.shape[0] > 0 else 1), test_auc]
                    list0.extend(list1)
                    data_test = pd.DataFrame([list0])
                    data_test.to_csv(save_csv_path, mode='a', header=False, index=False, float_format='%.4f')

    
        
        
        
        
               
                
                      
                
                
            
            
            
            
            


            