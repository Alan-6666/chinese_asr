import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import os
from model.model import *
from load_data_function import ASR_Dataset, ASR_DataLoader,stand_para, read_stand_para
from ctc_decode import remove_blank, greedy_decode, beam_decode, compute_wer


def train(model,save_address,model_name, epochs, data_path,test_path , dictionary_path,stand_para_path,device):
     
    print(model)
    model.to(device)
    model.train()

#    final_mean, final_stdevs = stand_para(data_path)
    final_mean, final_stdevs = read_stand_para(stand_para_path)   
    #读取训练数据
    train_dataset = ASR_Dataset(data_path, dictionary_path, final_mean, final_stdevs) #输出，特征地址以及文字标签
    batch_size = 64
    train_dataloader = ASR_DataLoader(train_dataset,batch_size=batch_size, num_workers=8,shuffle=True)
    train_data_number = len(train_dataloader.dataset)

    #读取测试数据

    test_dataset = ASR_Dataset(test_path, dictionary_path, final_mean, final_stdevs) #输出，特征地址以及文字标签
    test_dataloader = ASR_DataLoader(test_dataset,batch_size=batch_size, num_workers=8,shuffle=True)
    test_data_number = len(test_dataloader.dataset)

#    print("data_number",data_number)
    optimizer = torch.optim.Adam(
        params = model.parameters(),
        lr           = 0.001,
        weight_decay = 0
    )
    #读取字典    
    with open(dictionary_path) as f:
        dictionary = f.readlines()
    dictionary = [x.strip() for x in dictionary]
    #损失函数
    criterion = nn.CTCLoss(blank=0, reduction="mean")
    h_state  = None 
    
    #存储列表
    train_recodes = []
    test_recodes  = [] 
    #训练   
    for i in range(1, epochs+1):
        epoch_loss = []
        loss_sum = 0
        model.train()   
#        print("epochs:{}".format(i))
        for j,(features, labels, labels_length)  in enumerate(train_dataloader):
            prediction = model(features.to(device))    
#            print("pre_shape",prediction.shape ) 
            n,t,c= prediction.shape
            #(输入序列长度，batch_size,字符集总长度)
            log_probs = prediction[0]
            for k in range(1,n):
                log_probs = torch.cat((log_probs, prediction[k]),1)
           # log_probs = log_probs.reshape(t,n,c).log_softmax(2).requires_grad_().to(device)
            log_probs = log_probs.reshape(t,n,c).log_softmax(2)
        #    print("log_probs", log_probs)
            targets_temp = []  
            for label in labels:
                targets_temp.extend(label)
            targets = torch.tensor(targets_temp).to(device) 
#            print("targets",targets)
            #N
            input_lengths  = torch.tensor([features.shape[1]]*features.shape[0]).to(device)  
#            print("input_length",input_lengths)    
            #N
            target_lengths = torch.tensor(labels_length).to(device)
#            print("target_lengths", target_lengths)

            loss = criterion(log_probs.cpu(), targets, input_lengths, target_lengths)
            loss_sum +=loss.item()
            optimizer.zero_grad()
              
            loss.backward()
            optimizer.step()  
           # h_state = h_state.detach()  # 这一行很重要，将每一次输出的中间状态传递下去(不带梯度
            if (j+1) % 50==0:
           # print("prediction.shape", prediction.shape)
                real_words = greedy_decode(labels, dictionary,blank=0,decode_model="real")
                pred_words = greedy_decode(prediction.cpu().detach().numpy(), dictionary,blank=0, decode_model="prediction")
                wer = compute_wer(real_words, pred_words, dictionary)
                recode = "epoch:{}, train_iteration:{}, loss:{}, wer:{}".format(i,j+1,loss,wer)
                train_recodes.append(recode)
                print(recode)    
         
        train_loss = "train_loss_sum:{}".format(loss_sum/train_data_number) 
        train_recodes.append(train_loss)
        print(train_loss) 
    #测试
        model.eval()
        loss_sum == 0
        with torch.no_grad():
            for j,(features, labels, labels_length)  in enumerate(test_dataloader):
                prediction  = model(features.to(device))    
                n,t,c= prediction.shape
            #(输入序列长度，batch_size,字符集总长度)
                log_probs = prediction[0]
                for i in range(1,n):
                    log_probs = torch.cat((log_probs, prediction[i]),1)
                log_probs = log_probs.reshape(t,n,c).log_softmax(2).requires_grad_().to(device)
                targets_temp = []  
                for label in labels:
                   targets_temp.extend(label)
                targets = torch.tensor(targets_temp).to(device) 
                input_lengths  = torch.tensor([features.shape[1]]*features.shape[0]).to(device)      
                #N
                target_lengths = torch.tensor(labels_length).to(device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss_sum +=loss.item()
                if (j+1) % 50==0:
           # print("prediction.shape", prediction.shape)
                    real_words = greedy_decode(labels, dictionary,blank=0,decode_model="real")
                    pred_words = greedy_decode(prediction.cpu().detach().numpy(), dictionary,blank=0, decode_model="prediction")
                    wer = compute_wer(real_words, pred_words, dictionary)
           
                    recode = "epoch:{}, test_iteration:{}, loss:{}, wer:{}".format(i,j+1,loss,wer)
                    test_recodes.append(recode)
                    print(recode)     
            test_loss = "test_loss_sum:{}".format(loss_sum/test_data_number) 
            test_recodes.append(test_loss)
            print(test_loss)

    with open("model_file/thchs_30/{}_train_recodes.txt".format(model_name), "w") as f:
        for i in range(len(train_recodes)):
            f.write(train_recodes[i]+"\n") 
    with open("model_file/thchs_30/{}_test_recodes.txt".format(model_name), "w") as f:
        for i in range(len(test_recodes)):
            f.write(test_recodes[i]+"\n") 
    #保存
    torch.save(model.state_dict(),"{}/{}.pkl".format(save_address, model_name))

def test(model, dictionary_path, test_path,device):
    model.to(device)
    model.eval()
    h_state = None
    with open(dictionary_path) as f:
        dictionary = f.readlines()
    dictionary = [x.strip() for x in dictionary]
    with open(test_path) as f:
        real_labels = f.readlines()
#    print("dictionary", dictionary)
 #   print("real_labels",real_labels)   
    test_dataset = ASR_Dataset(test_path,dictionary_path) #输出，特征地址以及文字标签
    test_dataloader = ASR_DataLoader(test_dataset,batch_size= 4, num_workers=8, shuffle=True)
    wer_sum = 0   
    with torch.no_grad():
        for j,(features, labels, labels_length)  in enumerate(test_dataloader):
            prediction  = model(features.to(device))
            print(prediction.shape)
            real_words = greedy_decode(labels, dictionary,blank=0,decode_model="real") #贪婪搜索
            pred_words = beam_decode(prediction.cpu().detach().numpy(), dictionary,blank=0, decode_model="prediction",beam_size=5)

            wer = compute_wer(real_words, pred_words, dictionary)
            wer_sum += wer
            print("wer", wer)
             
    
