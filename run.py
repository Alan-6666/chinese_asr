from utils.train import train, test
from utils.load_data_function import get_wavs_and_labels


if __name__=="__main__":
  

#1、获取文件名字和标签
    data_path  = "../data/thchs_30/data_thchs30/data"
    dev_path  = "../data/thchs_30/data_thchs30/dev"
    file_name = "txt_data/thchs_30/test.txt"
    wav_files, labels = get_wavs_and_labels(data_path,dev_path,file_name)    

#2、获取字典


#3、计算标准化特征


#4、训练
    dataset = "thchs_30"    
    dictionary_path = './txt_data/{}/dictionary.txt'.format(dataset) 
    save_address    = "./model_file/{}".format(dataset)
    model_name      = "bi_lstm_150_nl"
    train_data_path = 'txt_data/{}/train.txt'.format(dataset)
    test_data_path  = "txt_data/{}/test.txt".format(dataset)
    stand_para_path = save_address + "/stand_nor.txt"
    epochs    = 150
    with open(dictionary_path) as f:
        temp = f.readlines()
 
    input_size, hidden_size, num_layers, out_size = 80, 512, 1, len(temp)
#    model = RNN(input_size  = input_size ,
#                hidden_size = hidden_size, 
#                num_layers  = num_layers , 
#                out_size    = out_size   ,        
#   )
    model = bi_lstm(
                input_size  = input_size ,
                hidden_size = hidden_size, 
                num_layers  = num_layers , 
                out_size    = out_size   ,        
)
  
#    print(model)
#    torch.cuda.set_device(0)
    if torch.cuda.is_available():
         device = torch.device("cuda")
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
###################### 训练 #################################################
    train(model, save_address, model_name, epochs, train_data_path, test_data_path,dictionary_path, stand_para_path,device) 

###################### 测试 #################################################   
#    model = rnn()
#    model.load_state_dict(torch.load("{}/{}.pkl".format(save_address, model_name))) 
#    model = torch.load("{}/{}.pkl".format(save_address , model_name))
#    test(model, dictionary_path, test_data_path, device)

#5、测试
