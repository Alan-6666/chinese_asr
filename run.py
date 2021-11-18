from utils.train import train, test
from utils.load_data_function import get_wavs_and_labels


if __name__=="__main__":
  
#1、获取文件名字和标签
    #训练集
    data_path  = "data/data_thchs30/data"
    train_path = "data/data_thchs30/train"
    file_name  = "data_file/thchs_30/train.txt"
    wav_files, labels = get_wavs_and_labels(data_path,train_path,file_name)    
    #测试集
    data_path  = "data/data_thchs30/data"
    test_path  = "data/data_thchs30/test"
    file_name  = "data_file/thchs_30/test.txt"
    wav_files, labels = get_wavs_and_labels(data_path,test_path,file_name)   
    
#2、获取字典
    dictionary_path = "data_file/thchs_30/dictionary.txt"
    
    with open(dictionary_path) as f:
        dictionary = f.readlines()    
    
#3、计算标准化特征
    stand_para_path = "data_file/thchs_30/stand_nor.txt"

#4、训练
    dataset = "thchs_30"    
    save_address    = "model_file/{}".format(dataset)
    model_name      = "bi_lstm_150_nl"
    train_data_path = save_address + "/train.txt'
    test_data_path  = save_address + "/test.txt'
    epochs    = 150
    #定义参数
    input_size, hidden_size, num_layers, out_size = 80, 512, 1, len(temp)
    model = bi_lstm(
                input_size  = input_size ,
                hidden_size = hidden_size, 
                num_layers  = num_layers , 
                out_size    = out_size   ,        
)
    if torch.cuda.is_available():
         device = torch.device("cuda")
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
###################### 训练 #################################################
    train(model, save_address, model_name, epochs, train_data_path, test_data_path,dictionary, stand_para_path,device) 

###################### 测试 #################################################   
#    model = rnn()
#    model.load_state_dict(torch.load("{}/{}.pkl".format(save_address, model_name))) 
#    model = torch.load("{}/{}.pkl".format(save_address , model_name))
#    test(model, dictionary_path, test_data_path, device)

#5、测试
