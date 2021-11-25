from utils.train_test import train, test
from utils.load_data_function import get_wavs_and_labels, get_dictionary, read_stand_para
from utils.init_function import initialization 
from model.model import *
import argparse



if __name__=="__main__":
  
    parser = argparse.ArgumentParser(description='train or test')
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--decode", required=True)
    args = parser.parse_args()
    
    dictionary_path = "data_file/thchs_30/dictionary.txt"
    dictionary ,dict_length= get_dictionary(dictionary_path)

    train_form, decode_function = initialization(args).run()
    
    if train_form =="train":
        print("训练模式")
        #设置解码模式
        #1、获取文件名字和标签
        #训练集
        data_root = "../../data/thchs_30/data_thchs30"
        file_root = "data_file/thchs_30"
        data_path  = "".join([data_root, "/data"])
        train_path = "".join([data_root, "/train"])
        train_data_path  = "".join([file_root,"/train.txt"])
        wav_files, labels = get_wavs_and_labels(data_path,train_path, train_data_path )    
        print("读取训练集path和label 完成！")

        #验证集
        dev_path = "".join([data_root, "/dev"])
        dev_data_path  = "".join([file_root,"/dev.txt"])
        wav_files, labels = get_wavs_and_labels(data_path,dev_path, dev_data_path)  
        print("读取验证集path和label 完成！")

        #测试集
        test_path = "".join([data_root, "/test"])
        test_data_path  = "".join([file_root,"/test.txt"])
        wav_files, labels = get_wavs_and_labels(data_path,test_path, test_data_path)  
        print("读取测试集path和label 完成！")
        #3、得到标准化特征
        stand_para_path = "data_file/thchs_30/stand_nor.txt"

        #4、训练
        dataset = "thchs_30"    
        model_name      = "model/bi_lstm_150_c3"
        save_address    = "result/{}/model/{}".format(dataset, model_name )

        #定义参数
        input_size, hidden_size, num_layers, out_size = 40, 512, 3, dict_length
        model = bi_lstm_c(
                input_size  = input_size ,
                hidden_size = hidden_size, 
                num_layers  = num_layers , 
                out_size    = out_size   ,        
        )

        epochs  ,batch_size  = 1, 32
        
        print("epochs",epochs)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
 
        train(model, save_address, model_name, epochs,batch_size, train_data_path, test_data_path,dictionary, stand_para_path,device, decode_function) 

    elif train_form =="test":
        print("测试模式")
        
        data_file = "data_file/thchs_30"
        test_data_path  =  data_file+ "/dev.txt"

        #3、计算标准化特征
        stand_para_path = "data_file/thchs_30/stand_nor.txt"
        final_mean, final_stdevs = read_stand_para(stand_para_path)
        input_size, hidden_size, num_layers, out_size = 40, 512, 2, dict_length
        model = bi_lstm_c(
                input_size  = input_size ,
                hidden_size = hidden_size, 
                num_layers  = num_layers , 
                out_size    = out_size   ,        
        ) 
        if torch.cuda.is_available():
            device = torch.device("cuda")
        model      = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        model_name = "bi_lstm_150_c2"
        model.load_state_dict(torch.load("model/{}.pkl".format(model_name))) 

        test(model, dictionary, test_data_path, device, final_mean, final_stdevs ,decode_function)
    else:
        print("模式设置错误")

