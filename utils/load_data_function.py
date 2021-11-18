import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#torchaudio提取fank特征
class get_features():
    def __init__(self,audio_path):
        self.audio_path   = audio_path
        self.num_mel_bins = 80
        self.frame_length = 25
        self.frame_shift  = 10
        self.dither       = 0.1
        self.energy_floor = 0.0
     
    def fbank(self):
        waveform, sample_rate = torchaudio.load(self.audio_path)
        mat =  kaldi.fbank(waveform,
                      num_mel_bins= self.num_mel_bins,
                      frame_length= self.frame_length,
                      frame_shift= self.frame_shift,
                      dither= self.dither,
                      energy_floor= self.energy_floor,
                      sample_frequency=sample_rate)
        return mat
    def mfcc(self): 
        waveform, sample_rate = torchaudio.load(self.audio_path)
        mat =  kaldi.mfcc(waveform,
                      num_mel_bins= self.num_mel_bins,
                      frame_length= self.frame_length,
                      frame_shift= self.frame_shift,
                      dither= self.dither,
                      energy_floor= self.energy_floor,
                      sample_frequency=sample_rate)
        return mat


#获得特征归一化的均值和方差
def stand_para(train_data_path):
    
    mean_list      = [0]*80
    std_list       = [0]*80
    features_list  = []
    with open(train_data_path) as f:
        path_list = f.readlines()
        
    path_list = [path.split(",")[0] for path in path_list]
#    path_list = ["A11_119.wav", "A11_1.wav"]
    data_num  = len(path_list)
    for i in tqdm(range(80)):
        standard_deviation = 0
        for path in path_list:
            feature = get_features(path).fbank()
            mean_list[i] += feature[:,i].mean()
            std_list[i] += feature[:,i].std()

    final_means = np.asarray(mean_list) / data_num
    final_stdevs = np.asarray(std_list) / data_num

    print("mean", final_means)
    print("final_stdevs", final_stdevs)
    return torch.tensor(final_means), torch.tensor(final_stdevs)

#归一化函数
def standardization(feature,final_mean,final_stdevs):
    out = (feature-final_mean)/final_stdevs
    return out

#获得地址以及标签
def get_wavs_and_labels(wav_path,train_path,file_name):
    wav_files = []
    labels = []
    for (dirpath, dirnames, filenames) in os.walk(train_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                # print(filename)
                filename_path = os.path.join(dirpath, filename)
                # print(filename_path)
                wav_files.append(filename_path)
                #读取label
                label_file = os.path.join(wav_path, filename + ".trn")
                with open(label_file, encoding='utf-8', errors='ignore') as f:
                    label = f.readline()
                    labels.append(label.replace(" ","").strip())
    with open(file_name,"w") as f:
        for i in range(len(wav_files)):
             f.write("".join([wav_files[i],",",labels[i],"\n"]))

    return wav_files, labels
        
#dataset函数获取特征以及label
class ASR_Dataset(Dataset):
    def __init__(self, index_path, asr_dict,final_mean, final_stdevs):
        with open(index_path) as f:
            idx = f.readlines()
        idx = [x.strip().split(",", 1) for x in idx]
        self.idx = idx

        self.asr_dict = [x.replace("\n","") for x in asr_dict]

    def __getitem__(self, index):
        audio_path, transcript = self.idx[index]
 
        #计算特征
        feature = get_features(audio_path).fbank()
        feature = standardization(features,final_mean, final_stdevs)
        transcript = [self.asr_dict.index(x) for x in transcript]
        return feature, transcript

    def __len__(self):
        return len(self.idx)


def _collate_fn(batch):
   #获取特征的tensor
   batch_size = len(batch)
   features_input = [x[0] for x in batch]
   features_max_length = max([len(x) for x in features_input])
   feature_length = 80
   features_temp = [torch.cat([x,torch.zeros(features_max_length-len(x),feature_length)]) for x in  features_input]  #填充0
   features_out  = features_temp[0]
   for i in range(1,batch_size):
       features_out = torch.cat([features_out,features_temp[i]])
   features_out = features_out.reshape(batch_size,-1,feature_length)
   #获取label的tensor
   labels   = [x[1] for x in batch]
   labels_length = [len(x) for x in labels]
      
   return features_out,  labels, labels_length

class ASR_DataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ASR_DataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


if __name__=="__main__":
    
    #生成地址以及labels
    data_path  = "../data/thchs_30/data_thchs30/data"
    dev_path  = "../data/thchs_30/data_thchs30/dev"
    file_name = "txt_data/thchs_30/test.txt"
    wav_files, labels = get_wavs_and_labels(data_path,dev_path,file_name)     
    print("wav_files",wav_files[:4])
    print("labels", labels[:4])    


    labels_path = "txt_data/thchs_30/dictionary.txt"
    data_path   = "txt_data/thchs_30/train.txt"  
    train_dataset = ASR_Dataset(data_path,labels_path) #输出，特征地址以及文字标签
   
 
#    print(train_dataset[0])
    train_dataloader = ASR_DataLoader(train_dataset,batch_size=2, num_workers=8,shuffle=True)
     
    for i,(features, labels, label_length) in enumerate(train_dataloader):
        print("batch{}".format(i))
        print(features.size())
        print(features)
       # print(data[0])
        #print(label)
        if i > 4:break
