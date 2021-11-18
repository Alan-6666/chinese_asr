import numpy as np
import Levenshtein 

#去除blank
def remove_blank(labels, blank=0):
    out_labels = []
    pre = labels[0]
    #去掉重复
    for i in range(1,len(labels)):
        temp = labels[i]
        if temp != pre:
            out_labels.append(temp)
            pre = temp
    #去掉blank
    out_labels = [label for label in out_labels if label!=blank]
    return out_labels

#softmax函数
def softmax(array):
    #输入 t,f，输出,t,f
    t,f = array.shape
    e_x = np.exp(array - np.max(array,axis=1).reshape(t, -1))
    out = e_x / e_x.sum(axis=1).reshape(t, -1)    
    return out

#贪心搜索
def greedy_decode(index_list,dictionary ,blank,decode_model="real"):
    
    if decode_model == "prediction":
        out       = np.argmax(index_list,axis=2)
        label_number = out.shape[0] 
#        print("out_shape",out.shape)
    elif decode_model=="real":
        out        = index_list
        label_number = len(out)
    decode_out = []
    for i in range(label_number):
#        if decode_model=="prediction":
#        print("out_i", out[i])
        temp_labels = remove_blank(out[i], blank)
#        print("temp",temp_labels)
        words = words_decode(temp_labels, dictionary) 
#        print(words)
        decode_out.append(words)
    #根据字典获得文本
    return decode_out

#下标转文字
def words_decode(temp_labels, dictionary):
    words = [dictionary[index] for index in temp_labels]
    words = " ".join(words)    
    return words

#计算字错误率
def compute_wer(real_words, pred_words, dictionary):

    ls_sum      = 0
    words_length = 0
    words_num    = len(real_words)
    for i in range(words_num):
        real , pred = real_words[i].replace(" ",""), pred_words[i].replace(" ","")
        print("real",real)
        print("pred",pred)
        ls = Levenshtein.distance(real,  pred)
        ls_sum +=ls
        words_length += len(real)
    wer_average =  ls_sum/words_length

    return wer_average

if __name__=="__main__":
    labels = [0,1,0,2,2,0,2,0,3,0,3,0,0]
    out=remove_blank(labels) 
    print("labels",labels)
    print("out", out)
