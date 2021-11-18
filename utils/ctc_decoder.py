import numpy as np
import Levenshtein 

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

#束搜索
def beam_decode(index_list, dictionary, blank, decode_model="real",beam_size=5,):
    
    #对预测结果解码
    decode_out = []    
    if decode_model == "prediction":  
        pred = index_list
        print(pred.shape)
        label_number, T, F  = pred.shape
        for k in range(label_number):
            print(pred[k].shape)
            print("pred",pred[k])
            sm_pred  = softmax(pred[k])
            print("sm",sm_pred)
            log_pred = np.log(sm_pred)
            result = [([], 0)]
            for t in range(T):
                temp_beam = []
                for prefix, score in result:
                    for i in range(F):
                        new_prefix = prefix + [i]
                        new_score  = score + log_pred[t,i]
                        temp_beam.append((new_prefix, new_score)) 
                temp_beam.sort(key=lambda x:x[1], reverse=True)
                result = temp_beam[:beam_size]   
            result = result[0][0]
            decode_out.append(result)
        return decode_out

    #解码真实标签
    elif decode_model=="real":
        pred  = index_list
        label_number  = len(pred)
        
        for i in range(label_number):
            temp_labels = remove_blank(out[i], blank)
            words = words_decode(temp_labels, dictionary) 
            decode_out.append(words)


        return decode_out

def words_decode(temp_labels, dictionary):
    words = [dictionary[index] for index in temp_labels]
    words = " ".join(words)    
    return words



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
