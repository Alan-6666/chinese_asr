# chinese_asr_demo
一个端到端的中文语音识别模型训练、测试框架
具备数据预处理、模型训练、解码、计算wer等等功能
### 训练数据
训练数据采用thchs_30，下载地址：http://www.openslr.org/18/
下载后将data_thchs_30解压到data文件中
### 使用模型
bi-lstm  + ctc
可直接将语音转换为中文

## 生成训练文件格式
![image](https://user-images.githubusercontent.com/53568883/142419223-2640cd2c-8479-4a92-b977-798eb5136298.png)

## 字典
_ 作为ctc的blank

![image](https://user-images.githubusercontent.com/53568883/142418123-b8314cbc-c091-493e-a394-9eb59175c44c.png)

### 预处理
1、提取80维fbank特征

2、标准化：计算特征的均值和方差进行标准化，加快模型收敛速度，因为计算标准化的速度较慢，已将参数放到data_file/thchs_30stand_nor.txt文件中，可直接调用

### 模型下载
百度云链接：https://pan.baidu.com/s/1VVavLKLeY584HudHtC5uIQ 
提取码：n18y

### 计划
1、加入CNN降维度

2、更新tranformer，conformer模型

3、更新其他数据集的训练方法

4、更新其他解码方法
