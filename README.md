# chinese_asr_demo
一个端到端的中文语音识别系统
### 使用模型
cnn + lstm  + ctc

## 生成训练文件格式
../data/thchs_30/data_thchs30/train/A36_199.wav,该院特聘钱伟长唐克为顾问聂卫平为名誉院长
../data/thchs_30/data_thchs30/train/C14_683.wav,美国一些机构往往机关算尽结果是越搅越混越抹越黑最后常常损及自身
../data/thchs_30/data_thchs30/train/C4_528.wav,用人单位聘用外国人从事的岗位应是有特殊需要国内暂缺适当人选且不违反国家有关规定的岗位
../data/thchs_30/data_thchs30/train/C17_684.wav,但有一个桥墩外表光洁度不够影响桥梁外观有人认为这一小疵点无碍质量不必返工
../data/thchs_30/data_thchs30/train/C6_726.wav,如果超过成本目标一概否决要么干要么让位要么受奖要么挨罚
../data/thchs_30/data_thchs30/train/A34_208.wav,此案中盗窃文物的刘农军刘进文西山李军等四名主犯也在九月十八日被依法判处
../data/thchs_30/data_thchs30/train/B33_468.wav,早被国外农户广泛采用的微生物菌养禽畜技术已落户中国

## 字典
![image](https://user-images.githubusercontent.com/53568883/142418123-b8314cbc-c091-493e-a394-9eb59175c44c.png)

## 2、模块化，分为预处理，训练，以及测试
### 预处理
提取fbank特征后进行标准化

### 模型下载
百度云链接：https://pan.baidu.com/s/1VVavLKLeY584HudHtC5uIQ 
提取码：n18y
