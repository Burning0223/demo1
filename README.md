# 新闻文本分类任务
基于 bert-base-chinese模型，使用 PyTorch 和 Hugging Face ，完成文本分类任务的训练与评估。
## 数据集
来源：https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset  来自今日头条客户端，共15个类别：
``` 
100 民生 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu 
109 科技 科技 news_tech
110 军事 军事 news_military
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
``` 
## 项目结构
``` 
├── Bert_Config.json   # 配置参数
├── untils.py          # 功能类
├── model.py           # BERT 模型定义
├── data_process.py    # 数据预处理
├── run_trainer.py     # 训练器以及程序入口       
├── test.py            # 测试评估
├── requirements.txt   # 环境依赖
└── README.md          # 项目说明
``` 
## 环境依赖
``` 
pip install -r requirements.txt
``` 
## 模型参数设置
``` 
    "model_path":"E:/demo1/bert-base-chinese",
    "max_length":128,
    "num_epochs":15,
    "batch_size":16,
    "learning_rate":2e-5,
    "patience":3,
    "num_classes":15,
    "data_path":"./0.demo1文本分类/toutiao_cat_data.txt",
    "dropout":0.1,
    "test_size":0.15,
    "dev_size":0.15,
    "random":42,
    "delta":0
``` 
## 实验结果
``` 
训练损失:0.3297
训练准确率:0.9019
验证损失:0.3803
验证准确率:0.8906
训练集报告：
Class     Precision      Recall         F1-Score       Support   
100       0.87           0.86           0.87           4391      
101       0.90           0.91           0.90           19622     
102       0.92           0.92           0.92           27577     
103       0.96           0.95           0.95           26298     
104       0.84           0.83           0.83           18960     
106       0.93           0.93           0.93           12370     
107       0.94           0.94           0.94           25049     
108       0.91           0.92           0.92           18941     
109       0.88           0.89           0.89           29080     
110       0.89           0.89           0.89           17489     
112       0.86           0.88           0.87           14995     
113       0.86           0.85           0.85           18836     
114       0.50           0.00           0.01           238       
115       0.89           0.90           0.89           13525     
116       0.93           0.93           0.93           20510     

macro avg 0.87           0.84           0.84           267881    

weighted avg0.90           0.90           0.90           267881
验证集报告：
Class     Precision      Recall         F1-Score       Support   
100       0.84           0.82           0.83           941       
101       0.85           0.91           0.88           4204      
102       0.90           0.92           0.91           5909      
103       0.96           0.93           0.95           5635      
104       0.83           0.82           0.83           4062      
106       0.92           0.92           0.92           2651      
107       0.95           0.92           0.93           5368      
108       0.91           0.90           0.90           4058      
109       0.87           0.89           0.88           6231      
110       0.87           0.89           0.88           3748      
112       0.84           0.85           0.85           3214      
113       0.87           0.82           0.85           4037      
114       0.00           0.00           0.00           51        
115       0.86           0.89           0.88           2899      
116       0.93           0.90           0.91           4395      

macro avg 0.83           0.83           0.83           57403     

weighted avg0.89           0.89           0.89           57403

测试损失: 0.3791
测试准确率: 0.8903
测试集报告：
Class     Precision      Recall         F1-Score       Support   
100       0.82           0.83           0.83           941       
101       0.85           0.92           0.88           4205      
102       0.90           0.91           0.91           5910      
103       0.96           0.94           0.95           5635      
104       0.83           0.81           0.82           4063      
106       0.93           0.92           0.92           2651      
107       0.94           0.92           0.93           5368      
108       0.92           0.90           0.91           4059      
109       0.86           0.88           0.87           6232      
110       0.86           0.90           0.88           3747      
112       0.85           0.85           0.85           3213      
113       0.87           0.82           0.84           4036      
114       0.00           0.00           0.00           51        
115       0.87           0.89           0.88           2898      
116       0.94           0.90           0.92           4395      

macro avg 0.83           0.83           0.83           57404     

weighted avg0.89           0.89           0.89           57404 
``` 
<img width="1490" height="578" alt="image" src="https://github.com/user-attachments/assets/322fbe66-f6d4-455b-834f-20fce04ca023" />
<img width="1497" height="582" alt="image" src="https://github.com/user-attachments/assets/c97252d7-147e-43e9-80a7-1e61e2c6f24e" />


