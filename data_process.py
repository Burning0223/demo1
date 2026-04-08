import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


class TextClassificationDataset(Dataset):
    def __init__(self,data_path,dataset_type,config):
        super().__init__()
        self.texts,self.keywords,self.labels=self.load_data(data_path)
        self.train_texts,self.train_keywords,self.train_labels,\
        self.dev_texts,self.dev_keywords,self.dev_labels,\
        self.test_texts,self.test_keywords,self.test_labels=self.data_split(
            self.texts,self.keywords,self.labels
            )
        
        if dataset_type=="train":
            self.texts,self.keywords,self.labels=self.train_texts,self.train_keywords,self.train_labels
        elif dataset_type=="dev":
            self.texts,self.keywords,self.labels=self.dev_texts,self.dev_keywords,self.dev_labels
        elif dataset_type=="test":
            self.texts,self.keywords,self.labels=self.test_texts,self.test_keywords,self.test_labels
        else:
            print("dataset_type无效")
        
        self.tokenizer=BertTokenizer.from_pretrained(config.get("model_path","../bert-base-uncased"))
        self.max_length=config.get("max_length",128)
        self.test_size=config.get("test_size",0.15)
        self.dev_size=config.get("dev_size",0.15)
        self.random=config.get("random",42)
    def load_data(self,data_path):
    
        texts=[]
        labels=[]
        keywords=[]

        labels2id={
            100:0,101:1,102:2,103:3,104:4,106:5,
            107:6,108:7,109:8,110:9,112:10,
            113:11,114:12,115:13,116:14
        }

        with open(data_path,'r',encoding='utf-8') as f:
            for line in f:
                parts=line.strip().split("_!_")

                label_code=int(parts[1])
                text=parts[3]
                keyword=parts[4]

                keyword=keyword.replace(","," ")

                label=labels2id[label_code]

                texts.append(text)
                keywords.append(keyword)
                labels.append(label)

        return texts,keywords,labels
    
    def data_split(self,texts,keywords,labels):
        x_train,x_temp,y_train,y_temp=train_test_split(
            texts,labels,test_size=self.test_size+self.dev_size,random_state=self.random,stratify=labels
        )
        x_dev,x_test,y_dev,y_test=train_test_split(
            x_temp,y_temp,test_size=0.5,random_state=self.random,stratify=y_temp
        )
        keywords_train=keywords[:len(x_train)]
        keywords_dev=keywords[len(x_train):len(x_train)+len(x_dev)]
        keywords_test=keywords[len(x_train)+len(x_dev):]
        return x_train,keywords_train,y_train,x_dev,keywords_dev,y_dev,x_test,keywords_test,y_test

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        text=self.texts[idx]
        keyword=self.keywords[idx]
        label=self.labels[idx]
        return text,keyword,label

    def collate_fn(self,batch):
        texts,keywords,labels=zip(*batch)
        encoding=self.tokenizer(list(texts),list(keywords),return_tensors='pt',max_length=self.max_length,
                            padding=True,truncation=True)
        return {
                'input_ids':encoding['input_ids'],
                'attention_mask':encoding['attention_mask'],
                'token_type_ids':encoding['token_type_ids'],
                'labels':torch.tensor(labels)
                }
    


    