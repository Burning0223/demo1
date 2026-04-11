import swanlab
from torch import nn
import torch
from untils import Metrics,EarlyStopping,Cls_Config
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
from data_process import TextClassificationDataset
from model import BertClassifier
from torch.utils.data import DataLoader

def random_seed(seed):
    random .seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer():
    def __init__(self,model,patience,id2label,max_length,num_epochs,batch_size,learning_rate,optimizer,scheduler):
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.early_stopping=EarlyStopping(patience=patience,verbose=True)
        self.id2label=id2label
        self.max_length=max_length
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate


        experiment_name=f"max_length:{self.max_length},num_epochs:{self.num_epochs},batch_size:{self.batch_size},learning_rate:{self.learning_rate}"
        swanlab.init(project="Bert_text_classification",
             experiment_name=experiment_name,
             config={
                "max_length":self.max_length,
                "num_epochs":self.num_epochs,
                "batch_size":self.batch_size,
                "learning_rate":self.learning_rate
             },
             mode="offline")

    def train(self,dataloader):
        self.model.train()
        total_loss=0.0

        for batch in dataloader:
            self.optimizer.zero_grad()
            labels=batch['labels']
            output=self.model(batch)
            preds=torch.argmax(output,dim=1)
            loss=nn.CrossEntropyLoss()(output, labels)
            total_loss+=loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
           
        ave_loss=total_loss/len(dataloader)
        train_metrics=Metrics(labels,preds,self.id2label)
        train_acc=train_metrics.acc
        train_support=train_metrics.support
        return ave_loss,train_acc,train_support
    
    def dev(self,dataloader):
        self.model.eval()
        total_loss=0.0
        
        with torch.no_grad():
            for batch in dataloader:
                output=self.model(batch)
                labels=batch['labels']
                loss=nn.CrossEntropyLoss()(output, labels)
                preds=torch.argmax(output,dim=1)
                total_loss+=loss.item()
        ave_loss=total_loss/len(dataloader)
        dev_metrics=Metrics(labels,preds,self.id2label)
        dev_acc=dev_metrics.acc
        dev_report=dev_metrics.report
        return ave_loss,dev_acc,dev_report

    def train_with_early_stopping(self,train_dataloader,dev_dataloader):
        dev_best_acc=0
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}")
            train_loss,train_acc,train_support=self.train(train_dataloader)
            dev_loss,dev_acc,dev_report=self.dev(dev_dataloader)
        
            print(f"训练损失:{train_loss:.4f}")
            print(f"训练准确率:{train_acc:.4f}")
            print(f"训练集各分类样本数量：{train_support}")
            print(f"验证损失:{dev_loss:.4f}")
            print(f"验证准确率:{dev_acc:.4f}")
            print(f"验证报告:\n{dev_report}")
            
            swanlab.log({
                "epoch":epoch,
                "训练损失":train_loss,
                "训练准确率":train_acc,
                "验证损失":dev_loss,
                "验证准确率":dev_acc
            })

            if dev_acc>dev_best_acc:
                dev_best_acc=dev_acc
                print(f"当前模型最优准确率为：{dev_best_acc}")
            if self.early_stopping(dev_loss,dev_acc,self.model,self.optimizer,self.scheduler,epoch):
                print(f"在epoch{epoch+1}发生训练早停")
                break
            
            
        print(f"模型最优准确率为：{dev_best_acc}")
    
def main():
    config=Cls_Config("Bert_Config.json")
    data_path=config.get("data_path","../0.demo1文本分类/toutiao_cat_data.txt")
    batch_size=config.get("batch_size",32)
    max_length=config.get("max_length",128)
    num_epochs=config.get("num_epochs",10)
    batch_size=config.get("batch_size",16)
    learning_rate=config.get("learning_rate",2e-5)
    patience=config.get("patience",2)
    random_seed(42)

    train_dataset=TextClassificationDataset(data_path=data_path,dataset_type="train",config=config)
    dev_dataset=TextClassificationDataset(data_path=data_path,dataset_type="dev",config=config)

    id2label=train_dataset.id2label
    num_classes=train_dataset.num_classes

    train_dataloader=DataLoader(train_dataset,batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    dev_dataloader=DataLoader(dev_dataset,batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn)
   
    model=BertClassifier(config,num_classes)

    optimizer=torch.optim.AdamW(
        model.parameters(),lr=learning_rate
    )
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    trainer=Trainer(model,patience,id2label,max_length,num_epochs,batch_size,learning_rate,optimizer,scheduler)
    trainer.train_with_early_stopping(train_dataloader,dev_dataloader)
    
if __name__=="__main__":
    main()