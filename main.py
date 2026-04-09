import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import Bert_Config
from data_process import TextClassificationDataset
from model import BertClassifier
from train import Trainer
import torch 
from Bert_Config import Cls_Config
from torch.utils.data import DataLoader


def random_seed(seed):
    random .seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
random_seed(42)



def main():
    config=Cls_Config("Bert_Config.json")
    data_path=config.get("data_path","../0.demo1文本分类/toutiao_cat_data.txt")
    batch_size=config.get("batch_size",32)

    train_dataset=TextClassificationDataset(data_path=data_path,dataset_type="train",config=config)
    dev_dataset=TextClassificationDataset(data_path=data_path,dataset_type="dev",config=config)

    id2label=train_dataset.id2label
    num_classes=train_dataset.num_classes

    train_dataloader=DataLoader(train_dataset,batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    dev_dataloader=DataLoader(dev_dataset,batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn)
   
    model=BertClassifier(config,num_classes)

    optimizer=torch.optim.AdamW(
        model.parameters(),lr=Bert_Config.learning_rate
    )
    total_steps = len(train_dataloader) * Bert_Config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    trainer=Trainer(model,optimizer,scheduler,patience=Bert_Config.patience,id2label=id2label)
    trainer.train_with_early_stopping(train_dataloader,dev_dataloader)
    
if __name__=="__main__":
    main()
    