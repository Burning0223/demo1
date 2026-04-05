import torch
from model import BertClassifier
from data_process import data_loader,load_data,data_split
from train import Trainer
import config
def test():
    texts,keywords,labels=load_data('0.demo1文本分类/toutiao_cat_data.txt')
    train_texts,train_keywords,train_labels,dev_texts,dev_keywords,dev_labels,test_texts,test_keywords,test_labels=data_split(
        texts,keywords,labels
    )
    test_dataloader=data_loader(test_texts,test_keywords,test_labels,shuffle=False)
    model=BertClassifier()
    checkpoint=torch.load("checkpoint.pt") #手动填写
    model.load_state_dict(checkpoint['model'])

    trainer=Trainer(model,optimizer=False,scheduler=False,patience=config.patience)
    test_loss,test_acc,test_report=trainer.dev(test_dataloader)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试报告:\n{test_report}")

if __name__=='__main':
    test()
