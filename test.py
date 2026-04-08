import torch
from model import BertClassifier
from train import Trainer
from Bert_Config import Cls_Config
from data_process import TextClassificationDataset
from torch.utils.data import DataLoader
def test():
    config=Cls_Config("Bert_Config.json")
    data_path=config.get("data_path","../0.demo1文本分类/toutiao_cat_data.txt")
    batch_size=config.get("batch_size",32)
    patience=config.get("patience",2)

    test_dataset=TextClassificationDataset(data_path=data_path,dataset_type="test",config=config)
    test_dataloader=DataLoader(test_dataset,batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)

    model=BertClassifier()
    checkpoint=torch.load("checkpoint.pt") #手动填写
    model.load_state_dict(checkpoint['model'])

    trainer=Trainer(model,optimizer=None,scheduler=None,patience=patience)
    test_loss,test_acc,test_report=trainer.dev(test_dataloader)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试报告:\n{test_report}")

if __name__=="__main__":
    test()
