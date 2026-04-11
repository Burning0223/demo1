import torch
from model import BertClassifier
from run_trainer import Trainer
from untils import Cls_Config,EarlyStopping
from data_process import TextClassificationDataset
from torch.utils.data import DataLoader
def test():
    config=Cls_Config("Bert_Config.json")
    test_dataset=TextClassificationDataset(dataset_type="test",config=config)
    test_dataloader=DataLoader(test_dataset,config.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)
    num_classes=test_dataset.num_classes
    id2label=test_dataset.id2label
    model=BertClassifier(config,num_classes)

    early_stopping=EarlyStopping(config)
    if not early_stopping.best_model_path:
        print("未找到最优模型，无法进行测试！")
        return
    else:
        checkpoint=torch.load(early_stopping.best_model_path)
        model.load_state_dict(checkpoint['model'])
        best_epoch=checkpoint['epoch']+1
        print(f"加载的最优模型来自第{best_epoch}个epoch")

    trainer=Trainer(model,config,id2label,optimizer=None,scheduler=None)
    test_loss,test_acc,test_report=trainer.dev(test_dataloader)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试报告:\n{test_report}")

if __name__=="__main__":
    test()
