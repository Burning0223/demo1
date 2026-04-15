import torch
import os
from model import BertClassifier
from run_trainer import Trainer
from untils import Cls_Config
from data_process import TextClassificationDataset
from torch.utils.data import DataLoader
def test():
    config=Cls_Config("Bert_Config.json")
    test_dataset=TextClassificationDataset(dataset_type="test",config=config)
    test_dataloader=DataLoader(test_dataset,config.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)
    id2label=test_dataset.id2label
    model=BertClassifier(config)

    checkpoint_path="experiment\max_length_128_num_epochs_15_batch_size_16_lr_2e-05\checkpoint_epoch_2.pt"
    if not os.path.exists(checkpoint_path):
        print("未找到最优模型，无法进行测试！")
        return
    else:
        checkpoint=torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])

    trainer=Trainer(model,config,id2label,optimizer=None,scheduler=None)
    test_loss,test_acc=trainer.dev(test_dataloader)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")

if __name__=="__main__":
    test()
