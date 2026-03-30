import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import config
from data_process import data_loader,load_data
from model import BertClassifier
from train import train,dev,test
import swanlab
import torch 

def random_seed(seed):
    random .seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
random_seed(42)

swanlab.init(project="Bert_text_classification",
             experiment_name="exp1")

def main():
    train_texts,train_keywords,train_labels=load_data('0.demo1文本分类/train_3k.txt')
    dev_texts,dev_keywords,dev_labels=load_data('0.demo1文本分类/dev_1k.txt')
    ##test_texts,test_keywords,test_labels=load_data('0.demo1文本分类/test_1k.txt')

    train_dataloader=data_loader(train_texts,train_keywords,train_labels,shuffle=True)
    dev_dataloader=data_loader(dev_texts,dev_keywords,dev_labels,shuffle=False)
    

    model=BertClassifier()

    optimizer=torch.optim.AdamW(
        model.parameters(),lr=config.learning_rate,eps=1e-8
    )
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    best_acc=0
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}")
        train_loss,train_acc=train(model,train_dataloader,optimizer,scheduler)
        dev_loss,dev_acc,dev_report=dev(model,dev_dataloader)
        
        print(f"训练损失:{train_loss:.4f}")
        print(f"训练准确率:{train_acc:.4f}")
        print(f"测试损失:{dev_loss:.4f}")
        print(f"测试准确率:{dev_acc:.4f}")
        print(f"测试报告:/n{dev_report}")
        """
        :.4f 这种格式化写法只能用在 f-string 里
        不能直接写在字典里
        """
        swanlab.log({
            "epoch":epoch,
            "训练损失":train_loss,
            "训练准确率":train_acc,
            "测试损失":dev_loss,
            "测试准确率":dev_acc
        })

        if dev_acc>best_acc:
            best_acc=dev_acc
            print(f"当前最优模型准确率为：{best_acc:.4f}")
            torch.save(model.state_dict(),"bert_cls_best.pt")
    
    torch.save(model.state_dict(),'bert_cls_final.pt')
    print(f"最优模型准确率为：{best_acc:.4f}")
    ##test_loss,test_acc,test_report=test(model,test_dataloader)
    #print(f"测试损失:{test_loss:.4f}")
    #print(f"测试准确率:{test_acc:.4f}")
    #print(f"测试报告:/n{test_report}")
if __name__=="__main__":
    main()
    