from torch import nn
import torch
from sklearn.metrics import accuracy_score,classification_report

def train(model,dataloader,optimizer,scheduler):
    model.train()
    total_loss=0.0
    all_labels=[]
    all_preds=[]
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids=batch['input_ids']
        attention_mask=batch['attention_mask']
        token_type_ids=batch['token_type_ids']
        labels=batch['labels']
        output=model(input_ids,attention_mask,token_type_ids)
        pred=torch.argmax(output,dim=1)
        loss=nn.CrossEntropyLoss()(output, labels)
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        all_labels.extend(labels.numpy())
        all_preds.extend(pred.numpy())
    
    ave_loss=total_loss/len(dataloader)
    acc=accuracy_score(all_labels,all_preds)
    return ave_loss,acc

def dev(model,dataloader):
    model.eval()
    total_loss=0.0
    all_labels=[]
    all_preds=[]
    with torch.no_grad():
        for batch in dataloader:
            input_ids=batch['input_ids']
            attention_mask=batch['attention_mask']
            token_type_ids=batch['token_type_ids']
            labels=batch['labels']
            output=model(input_ids,attention_mask,token_type_ids)
            loss=nn.CrossEntropyLoss()(output, labels)
            pred=torch.argmax(output,dim=1)
            total_loss+=loss.item()
            all_labels.extend(labels.numpy())
            all_preds.extend(pred.numpy())
    ave_loss=total_loss/len(dataloader)
    acc=accuracy_score(all_labels,all_preds)
    report=classification_report(all_labels,all_preds)
    return ave_loss,acc,report

def test(model,dataloader):
    model.eval()
    total_loss=0.0
    all_labels=[]
    all_preds=[]
    with torch.no_grad():
        for batch in dataloader:
            input_ids=batch['input_ids']
            attention_mask=batch['attention_mask']
            token_type_ids=batch['token_type_ids']
            labels=batch['labels']
            output=model(input_ids,attention_mask,token_type_ids)
            loss=nn.CrossEntropyLoss()(output, labels)
            pred=torch.argmax(output,dim=1)
            total_loss+=loss.item()
            all_labels.extend(labels.numpy())
            all_preds.extend(pred.numpy())
    ave_loss=total_loss/len(dataloader)
    acc=accuracy_score(all_labels,all_preds)
    report=classification_report(all_labels,all_preds)
    return ave_loss,acc,report