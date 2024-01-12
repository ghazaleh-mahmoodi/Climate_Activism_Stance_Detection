from constant import *
from imports import *
from model import *
from losses import *


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, trial):
    model = model.to(device)
    model.train()
    train_loss = 0
    train_acc = 0
    actuals, predictions = [], []

    loop = tqdm(dataloader, total=len(dataloader),desc='Train')

    for b , data in enumerate(loop):
        optimizer.zero_grad()
        mask = data['attention_mask'].to(device)
        ids = data['input_ids'].to(device)
        labels = data['label'].type(torch.LongTensor).to(device)
        out = model(ids, mask)

        cur_train_loss = criterion(out, labels)
        train_loss += cur_train_loss.item()

        actuals.extend(labels.cpu().numpy().astype(int))
        predictions.extend(F.softmax(out, 1).cpu().detach().numpy())
      
        cur_train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


    predictions = np.array(predictions)
    predicted_labels = predictions.argmax(1)
    accuracy = (predicted_labels == actuals).mean()
    precision = precision_score(actuals, predicted_labels, average='macro')
    recall = recall_score(actuals, predicted_labels, average='macro')
    f1 = f1_score(actuals, predicted_labels, average='macro')
    return train_loss/len(dataloader) , accuracy, precision, recall, f1

def valid_one_epoch(model,dataloader, criterion):

    model = model.to(device)
    val_loss = 0
    val_acc = 0
    actuals, predictions = [], []

    model.eval()
    with torch.no_grad():

        loop = tqdm(dataloader, total=len(dataloader),desc='Valid')

        for b , data in enumerate(loop):
            mask = data['attention_mask'].to(device)
            ids = data['input_ids'].to(device)
            labels = data['label'].type(torch.LongTensor).to(device)
            out = model(ids, mask)

            actuals.extend(labels.cpu().numpy().astype(int))
            predictions.extend(F.softmax(out, 1).cpu().detach().numpy())

            cur_valid_loss = criterion(out, labels)
            val_loss += cur_valid_loss.item()


    predictions = np.array(predictions)
    predicted_labels = predictions.argmax(1)
    accuracy = (predicted_labels == actuals).mean()
    precision = precision_score(actuals, predicted_labels, average='macro')
    recall = recall_score(actuals, predicted_labels, average='macro')
    f1 = f1_score(actuals, predicted_labels, average='macro')
    return val_loss/len(dataloader), accuracy, precision, recall, f1

def train_model(bert, train_dataloader, dev_dataloader, test_dataloader, class_weights, total_steps, params, trial):

    history = {'train_loss': [], 'train_acc': [], 'train_f1' : [],  'train_recall': [], 'train_precision': [],
               'val_loss': [], 'val_acc': [], 'val_f1' : [], 'val_recall': [], 'val_precision': [],
               'test_loss': [], 'test_acc': [], 'test_f1' : [], 'test_f1' : [], 'test_recall': [], 'test_precision': []}
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0

    if params['cleassifier'] =='cnn':
        model = CNNBert(bert, params['Dropout']).to(device)
    else:
        model = lm_classifier(bert, params['Dropout']).to(device)
    
    # criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    # criterion = FocalLoss(num_class=3)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 3e-2, weight_decay=3e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    if params['loss'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    elif params['loss'] == 'focal':
        criterion = FocalLoss(gamma = params['focal_gamma'], num_class=3)   
    else:
        criterion = SelfAdjDiceLoss()
   
    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])
    # optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = int(total_steps/10), # Default value in run_glue.py
                                            num_training_steps = total_steps)

    for epoch in range(NUM_EPOCHS):
        #precision, recall, f1
        train_loss , train_acc, train_precision, train_recall, train_f1 = train_one_epoch(model=model, dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, trial=trial)
        val_loss , val_acc, val_precision, val_recall, val_f1 = valid_one_epoch(model=model, dataloader=dev_dataloader, criterion=criterion)
        test_loss , test_acc, test_precision, test_recall, test_f1 = valid_one_epoch(model=model, dataloader=test_dataloader, criterion=criterion)

        print(f"\n Epoch:{epoch + 1} / {NUM_EPOCHS},train loss:{train_loss:.5f}, train acc: {train_acc:.5f}, train f1:{train_f1:.5f}, valid loss:{val_loss:.5f}, valid acc:{val_acc:.5f}, valid f1:{val_f1:.5f}, test loss:{test_loss:.5f}, test acc:{test_acc:.5f}, test f1:{test_f1:.5f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_recall'].append(train_recall)
        history['train_precision'].append(train_precision)
        history['train_f1'].append(train_f1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_recall'].append(val_recall)
        history['val_precision'].append(val_precision)
        history['val_f1'].append(val_f1)
        
        history['test_loss'].append(train_loss)
        history['test_acc'].append(train_acc)
        history['test_recall'].append(test_recall)
        history['test_precision'].append(test_precision)
        history['test_f1'].append(train_f1)
       

        if val_f1 > best_f1:
            if val_f1 > F1_THERESHOLD:
                torch.save(model.state_dict(),'model/best_bert.pth')
            best_f1 = val_f1
            best_precision = val_precision
            best_recall = val_recall
        elif val_f1 == best_f1:
            break
        elif val_f1 < 0.40:
            break
        trial.report(val_f1, epoch)

        if trial.should_prune():
            print("prune!")
            raise optuna.TrialPruned()
    
    if best_f1 > F1_THERESHOLD :
        torch.save(model.state_dict(),f'model/best_bert_trial_{params["trial_num"]}.pth')
        torch.save(model.state_dict(),'model/best_bert.pth')
        df = pd.DataFrame.from_dict(history)
        df.to_csv(f'result/report_trial_{params["trial_num"]}.csv', index = False, header=True)

    return model, best_precision, best_recall, best_f1

def model_prediction(model, dataloader, path='model/best_bert.pth'):
    model = model.to(device)
    model.load_state_dict(torch.load(path))

    predictions = []
    model.eval()
    with torch.no_grad():
        loop = tqdm(dataloader, total=len(dataloader), desc='Test')
        for b , data in enumerate(loop):
            mask = data['attention_mask'].to(device)
            ids = data['input_ids'].to(device)
            out = model(ids,mask)
            predictions.extend(F.softmax(out, 1).cpu().detach().numpy())

    predictions = np.array(predictions)
    predicted_labels = predictions.argmax(1)
    predicted_labels = list(predicted_labels)
    predicted_labels = [int(x)+1 for x in predicted_labels]

    return predicted_labels