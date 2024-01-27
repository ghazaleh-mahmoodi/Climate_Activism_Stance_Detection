from imports import *
from dataset import *
from train import *
from constant import *


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def create_submission_file(df_test, path=f'result/submission.json'):
    predictions = []
    for _, row in df_test.iterrows():
        predictions.append({"index":  int(row['index']), "prediction": int(row['prediction'])})
    
    predictions = sorted(predictions, key=lambda i: i['index'])
    with open(path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in predictions))
          
def prepare_dataloader(tokenizer, study_name, data_aug_file, BATCH_SIZE):
    TEXT_COLUMN_NAME = study_name.replace('climate_stance_experiment_on_column_', '').replace('data_aug_', '')
    TEXT_COLUMN_NAME = TEXT_COLUMN_NAME.replace('_10', '').replace('_20', '').replace('_40', '').replace('_1', '').replace('_2', '').replace('_4', '')

    df_train = pd.read_csv(TRAIN_FILE_PATH)
    X_train = df_train[TEXT_COLUMN_NAME].tolist()
    df_train['label_LM_output'] = df_train['label'] - 1
    y_train = df_train['label_LM_output'].tolist()
    
    if data_aug_file != '':
        print("aug")
        df_aug = pd.read_csv(data_aug_file)
        X_train_aug = df_aug[TEXT_COLUMN_NAME].tolist()
        df_aug['label_LM_output'] = df_aug['label'] - 1
        y_train_aug = df_aug['label_LM_output'].tolist()
        print("before : ", len(X_train))
        X_train.extend(X_train_aug)
        print("after :", len(X_train))
        y_train.extend(y_train_aug)
        
        
    trainset = ClimateTextDataset(X_train, y_train, tokenizer, max_token_len=MAX_LENGHT)
    train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    total_steps = len(y_train) * NUM_EPOCHS

    df_dev= pd.read_csv(DEV_FILE_PATH)
    X_dev = df_dev[TEXT_COLUMN_NAME].tolist()
    df_dev['label_LM_output'] = df_dev['label'] - 1
    y_dev = df_dev['label_LM_output'].tolist()
    devset = ClimateTextDataset(X_dev, y_dev, tokenizer, max_token_len=MAX_LENGHT)
    dev_dataloader = DataLoader(devset, batch_size=BATCH_SIZE, shuffle=False)

    df_test = pd.read_csv(TEST_FILE_PATH)
    X_test = df_test[TEXT_COLUMN_NAME].tolist()
    if 'label' in list(df_test.columns):
        df_test['label_LM_output'] = df_test['label'] - 1
        y_test = df_test['label_LM_output'].tolist()
        testset = ClimateTextDataset(X_test, y_test, tokenizer, max_token_len=MAX_LENGHT)
    else:
        testset = ClimateTextDatasetTestPhase(X_test, tokenizer, max_token_len=MAX_LENGHT)
    test_dataloader = DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False)
    
    df_semeval_climate  = pd.read_csv(SEMEVAL_CLIMATE)
    X_semeval_climate = df_semeval_climate[TEXT_COLUMN_NAME].tolist()
    y_semeval_climate = df_semeval_climate['label']
    semeval_climate_set = ClimateTextDataset(X_semeval_climate, y_semeval_climate, tokenizer, max_token_len=MAX_LENGHT)
    semeval_climate_dataloader = DataLoader(semeval_climate_set, batch_size=BATCH_SIZE, shuffle=False)

    df_semeval_abortion = pd.read_csv(SEMEVAL_ABORTIPN)
    X_semeval_abortion = df_semeval_abortion[TEXT_COLUMN_NAME].tolist()
    y_semeval_abortion = df_semeval_abortion['label']
    semeval_abortion_set = ClimateTextDataset(X_semeval_abortion, y_semeval_abortion, tokenizer, max_token_len=MAX_LENGHT)
    semeval_abortion_dataloader = DataLoader(semeval_abortion_set, batch_size=BATCH_SIZE, shuffle=False)


    class_weights= compute_class_weight('balanced', classes = np.unique(y_train), y = np.array(y_train))
    class_weights= torch.tensor(class_weights,dtype=torch.float).to(device)

    return train_dataloader, dev_dataloader, test_dataloader, class_weights, total_steps, df_dev, df_test, semeval_climate_dataloader, semeval_abortion_dataloader

class objective(object):
    def __init__(self, study_name, data_aug_file):
        # Hold this implementation specific arguments as the fields of the class.
        self.study_name = study_name
        self.data_aug_file = data_aug_file  

    def __call__(self, trial):
        params= {
                    'cleassifier' : trial.suggest_categorical('cleassifier', ['cnn', 'mlp']),
                    'Dropout' : trial.suggest_float('Dropout', 0.1, 0.5, step = 0.1),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
                    'BATCH_SIZE' :  trial.suggest_categorical('BATCH_SIZE', [4, 8]),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", 'AdamW']),
                    'loss' : trial.suggest_categorical("loss", ["cross_entropy", "focal"]),
                    'num_warmup_steps_divide' : trial.suggest_int("num_warmup_steps_divide", 2, 15, step=1)
        } 
        params["study_name"] = self.study_name
        params["trial_num"] = trial.number
        
        
        if params['cleassifier'] == 'cnn': 
            params['In_Channel']= trial.suggest_int('In_Channel', 1, 5, step = 1)
            params['MODEL_NAME']= trial.suggest_categorical('MODEL_NAME_CNN', ['vinai/bertweet-base', 'bert-base-uncased', 'xlm-roberta-base','roberta-base'])
            bert = AutoModel.from_pretrained(params['MODEL_NAME'], output_hidden_states=True)
                    
        else:
            params['MODEL_NAME'] = trial.suggest_categorical('MODEL_NAME_MLP', ['vinai/bertweet-base', 'bert-base-uncased', 'xlm-roberta-base',  'microsoft/deberta-base','roberta-base'])
            bert = AutoModel.from_pretrained(params['MODEL_NAME'], return_dict=True)
        
        tokenizer = AutoTokenizer.from_pretrained(params['MODEL_NAME'])
                    
            
        if params['loss'] == 'focal' : 
            params['focal_gamma'] = trial.suggest_int('focal_gamma', 1, 5)
            

        for param in bert.parameters():
            param.requires_grad = True
        
        train_dataloader, dev_dataloader, test_dataloader, class_weights, total_steps, df_dev, df_test, semeval_climate_dataloader, semeval_abortion_dataloader = prepare_dataloader(tokenizer, self.study_name, self.data_aug_file, params['BATCH_SIZE'])
        
        model, precision, recall, f1 = train_model(bert, train_dataloader, dev_dataloader, test_dataloader,semeval_climate_dataloader, semeval_abortion_dataloader, class_weights, total_steps, params, trial)

        if f1 > F1_THERESHOLD :
            df_dev['prediction'] = model_prediction(model,dev_dataloader)
            df_dev.to_csv(f'result/df_dev_{params["study_name"]}_trial_{params["trial_num"]}_{round(f1, 4)}.csv')
            df_test['prediction'] = model_prediction(model,test_dataloader)
            df_test.to_csv(f'result/df_test_{params["study_name"]}_trial_{params["trial_num"]}_{round(f1, 4)}.csv')

            create_submission_file(df_test, path='model/best_model.pth')
        
            os.rename('model/best_model.pth', f'model/best_model_{params["study_name"]}_trial_{params["trial_num"]}_{round(f1, 4)}.pth')
        
        return f1
        
if __name__ == "__main__":
    set_seed()
    # wandb_kwargs = {"entity":"gh_mhdi", "project": "test-stance"}
    # wandbc = WeightsAndBiasesCallback(metric_name="f1", wandb_kwargs=wandb_kwargs)
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", default='tweet')
    parser.add_argument("--data_aug", default='')
    parser.add_argument("--n_trials", default=2)
    
    args = parser.parse_args()
    study_name = args.study_name 
    data_aug_file = args.data_aug
    n_trials = int(args.n_trials)
    
    if data_aug_file != '':
        optuna_study_name = f"{study_name}_{data_aug_file.replace('data/train_', '').replace('_aug.csv', '')}"
    else:
        optuna_study_name = study_name    
    
    study = optuna.create_study(#sampler=TPESampler(seed=42),
                                direction='maximize',  
                                storage="sqlite:///stance_diffrent_trial_experiment.db", 
                                study_name=optuna_study_name,
                                load_if_exists=True
                                )
    study.optimize(objective(study_name, data_aug_file), n_trials = n_trials, timeout = 120000)

    df = study.trials_dataframe()
    df.to_csv(f'{study_name}_optuna_report.csv')

    print("  Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
   
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        