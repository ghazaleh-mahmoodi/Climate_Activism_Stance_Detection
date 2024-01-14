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
          
def prepare_dataloader(tokenizer, study_name):
    TEXT_COLUMN_NAME = study_name.replace('climate_stance_experiment_on_column_', '')
    df_train = pd.read_csv(TRAIN_FILE_PATH)
    X_train = df_train[TEXT_COLUMN_NAME].tolist()
    df_train['label_LM_output'] = df_train['label'] - 1
    y_train = df_train['label_LM_output'].tolist()
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
    
    class_weights= compute_class_weight('balanced', classes = np.unique(y_train), y = np.array(y_train))
    class_weights= torch.tensor(class_weights,dtype=torch.float).to(device)

    return train_dataloader, dev_dataloader, test_dataloader, class_weights, total_steps, df_dev, df_test

class objective(object):
    def __init__(self, study_name):
        # Hold this implementation specific arguments as the fields of the class.
        self.study_name = study_name

    def __call__(self, trial):
        params= {
                    'cleassifier' : trial.suggest_categorical('cleassifier', ['cnn', 'mlp']),
                    'Dropout' : trial.suggest_float('Dropout', 0.1, 0.5, step = 0.1),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
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
        
        train_dataloader, dev_dataloader, test_dataloader, class_weights, total_steps, df_dev, df_test = prepare_dataloader(tokenizer, self.study_name)
        model, precision, recall, f1 = train_model(bert, train_dataloader, dev_dataloader, test_dataloader,class_weights, total_steps, params, trial)

        if f1 > F1_THERESHOLD :
            df_dev['prediction'] = model_prediction(model,dev_dataloader)
            df_dev.to_csv(f'result/df_dev_{params["study_name"]}_trial_{params["trial_num"]}_{round(f1, 4)}.csv')
            df_test['prediction'] = model_prediction(model,test_dataloader)
            df_test.to_csv(f'result/df_test_{params["study_name"]}_trial_{params["trial_num"]}_{round(f1, 4)}.csv')

            create_submission_file(df_test, path=f'result/submission_{params["study_name"]}_trial_{params["trial_num"]}_{round(f1, 4)}.json')
        
        return f1
        
if __name__ == "__main__":
    set_seed()
    # wandb_kwargs = {"entity":"gh_mhdi", "project": "test-stance"}
    # wandbc = WeightsAndBiasesCallback(metric_name="f1", wandb_kwargs=wandb_kwargs)
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", default='tweet')
    args = parser.parse_args()
    study_name = args.study_name
    
    study = optuna.create_study(#sampler=TPESampler(seed=42),
                                direction='maximize',  
                                storage="sqlite:///stance_experiment.db", 
                                study_name=study_name,
                                load_if_exists=True)
    study.optimize(objective(study_name), n_trials = 20, timeout = 120000)

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
        