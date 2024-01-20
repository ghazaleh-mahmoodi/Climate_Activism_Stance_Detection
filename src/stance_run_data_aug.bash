declare -a data_cleaning_column=('tweet'
                                'tweet_remove_url' 
                                'tweet_remove_username' 
                                'tweet_remove_url_username' 
                                'tweet_remove_url_username_splite_hasghtag'
                                'tweet_remove_url_username_splite_hasghtag_lower_case'
                                'tweet_complete_cleaning')

declare -a data_augmentation_file=('data/train_EDA_aug.csv'
                                   'data/train_GPT_aug.csv')

for aug_file in "${data_augmentation_file[@]}"
    do
    for column_name in "${data_cleaning_column[@]}"
        do
            study_name="climate_stance_experiment_on_column_data_aug_${column_name}" 
            echo ${aug_file}  
            echo ${study_name}  
            # python main.py --study_name ${study_name} --data_aug ${aug_file}
        done
    done