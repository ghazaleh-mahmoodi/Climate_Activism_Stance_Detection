declare -a data_cleaning_column=('tweet'
                                'tweet_remove_url' 
                                'tweet_remove_username' 
                                'tweet_remove_url_username' 
                                'tweet_remove_url_username_splite_hasghtag'
                                'tweet_remove_url_username_splite_hasghtag_lower_case'
                                'tweet_complete_cleaning')

declare -a trials=(10 20 40)

for trial in "${trials[@]}"
    do
    for column_name in "${data_cleaning_column[@]}"
        do
            study_name="climate_stance_experiment_on_column_${column_name}_${trial}" 
            echo ${study_name}  
            python main.py --study_name ${study_name} --n_trials ${trial}
        done
    done