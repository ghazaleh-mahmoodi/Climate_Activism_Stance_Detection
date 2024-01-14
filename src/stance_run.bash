declare -a data_cleaning_column=('tweet'
                                'tweet_remove_url' 
                                'tweet_remove_username' 
                                'tweet_remove_url_username' 
                                'tweet_remove_url_username_splite_hasghtag'
                                'tweet_remove_url_username_splite_hasghtag_lower_case'
                                'tweet_complete_cleaning')

for column_name in "${data_cleaning_column[@]}"
    do
        study_name="climate_stance_experiment_on_column_${column_name}" 
        echo ${study_name}  
        python main.py --study_name ${study_name}
    done