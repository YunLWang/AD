import pandas as pd


#fold = 
data_final = pd.read_csv('weights/resnest50d_5fold_base/' + 'fold-0-imagenet_predtrained-efficientnetb5-submission.csv')

target_columns = ['s0']
for fold in [1,2,3,4]:
    
    path = 'weights/resnest50d_5fold_base/' + 'fold-' + str(fold)+ '-imagenet_predtrained-efficientnetb5-submission.csv'

    data_df_temp = pd.read_csv(path)
    
    data_final = data_final + data_df_temp

data_final['recording_id'] = data_df_temp['recording_id']
data_final.to_csv("./submission/baseline_efficientnetb5_5folds.csv", index=False)