import os
import glob
import pandas as pd
import json

project_path = "../"
configs_path = os.path.join(project_path,"configs")
results_path = os.path.join(project_path,"results")
logs_path = os.path.join(project_path, "logs")

log_files = glob.glob(os.path.join(logs_path,"*.log"))
config_files = glob.glob(os.path.join(configs_path,"**/*.json"))
results_files = glob.glob(os.path.join(results_path, "overall_*.csv"))

exp_dict = {}

for result_file in results_files:
    experiment_num = int(result_file.split("/")[-1].split("\\")[-1][8:-4])
    results = pd.read_csv(result_file)
    print(f"Exp Num : {experiment_num}")
    exp_dict[experiment_num] = [results]
    
# print(config_files)    
for config_file in config_files:
    try:
        experiment_num = int(config_file.split("/")[-1].split("\\")[-1][7:-5])
        print(f"Exp Num : {experiment_num}")
        
        with open(config_file, 'r') as file:
            config = json.load(file)
            experiment_num = int(config["experiment_num"])
            try:
                exp_dict[experiment_num].append(list(config["augmentations"]))
            except:
                pass
    except:
        pass
    
  
  

# for key in exp_dict.keys():
#     print(key, len(exp_dict[key]))

# input("Press Enter to continue...")

exp_df = pd.DataFrame()  
for experiment_num in exp_dict.keys(): 
    results = exp_dict[experiment_num][0]
    augmentations = exp_dict[experiment_num][1]
    
    results = results.T.reset_index(drop=True).drop(0)
    results["Augmentation"] = " + ".join(augmentations)
    results["Experiment_Num"] = experiment_num
    results = results[list(results.columns)[::-1]]
    
    exp_df = pd.concat([exp_df, results], axis=0)
    
exp_df.columns = ["Experiment_Num", "Augmentations", "MF1", "Acc"]
exp_df = exp_df.sort_values(by=["Experiment_Num"], ascending=[False])
file_name = "Summary_v3.csv"
exp_df.to_csv(file_name, index=False)
print(f"Summarised results stored in {file_name}")