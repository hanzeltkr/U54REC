import glob, re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main() :
    resnet_img = glob.glob("classifier_939715_*.out")
    resnet_header = glob.glob("classifier_939705_*.out")
    efficient_img = glob.glob("classifier_941328_*.out")
    efficient_header = glob.glob("classifier_948812_*.out")
    r_f1_img, r_auc_img = assign_data(resnet_img)
    r_f1_header, r_auc_header = assign_data(resnet_header)
    e_f1_img, e_auc_img = assign_data(efficient_img)
    e_f1_header, e_auc_header = assign_data(efficient_header)


    f1_rows = []
    data_setup(f1_rows, r_f1_img, 'Resnet50', False)
    data_setup(f1_rows, r_f1_header, 'Resnet50', True)
    data_setup(f1_rows, e_f1_img, 'EfficientNetB0', False)
    data_setup(f1_rows, e_f1_header, 'EfficientNetB0', True)
    f1_df = pd.DataFrame(f1_rows)

    sns.stripplot(data=f1_df, x='Model', y='F1', hue='group', dodge=True, alpha=0.7)
    sns.pointplot(data=f1_df, x='Model', y='F1', hue='group', dodge=0.4,markers='_', linestyle='none', errorbar=None, legend=False)

    plt.ylabel('F1 Score')
    plt.title('F1 Score by Models')
    plt.savefig('f1_scores.png')
    plt.close()

    auc_rows = []
    data_setup(auc_rows, r_auc_img, 'Resnet50', False)
    data_setup(auc_rows, r_auc_header, 'Resnet50', True)
    data_setup(auc_rows, e_auc_img, 'EfficientNetB0', False)
    data_setup(auc_rows, e_auc_header, 'EfficientNetB0', True)
    auc_df = pd.DataFrame(auc_rows)
    
    sns.stripplot(data=auc_df, x='Model', y='F1', hue='group', dodge=True, alpha=0.7)
    sns.pointplot(data=auc_df, x='Model', y='F1', hue='group', dodge=0.4,markers='_', linestyle='none', errorbar=None, legend=False)
    
    plt.ylabel('AUC')
    plt.title('AUC by Models')
    plt.savefig('auc.png')
    plt.close()

def assign_data(files) :
    f1_scores = []
    auc_scores = []
    for file in files :
        f = open(file)
        result = f.read()
        F1score = find(result, "F1 Score:")
        AUC = find(result, "auc:")
        print(F1score, AUC)
        f1_scores.append(F1score)
        auc_scores.append(AUC)
        f.close()
    return f1_scores, auc_scores

def find(string, name) :
    pattern = rf"{name} ([\d.]+)"
    all_matches = re.findall(pattern, string)
    if all_matches:
        score = float(all_matches[-1])
    return score

def data_setup(rows, data, model=None, have_header=False) :
    if have_header :
        for val in data:
            rows.append({'Model': f'{model}', 'group': 'Images + Headers', 'F1': val})
    else :
        for val in data:
            rows.append({'Model': f'{model}', 'group': 'Images', 'F1': val})
    return rows

if __name__ == "__main__":
    main()