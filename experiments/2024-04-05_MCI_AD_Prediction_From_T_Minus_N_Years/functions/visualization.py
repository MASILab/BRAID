import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def visualize_t_minus_n_prediction_results(results, dict_subsets, png):
    # hyperparameters
    fontsize = 9
    fontfamily = 'DejaVu Sans'
    fig = plt.figure(figsize=(6.5, 8), tight_layout=True)
    gs = gridspec.GridSpec(nrows=5, ncols=1, wspace=0, hspace=0, height_ratios=[1, 1, 1, 2, 0.05])
    clf_names = ['Logistic Regression','Linear SVM','Random Forest']
    dict_feat_combos = {
        'basic (chronological age + sex)': {'color': 'tab:gray', 'alpha': 0.2},
        'basic + WM age (purified)': {'color': 'tab:blue', 'alpha': 0.8},
        'basic + GM age (ours)': {'color': 'tab:red', 'alpha': 0.8},
        'basic + GM age (TSAN)': {'color': 'darkred', 'alpha': 0.2},
        'basic + GM age (DeepBrainNet)': {'color': 'indianred', 'alpha': 0.2},
        'basic + WM age (contaminated)': {'color': 'tab:purple', 'alpha': 0.2},
        'basic + WM age (purified) + GM age (ours)': {'color': 'tab:orange', 'alpha': 0.8},
        'basic + WM age (purified) + GM age (TSAN)': {'color': 'wheat', 'alpha': 0.2},
        'basic + WM age (purified) + GM age (DeepBrainNet)': {'color': 'gold', 'alpha': 0.2},
    }
    
    # draw AUCs
    for i, classifier in enumerate(clf_names):
        ax = fig.add_subplot(gs[i])
        for feat_combo in dict_feat_combos.keys():
            data = results.loc[(results['classifier']==classifier)&(results['feature_combination']==feat_combo), ].copy()
            ax.errorbar(
                x=data['time_to_event_mean'].values, 
                y=data['auc_mean'].values, 
                yerr=data['auc_std'].values, 
                color=dict_feat_combos[feat_combo]['color'], 
                alpha=dict_feat_combos[feat_combo]['alpha'], 
                label=feat_combo)
        ax.text(0.05, 0.95, classifier, fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes, verticalalignment='top')
        ax.invert_xaxis()
        ax.set_ylabel('AUC', fontsize=fontsize, fontfamily=fontfamily)

    # plot time to event raincloud plot
    
    ax = fig.add_subplot(gs[3])
    
    
    
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=600)
    