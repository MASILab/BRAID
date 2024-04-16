import textwrap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

def visualize_t_minus_n_prediction_results(results, dict_subsets, png):
    # hyperparameters
    fontsize = 9
    fontfamily = 'DejaVu Sans'
    linewidth = 1.5
    capsize = 5
    fig = plt.figure(figsize=(6.5, 8), tight_layout=True)
    gs = gridspec.GridSpec(nrows=4, ncols=3, wspace=0, hspace=0, width_ratios=[0.75, 0.25, 1],  height_ratios=[1, 1, 1, 1])
    clf_names = ['Logistic Regression','Linear SVM','Random Forest']
    dict_feat_combos = {
        'basic (chronological age + sex)': {'color': 'tab:gray', 'alpha': 0.4},
        'basic + WM age (purified)': {'color': 'tab:blue', 'alpha': 0.8},
        'basic + GM age (ours)': {'color': 'tab:red', 'alpha': 0.8},
        'basic + GM age (TSAN)': {'color': 'darkred', 'alpha': 0.4},
        'basic + GM age (DeepBrainNet)': {'color': 'indianred', 'alpha': 0.4},
        'basic + WM age (contaminated)': {'color': 'tab:purple', 'alpha': 0.4},
        'basic + WM age (purified) + GM age (ours)': {'color': 'tab:orange', 'alpha': 0.8},
        'basic + WM age (purified) + GM age (TSAN)': {'color': 'wheat', 'alpha': 0.4},
        'basic + WM age (purified) + GM age (DeepBrainNet)': {'color': 'gold', 'alpha': 0.4},
    }
    dict_abbrev = {
        'basic (chronological age + sex)': 'basic (chronological age + sex)',
        'basic + WM age (purified)': 'basic + WM age (purified)',
        'basic + GM age (ours)': 'basic + GM age (ours)',
        'basic + GM age (TSAN)': 'basic + GM age (TSAN)',
        'basic + GM age (DeepBrainNet)': 'basic + GM age (DeepBrainNet)',
        'basic + WM age (contaminated)': 'basic + WM age (contaminated)',
        'basic + WM age (purified) + GM age (ours)': 'basic + WM age (purified) + GM age (ours)',
        'basic + WM age (purified) + GM age (TSAN)': 'basic + WM age (purified) + GM age (TSAN)',
        'basic + WM age (purified) + GM age (DeepBrainNet)': 'basic + WM age (purified) + GM age (DeepBrainNet)',
    }
    timetoevent_col = [col for col in results.columns if 'time_to_' in col]
    assert len(timetoevent_col) == 1
    timetoevent_col = timetoevent_col[0]

    # Upper left block: draw the legend
    ax = fig.add_subplot(gs[:3,0])
    lines = []
    for feat_combo in dict_feat_combos.keys():
        label_txt = textwrap.fill(dict_abbrev[feat_combo], width=25)
        line = mlines.Line2D([], [], color=dict_feat_combos[feat_combo]['color'], alpha=dict_feat_combos[feat_combo]['alpha'], linewidth=linewidth, label=label_txt)
        lines.append(line)
    ax.legend(handles=lines, 
              prop={'size':fontsize, 'family':fontfamily}, 
              labelspacing=2.5, 
              frameon=True, 
              loc='upper left', 
              bbox_to_anchor=(0, 1), 
              title='Feature Combination',
              title_fontproperties={'size':fontsize, 'family':fontfamily, 'weight':'bold'})
    ax.axis('off')
    
    # Upper middle block: y axis label
    ax = fig.add_subplot(gs[:3,1])
    ax.text(0.2, 0.5, 'Area under the ROC Curve', fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', rotation='vertical', transform=ax.transAxes)
    ax.axis('off')
    
    # Upper right block: draw the AUC plots
    for i, classifier in enumerate(clf_names):
        ax = fig.add_subplot(gs[i,2])
        for feat_combo in dict_feat_combos.keys():
            data = results.loc[(results['classifier']==classifier)&(results['feature_combination']==feat_combo), ].copy()
            ax.errorbar(
                x=data[timetoevent_col].values, 
                y=data['auc_mean'].values, 
                yerr=data['auc_std'].values, 
                capsize=capsize, linewidth=linewidth,
                color=dict_feat_combos[feat_combo]['color'], 
                alpha=dict_feat_combos[feat_combo]['alpha'], 
                label=feat_combo)
        ax.vlines(x=results[timetoevent_col].unique(), ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color='black', linestyle='--', linewidth=linewidth, alpha=0.2)
        ax.text(0.02, 0.95, classifier, fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes, verticalalignment='top')
        ax.set_xlim(left=-0.25, right=4.25)
        ax.invert_xaxis()
        ax.set_ylabel('')

    # Bottom block: draw time-to-event distribution raincloud plot
    data_subsets = {'subset_id': [], timetoevent_col: []}
    for subset_id, df in dict_subsets.items():
        data_subsets['subset_id'] += [subset_id]*len(df)
        data_subsets[timetoevent_col] += df[timetoevent_col].values.tolist()
    mean_subsets = {'subset_id': [], timetoevent_col: []}
    for subset_id, df in dict_subsets.items():
        mean_subsets['subset_id'].append(subset_id)
        mean_subsets[timetoevent_col].append(df[timetoevent_col].mean())
    
    ax = fig.add_subplot(gs[3,:])
    sns.violinplot(data=data_subsets, x=timetoevent_col, y='subset_id', orient='h', split=True, inner=None, ax=ax)
    ax.scatter(x=mean_subsets[timetoevent_col], y=mean_subsets['subset_id'], c='black', marker='d', s=12, label='average')
    ax.legend(loc='upper left', fontsize=fontsize, frameon=True)
    ax.vlines(x=results[timetoevent_col].unique(), ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color='black', linestyle='--', linewidth=linewidth, alpha=0.2)
    ax.set_xlim(left=-0.25, right=8.75)
    ax.invert_xaxis()
    ax.set_yticks([])
    ax.set_ylabel(f'Subsets (N={len(dict_subsets[0].index)} each)', fontsize=fontsize, fontfamily=fontfamily)
    
    ax.set_xlabel(f"{timetoevent_col.replace('time_to_', 'Time to ')} (years)", fontsize=fontsize, fontfamily=fontfamily)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=600)
    