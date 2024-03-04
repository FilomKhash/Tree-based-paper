################################################################################
# 
# (1) To run from command line:
#     python catboost_shap_complexity.py --json <json_name>
# 
# (2) For any given experiment in the folder results
#     the json file is located in results/<experiment_tag>/<experiment_tag>.json
# 
################################################################################
# catboost libraries path:
import os
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import json

def main( ):

    plots_folder = "pics"

    if not os.path.exists(plots_folder):
        os.mkdir(plots_folder)

    main_json = 'experiment_dim_40_depth_4_12_32_threads.json'    
    cbr_times_dfs_json = "cbr_times_dfs.json"
    model_stats_json = 'model_stats.json'

    with open(main_json, 'r') as f:
        json_dict = json.load(f)

    with open(cbr_times_dfs_json, 'r') as f:
        cbr_times_dfs_dict = json.load(f)

    with open(model_stats_json, 'r') as f:
        model_stats_dict = json.load(f)
    
    cbr_times_comp    = cbr_times_dfs_dict["cbr_times_comp"]
    cbr_times_precomp = cbr_times_dfs_dict["cbr_times_precomp"]
        
    max_depth_list = json_dict["max_depth"]
    n_trees = model_stats_dict['n_trees']   
    n_expl_catboost = json_dict["expl_size_catboost"]
   
    # plotting:
    color_cbr = "blue"
    color_cbr_precomp = "cyan"				
    figsize = (10,8)
    markersize_data = 10
    markeredgewidth = 0.5
    ticks_fontsize  = 16
    legend_fontsize = 16
    label_fontsize  = 16
    adj = 6
    depth_label = "depth=" +  r'''$\log(\mathcal{L})$'''
    
    print("\n[Plotting]...")

    # plot computation time for catboost per observation
    fig, ax = plt.subplots( figsize = figsize )
    plt.plot(max_depth_list,
                np.array(cbr_times_comp)/n_expl_catboost, color=color_cbr, 
                marker='o',
                markersize=markersize_data, 
                markeredgewidth=markeredgewidth,
                markeredgecolor = 'black', 
                label="marginal catboost-dfs (computation)")    
    
    plt.xlabel(depth_label,fontsize=label_fontsize+adj)		
    plt.ylabel("Computation time per observation", fontsize=label_fontsize+adj )
    plt.legend(fontsize = legend_fontsize+adj)	
    plt.xticks( fontsize=ticks_fontsize+adj )	
    plt.yticks( fontsize=ticks_fontsize+adj )
    plt.tight_layout()
    plt.savefig( fname = plots_folder + '/expl_catboost_comp_time.png')
    plt.close()		

    # plot  pre-computation time per tree per leaf
    
    fig, ax = plt.subplots( figsize = figsize )
    plt.plot(max_depth_list,
            np.array(cbr_times_precomp)/np.array(n_trees), color=color_cbr_precomp, 
            marker='o',
            markersize=markersize_data,
            markeredgewidth=markeredgewidth,
            markeredgecolor = 'black', 
            label="marginal catboost-dfs (precomputation)" )        
    plt.xlabel( depth_label, fontsize = label_fontsize+adj)
    plt.ylabel( "Precomputation time (per tree)" ,fontsize = legend_fontsize+adj )
    plt.legend( fontsize = label_fontsize+adj  )
    ax.set_yscale('log', base=2)
    plt.tight_layout()
    plt.savefig( fname = plots_folder + '/expl_catboost_precomp_time_renorm_log.png')

    plt.close()



if __name__=="__main__":
        main()