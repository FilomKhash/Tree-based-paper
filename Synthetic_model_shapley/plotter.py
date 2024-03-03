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
    err_cbr_json = "err_cbr.json"
    cbr_times_dfs_json = "cbr_times_dfs.json"
    cbr_times_native_json = "cbr_times_native.json"
    model_stats_json = 'model_stats.json'

    with open(main_json, 'r') as f:
        json_dict = json.load(f)

    with open(err_cbr_json, 'r') as f:
        err_cbr_dict = json.load(f)

    with open(cbr_times_dfs_json, 'r') as f:
        cbr_times_dfs_dict = json.load(f)

    with open(cbr_times_native_json, 'r') as f:
        cbr_times_native_dict = json.load(f)

    with open(model_stats_json, 'r') as f:
        model_stats_dict = json.load(f)

    err_cbr_test  = err_cbr_dict["err_cbr_test"]
    err_cbr_train = err_cbr_dict["err_cbr_train"]

    cbr_times_comp    = cbr_times_dfs_dict["cbr_times_comp"]
    cbr_times_precomp = cbr_times_dfs_dict["cbr_times_precomp"]
    
    cbr_tshap_times_comp   = cbr_times_native_dict["cbr_tshap_times_comp"]
    cbr_exact_times_precomp = cbr_times_native_dict["cbr_exact_times_precomp"]
    
    max_depth_list = json_dict["max_depth"]

    n_trees = model_stats_dict['n_trees']

    ave_num_leaves = model_stats_dict['ave_num_leaves']

    # find out the name and value of the fixed parameter 
    var_param="max_depth"
    n_expl_catboost = json_dict["expl_size_catboost"]
    n_expl_catboost_regular = json_dict["expl_size_catboost_regular"]    
    n_ave_list  = json_dict["ave_size"]
    sample_train_size = json_dict["sample_train_size"]

    # plotting:

    alpha = 0.5	
    color_cbr = "blue"
    color_cbr_precomp = "cyan"				
    figsize = (10,8)
    markersize_data = 10
    markeredgewidth = 0.5
    ticks_fontsize  = 16
    legend_fontsize = 16
    label_fontsize  = 16
    depth_label = "depth=" + r'''$\log(\mathcal{L})$'''
    
    print("\n[Plotting]...")

    # plotting errors:
    fig, ax = plt.subplots( figsize = figsize   )		
    plt.plot( max_depth_list, np.array( [ err_cbr_test[k]["rel_error"] for k in range(len(err_cbr_test))]), 					
                color = color_cbr,
                marker="o", markeredgewidth=0.5, 
                markeredgecolor="black",
                markersize=markersize_data,
                label="relative test error")
    plt.plot( max_depth_list, np.array( [ err_cbr_train[k]["rel_error"] for k in range(len(err_cbr_train))]), 					
                color = "green",
                marker="o", markeredgewidth=0.5, 
                markeredgecolor="black",
                markersize=markersize_data,
                label="relative train error")
    # plt.title("relative ML error, dim={0}, {1}={2}".format(dim,fixed_param,fixed_param_value))
    adj = 6
    plt.xticks( fontsize=ticks_fontsize+adj )	
    plt.yticks( fontsize=ticks_fontsize+adj )
    plt.xlabel(depth_label,fontsize=label_fontsize+adj)
    plt.ylabel("relative $L^2$-error", fontsize=label_fontsize+adj )
    plt.legend(fontsize = legend_fontsize+adj)		
    plt.tight_layout()
    plt.savefig( fname = plots_folder+'/ml_errors.png' )		
    plt.close()


    # plot computation time for catboost per observation
    fig, ax = plt.subplots( figsize = figsize )
    plt.plot(max_depth_list,
                np.array(cbr_times_comp)/n_expl_catboost, color=color_cbr, 
                marker='o',
                markersize=markersize_data, 
                markeredgewidth=markeredgewidth,
                markeredgecolor = 'black', 
                label="marginal catboost-dfs (computation)")
    # plt.title("Computation of explanation per observation, dim={0}, {1}={2}".format(dim,fixed_param,fixed_param_value))
    adj = 6
    plt.xlabel(depth_label,fontsize=label_fontsize+adj)		
    plt.ylabel("Computation time per observation", fontsize=label_fontsize+adj )
    plt.legend(fontsize = legend_fontsize+adj)	
    plt.xticks( fontsize=ticks_fontsize+adj )	
    plt.yticks( fontsize=ticks_fontsize+adj )
    plt.tight_layout()
    plt.savefig( fname = plots_folder + '/expl_catboost_comp_time.png')
    plt.close()		

    # plot pre-computation time for catboost-dfs
    for t in range(2):
        fig, ax = plt.subplots( figsize = figsize )
        plt.plot(max_depth_list,
                np.array(cbr_times_precomp), color=color_cbr_precomp, 				 
                marker='o',
                markersize=markersize_data, 
                markeredgewidth=markeredgewidth,
                markeredgecolor = 'black', 
                label="marginal catboost-dfs (precomputation)")
        # plt.title("Precomputation time, dim={0}, {1}={2}".format(dim,fixed_param,fixed_param_value))
        adj = 6
        plt.ylabel("Precomputation time",fontsize=label_fontsize+adj)
        plt.xlabel(depth_label,fontsize=label_fontsize+adj)
        plt.legend(fontsize = legend_fontsize+adj)
                    
        if t==1:
            plt.tight_layout()
            plt.savefig( fname = plots_folder + '/expl_catboost_precomp_time.png')
        else:
            ax.set_yscale('log', base=2)
            plt.tight_layout()
            plt.savefig( fname = plots_folder + '/expl_catboost_precomp_time_log.png')
        plt.close()		

    # plot  pre-computation time per tree per leaf
    for t in range(2):			
        fig, ax = plt.subplots( figsize = figsize )
        plt.plot(np.array(max_depth_list),
                np.array(cbr_times_precomp)/np.array(n_trees), color=color_cbr_precomp, 
                marker='o',
                markersize=markersize_data,
                markeredgewidth=markeredgewidth,
                markeredgecolor = 'black', 
                label="marginal catboost-dfs (precomputation)" )
        # plt.title("Precomputation time (per tree), dim={0}, {1}={2}".format(dim,fixed_param,fixed_param_value))
        adj = 6
        plt.xlabel( depth_label, fontsize = label_fontsize+adj)
        plt.ylabel( "Precomputation time (per tree)" ,fontsize = legend_fontsize+adj )
        plt.legend( fontsize = label_fontsize+adj  )
        if t==1:
            plt.tight_layout()
            plt.savefig( fname = plots_folder + '/expl_catboost_precomp_time_renorm.png')
        else:
            ax.set_yscale('log', base=2)
            plt.tight_layout()
            plt.savefig( fname = plots_folder + '/expl_catboost_precomp_time_renorm_log.png')

        plt.close()

    # plot computation time for all observations:
    fig, ax = plt.subplots( figsize = figsize )
    plt.plot(max_depth_list,
                np.array(cbr_times_comp)/n_expl_catboost, 
                color=color_cbr, 
                marker='o',
                markersize=markersize_data, 
                markeredgewidth=markeredgewidth,
                markeredgecolor = 'black', label="marginal catboost-dfs (computation), $|D|$={0}".format(sample_train_size))
    for k in range(len(cbr_tshap_times_comp)):
        plt.plot(max_depth_list,
                    np.array(cbr_tshap_times_comp[str(k)])/n_expl_catboost_regular, 
                    marker='o',
                markersize=markersize_data,
                markeredgewidth=markeredgewidth,
                markeredgecolor = 'black', 
                label="marginal catboost-regular-native, $|D^*|$={0}".format(n_ave_list[k]))
    # plt.title("Computation times (per observation), dim={0}, {1}={2}".format(dim,fixed_param,fixed_param_value))		
    plt.xlabel(depth_label, fontsize=label_fontsize)
    plt.ylabel("Computation times (per observation)",fontsize=label_fontsize)
    plt.xticks( fontsize=ticks_fontsize )	
    plt.yticks( fontsize=ticks_fontsize )
    plt.legend( fontsize = label_fontsize )
    plt.tight_layout()
    plt.savefig( fname = plots_folder + '/expl_comp_time_all.png')
    plt.close()		


    # plot computation time for all observations vs precomputation:
    fig, ax = plt.subplots( figsize = figsize )
    plt.plot(np.array(max_depth_list),
                np.array(cbr_times_precomp)/np.array(ave_num_leaves), 
                color=color_cbr_precomp, 
                marker='o',
                markersize=markersize_data, 
                markeredgewidth=markeredgewidth,
                markeredgecolor = 'black', label="marginal catboost-dfs (precomputation per leaf), $|D|$={0}".format(sample_train_size))
    for k in range(len(cbr_tshap_times_comp)):
        plt.plot(max_depth_list,
                    np.array(cbr_tshap_times_comp[str(k)])/n_expl_catboost_regular, 
                    marker='o',
                markersize=markersize_data,
                markeredgewidth=markeredgewidth,
                markeredgecolor = 'black', label="marginal catboost-regular-native, $|D^*|$={0} (per observ)".format(n_ave_list[k]))
    # plt.title("Precomputation per leaf vs computation per observation, dim={0}, {1}={2}".format(dim,fixed_param,fixed_param_value))
    plt.ylabel("Precomputation time (per leaf) vs computation", fontsize=label_fontsize )
    plt.xlabel(depth_label,fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks( fontsize=ticks_fontsize )	
    plt.yticks( fontsize=ticks_fontsize )
    plt.tight_layout()
    plt.savefig( fname = plots_folder + '/expl_comp_time_vs_precomp.png')
    plt.close()		

    # complexity in terms of power of leaves:
    for t in range(2):

        fig, ax = plt.subplots( figsize = figsize )

        plt.plot( np.array(max_depth_list), np.array(cbr_exact_times_precomp)/np.array(n_trees), 
                color="lime", 
                label="marginal catboost-exact-native", 
                marker="o", 
                markeredgewidth = 0.25,
                markersize=markersize_data, 
                markeredgecolor="black",
                alpha=1)

        plt.plot(np.array(max_depth_list),np.array(cbr_times_precomp)/np.array(n_trees), 
                color="blue", 
                label="marginal catboost-dfs", 
                marker="o", 
                markeredgewidth=0.25, 
                markersize=markersize_data,
                markeredgecolor="black",
                alpha=1)

        plt.legend( fontsize = legend_fontsize )
        plt.xlabel(depth_label,fontsize=label_fontsize)
        plt.ylabel("Time complexity (per tree)", fontsize=label_fontsize)
        plt.xticks( fontsize=ticks_fontsize )	
        plt.yticks( fontsize=ticks_fontsize )

        # plt.title("Time complexity per tree (all leaves precomputation )")
        if t==1:
            plt.tight_layout()
            plt.savefig( fname = plots_folder + '/comparison_native.png')
        else:
            ax.set_yscale('log', base=2)
            plt.tight_layout()
            plt.savefig( fname = plots_folder + '/comparison_native_log.png')

        plt.close()		


if __name__=="__main__":
        main()