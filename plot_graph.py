import os
import pandas as pd
import matplotlib.pyplot as plt


def save_graph():
    print("============================================================================================")
    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    # env_name = 'BipedalWalker-v2'
    env_name = 'RoboschoolWalker2d-v1'

    fig_num = 0     #### change this to prevent overwriting figures in same env_name folder
    plot_avg = True    # plot average of all runs; else plot all runs separately
    fig_width = 10
    fig_height = 6

    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson','gray', 'black']

    # make directory for saving figures
    figures_dir = "PPO_figs"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/' + env_name + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = figures_dir + '/PPO_' + env_name + '_fig_' + str(fig_num) + '.png'

    # get number of log files in directory
    log_dir = "PPO_logs" + '/' + env_name + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    for run_num in range(num_runs):

        log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)

        print("data shape : ", data.shape)

        all_runs.append(data)
        print("--------------------------------------------------------------------------------------------")

    ax = plt.gca()

    if plot_avg:
        # average all runs
        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        data_avg['reward_smooth'] = data_avg['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        data_avg['reward_var'] = data_avg['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

        data_avg.plot(kind='line', x='timestep' , y='reward_smooth',ax=ax,color=colors[0],  linewidth=linewidth_smooth, alpha=alpha_smooth)
        data_avg.plot(kind='line', x='timestep' , y='reward_var',ax=ax,color=colors[0],  linewidth=linewidth_var, alpha=alpha_var)

        # keep only reward_smooth in the legend and rename it
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0]], ["reward_avg_" + str(len(all_runs)) + "_runs"], loc=2)

    else:
        for i, run in enumerate(all_runs):
            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            run['reward_smooth_' + str(i)] = run['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
            run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

            # plot the lines
            run.plot(kind='line', x='timestep' , y='reward_smooth_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='timestep' , y='reward_var_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var)

        # keep alternate elements (reward_smooth_i) in the legend
        handles, labels = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        for i in range(len(handles)):
            if(i%2 == 0):
                new_handles.append(handles[i])
                new_labels.append(labels[i])
        ax.legend(new_handles, new_labels, loc=2)

    # ax.set_yticks(np.arange(0, 1800, 200))
    # ax.set_xticks(np.arange(0, int(4e6), int(5e5)))

    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)

    plt.title(env_name, fontsize=14)

    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)

    print("============================================================================================")
    plt.savefig(fig_save_path)
    print("figure saved at : ", fig_save_path)
    print("============================================================================================")
    
    plt.show()


if __name__ == '__main__':

    save_graph()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
