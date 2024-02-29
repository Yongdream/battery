import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_time_varying_correlations(time_steps, channel_correlations):
    plt.figure(figsize=(10, 6))
    plt.title('Time-Varying Channel Correlations')
    plt.xlabel('Time Steps')
    plt.ylabel('Correlation Coefficient')
    plt.plot(time_steps, channel_correlations, linestyle='-', color='purple', label='Correlation')

    plt.xlim(-50, 3000)
    plt.ylim(0.88, 1.02)

    plt.legend(loc='upper left')
    plt.show()


def save_time_varying_correlations_to_csv(time_steps, channel_correlations1, channel_correlations2, channel_correlations3, output_file):
    # Save time steps and correlation coefficients to a CSV file
    data_to_save = np.column_stack((time_steps,  channel_correlations1, channel_correlations2, channel_correlations3))
    np.savetxt(output_file, data_to_save, delimiter=',')


def calculate_time_varying_correlations(data, channel1, channel2):
    # Calculate time-varying correlations between specified channels
    time_steps = np.arange(data.shape[1] - 1)
    channel_correlations = []


    for t in range(1, data.shape[1]):
        if t == 1:
            correlation = 1  # Set correlation to 1 at t=0
        else:
            correlation = np.corrcoef(data[channel1, :t], data[channel2, :t])[0, 1]

        channel_correlations.append(correlation)

    return time_steps, np.array(channel_correlations)


def save_and_plot_correlations(file_name, channelref_index,channel1_index, channel2_index, channel3_index):
    file_path = f'../data/test/{file_name}'
    df = pd.read_csv(file_path, skiprows=1)
    data_np = df.values.T
    time_steps, channel_correlations1 = calculate_time_varying_correlations(data_np, channelref_index, channel1_index)
    time_steps, channel_correlations2 = calculate_time_varying_correlations(data_np, channelref_index, channel2_index)
    time_steps, channel_correlations3 = calculate_time_varying_correlations(data_np, channelref_index, channel3_index)

    # Save time-varying correlations to CSV
    output_file_path = f'../result/corFig/csv/{file_name}'
    save_time_varying_correlations_to_csv(time_steps, channel_correlations1, channel_correlations2, channel_correlations3,output_file_path)

    # Plot time-varying correlations
    # plot_time_varying_correlations(time_steps, channel_correlations)


# Example usage:
# file_name_cor = 'UDDS_Cor_0629_2.csv'
# file_name_isc = 'UDDS_Isc_10ohm_0630_2.csv'
# file_name_is2 = 'UDDS_Isc_5ohm_0629_2.csv'
#
# file_name_noi = 'UDDS_noi_0706_3.csv'
# file_name_sti = "UDDS_sti_0706_3.csv"

file_name_cor = 'uddscorisc05.csv'
# file_name_isc = 'UDDS_Isc_10ohm_0630_2.csv'
# file_name_nor = 'UDDS_Isc_10ohm_0630_2.csv'
channel1_index = 1
channel2_index = 2
channel3_index = 3
channel4_index = 4

# Call the new function to save and plot correlations
# save_and_plot_correlations(file_name_cor, channel1_index, channel2_index)
save_and_plot_correlations(file_name_cor, channel4_index, channel1_index, channel2_index, channel3_index)

#
# save_and_plot_correlations(file_name_isc, channel1_index, channel2_index)
# save_and_plot_correlations(file_name_is2, channel1_index, channel2_index)
# save_and_plot_correlations(file_name_isc, channel1_index, channel3_index)

# save_and_plot_correlations(file_name_noi, channel1_index, channel3_index)
# save_and_plot_correlations(file_name_sti, channel1_index, channel3_index)
