import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_time_varying_mean(time_steps, mean_values):
    plt.figure(figsize=(10, 6))
    plt.title('Time-Varying Mean Across Channels')
    plt.xlabel('Time Steps')
    plt.ylabel('Mean Value')

    plt.plot(time_steps, mean_values, linestyle='-', color='purple', label='Mean')

    plt.xlim(-50, 3000)
    plt.ylim(3.3, 4.13)  # Adjust as needed

    plt.legend(loc='upper left')
    plt.show()


def save_time_varying_mean_to_csv(time_steps, mean_values, output_file):
    # Save time steps and mean values to a CSV file
    data_to_save = np.column_stack((time_steps, mean_values))
    np.savetxt(output_file, data_to_save, delimiter=',')


def calculate_time_varying_mean(data):
    # Calculate time-varying mean across channels
    time_steps = np.arange(data.shape[1])
    mean_values = np.mean(data, axis=0)

    return time_steps, mean_values


def save_and_plot_mean(file_name, channel_indices):
    file_path = f'../data/udds/{file_name}'
    df = pd.read_csv(file_path, skiprows=1)
    data_np = df.values.T
    selected_channels_data = data_np
    time_steps, mean_values = calculate_time_varying_mean(selected_channels_data)

    # Save time-varying mean to CSV
    output_file_path = f'../result/meanFig/csv/{file_name}'
    save_time_varying_mean_to_csv(time_steps, mean_values, output_file_path)

    # Plot time-varying mean
    plot_time_varying_mean(time_steps, mean_values)


# Example usage:
file_name_cor = 'UDDS_Cor_0629_2.csv'
file_name_isc = 'UDDS_Isc_5ohm_0629_2.csv'
file_name_noi = 'UDDS_noi_0706_3.csv'
file_name_sti = "UDDS_sti_0706_3.csv"
channel_indices = [0, 1, 2]  # Specify the channel indices to calculate the mean

# Call the new function to save and plot mean
save_and_plot_mean(file_name_cor, channel_indices)
save_and_plot_mean(file_name_isc, channel_indices)
save_and_plot_mean(file_name_noi, channel_indices)
save_and_plot_mean(file_name_sti, channel_indices)
