import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_time_varying_variance(time_steps, variance_values):
    plt.figure(figsize=(10, 6))
    plt.title('Time-Varying Variance Across Channels')
    plt.xlabel('Time Steps')
    plt.ylabel('Variance Value')

    plt.plot(time_steps, variance_values, linestyle='-', color='orange', label='Variance')

    plt.xlim(-50, 3000)
    plt.ylim(-0.0005, 0.0085)  # Adjust as needed

    plt.legend(loc='upper left')
    plt.show()


def save_time_varying_variance_to_csv(time_steps, variance_values, output_file):
    # Save time steps and variance values to a CSV file
    data_to_save = np.column_stack((time_steps, variance_values))
    np.savetxt(output_file, data_to_save, delimiter=',')


def calculate_time_varying_variance(data):
    # Calculate time-varying variance across channels
    time_steps = np.arange(data.shape[1])
    variance_values = np.var(data, axis=0)

    return time_steps, variance_values


def save_and_plot_variance(file_name, channel_indices):
    file_path = f'../data/udds/{file_name}'
    df = pd.read_csv(file_path, skiprows=1)
    data_np = df.values.T
    selected_channels_data = data_np
    time_steps, variance_values = calculate_time_varying_variance(selected_channels_data)

    # Save time-varying variance to CSV
    output_file_path = f'../result/varFig/csv/{file_name}'
    save_time_varying_variance_to_csv(time_steps, variance_values, output_file_path)

    # Plot time-varying variance
    plot_time_varying_variance(time_steps, variance_values)


# Example usage:
file_name_cor = 'UDDS_Cor_0629_2.csv'
file_name_isc = 'UDDS_Isc_5ohm_0629_2.csv'
file_name_noi = 'UDDS_noi_0706_3.csv'
file_name_sti = "UDDS_sti_0706_3.csv"
file_name_nor = "UDDS_Nor_0712.csv"
channel_indices = [0, 1, 2]  # Specify the channel indices to calculate the variance

# Call the new function to save and plot variance
save_and_plot_variance(file_name_cor, channel_indices)
save_and_plot_variance(file_name_isc, channel_indices)
save_and_plot_variance(file_name_noi, channel_indices)
save_and_plot_variance(file_name_sti, channel_indices)
save_and_plot_variance(file_name_nor, channel_indices)
