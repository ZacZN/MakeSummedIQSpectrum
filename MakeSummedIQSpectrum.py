from iqtools import tools
import numpy as np
import toml

# Load the config
with open("config.toml", "r") as f:
    config = toml.load(f)

# Load settings from config
file_list = config["settings"]["file_list"]
file_path = config["settings"]["file_path"]
output_location = config["settings"]["output_location"]
t_skip = float(config["settings"]["t_skip"])
t_initial = config["settings"]["t_initial"]
t_final = config["settings"]["t_final"]
experiment_name = config["settings"]["experiment_name"]


# Parse the filelist into an array
def parse_dataset(dataset):
    
    data_arr = []
    with open(dataset, "r")as f:
        for line in f:
            val = line.splitlines()
            data_arr.append(val)

    return data_arr


# Do FFT on the data in each file, and sum the results together
def data_summer(dataset, path, output_location, t_skip):

    zz = np.array([])
    for filename in dataset:
        fullpath = path + filename[0]
        print(f"Processing {filename}")
        iq = tools.get_iq_object(fullpath)

        # Calculate the number of samples to read, and the number to skip
        time_end = iq.nsamples_total/iq.fs
        t_start = t_skip/time_end # As a fraction of total samples
        samples_startpoint = int(iq.nsamples_total * t_start)
        selected_samples = iq.nsamples_total - samples_startpoint

        iq.read_samples(iq.nsamples_total - samples_startpoint, samples_startpoint)
        z = tools.get_cplx_spectrogram(
            iq.data_array,
            lframes = selected_samples,
            nframes = 1
        )
        zz += np.abs(z)

    xx, yy, _ = iq.get_power_spectrogram(
        lframes = selected_samples,
        nframes = 1
    )
    xx += iq.center

    output_name = f"{experiment_name}_{t_initial}-{t_final}_{t_skip}tskip_spectrum"

    print(f"Saving data to file {output_name}.npz in location {output_location}")

    np.savez(output_location + output_name + ".npz", xx, np.abs(np.fft.fftshift(zz, axes=1)))


def main():

    dataset = parse_dataset(file_list)

    data_summer(dataset=dataset, path=file_path, output_location=output_location, t_skip=t_skip)


if __name__ == "__main__":
    main()