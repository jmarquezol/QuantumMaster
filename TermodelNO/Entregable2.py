import os
import numpy as np
import matplotlib.pyplot as plt

# Define the folder path
folder_path = "ARCHIVOS PARA ENTREGABLE 2(OPCIÓN 1)-20250401"

# Initialize dictionaries to store data
flux_data = {}
temp_data = {}

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.startswith("Flux") and filename.endswith(".txt"):
        # Extract position from the filename
        position = float(filename.replace("Flux", "").replace(".txt", ""))
        # Load data from the file, skipping metadata lines (lines starting with '%')
        data = np.loadtxt(
            os.path.join(folder_path, filename),
            comments="%",  # Skip lines starting with '%'
        )
        # Store time and flux values
        flux_data[position] = data

    elif filename.startswith("Temp") and filename.endswith(".txt"):
        # Extract position from the filename
        position = float(filename.replace("Temp", "").replace(".txt", ""))
        # Load data from the file, skipping metadata lines (lines starting with '%')
        data = np.loadtxt(
            os.path.join(folder_path, filename),
            comments="%",  # Skip lines starting with '%'
        )
        # Store time and temperature values
        temp_data[position] = data

for position, data in temp_data.items():
    print(f"Position: {position}")
    print(data[:5])  # Print the first 5 rows of data for each position
    print()

# Plot flux as a function of time for different positions
plt.figure(figsize=(10, 6))
for position, data in sorted(flux_data.items()):
    time = data[:, 0]
    flux = data[:, 1]
    plt.plot(time, flux, label=f"x={position}")
plt.title("Flux vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Flux (W/m^2)")
plt.legend()
plt.grid()
plt.show()