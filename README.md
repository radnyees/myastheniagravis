# myastheniagravis

Use of BCIs to reroute the neural signals in patients with Neuromuscular diseases like Myasthenia Gravis.

My aim is to exploit the adaptive neurofeedback and reroute the neural signals to BCI devices.
This project simulates a simple Brain-Computer Interface (BCI) system using synthetic EEG data to classify motor intentions (left vs. right hand) and send control commands for muscle stimulation. 
The script generates synthetic EEG signals with noise, applies a bandpass filter (1–40 Hz), and extracts Alpha and Beta band power features using Power Spectral Density (PSD). 
A Logistic Regression model classifies the motor intention based on these features, and the corresponding control command is printed. 

Output - 1) Displays Predicted Motor Intention: Indicates whether the signal corresponds to a "Left Hand" or "Right Hand" movement. 2) Control Command: Simulates a command for muscle stimulation based on the predicted intention (e.g., "Stimulate left-hand muscles"). 3) Visualisations of raw vs. filtered EEG signals and provides insight into the frequency components of the filtered EEG signal (Power Spectral Density- PSD).

Dependencies-  `numpy`, `scipy`, `matplotlib`, and `scikit-learn`. 

Files- `experiments.py` for the main code
        `Synthetic EEG_Signals.png` for visualization of raw vs. filtered EEG signals
        `Synthetic PSD_Signals.png` for visualization of  frequency components of the filtered EEG signal (Power Spectral Density- PSD).

