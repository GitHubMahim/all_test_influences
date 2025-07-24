# all_test_influences
Visualizes the influence of training data over the whole training data.

Interactive Influence Visualization Tool
plot_all_influences.py provides an interactive visualization for exploring the influence of training data points on test predictions in a continuous field reconstruction task (e.g., climate or satellite data).
Features
Two-panel interactive plot:
Left: Shows the reference field with a highlighted test location.
Right: Overlays the top N% most influential training points for the selected test location, colored by their influence value (purple-green gradient).
Sliders:
Longitude and Latitude sliders: Select any test coordinate to analyze.
Top % Influential Points slider: Adjusts the percentage of most influential training points shown (10% to 100%).
Colorbars: Show the value range for the field and for the influence scores.
Usage
Prepare your data:
test_coords.npy: Array of normalized test coordinates, shape (N_test, 2).
train_coords.npy: Array of normalized training coordinates, shape (N_train, 2).
yrefs.npy: 2D array of the reference field, shape (H, W).
influences_all_test.npy: Influence values, shape (N_test, N_train) or list of arrays.
Run the script:
Apply to plot_all_inf...
Run
Interact:
Use the sliders to select a test coordinate and the percentage of influential points to display.
Click on the left plot to select a test location directly.
Requirements
Python 3.x
numpy
matplotlib
torch (for tensor handling)
Customization
You can adapt the script to use different test indices, influence files, or field images by modifying the data loading section at the top of the script.
