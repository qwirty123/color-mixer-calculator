import cv2
import mixbox
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import numpy as np
import sys


def resize_to_fit_screen(image, max_width=1000, max_height=1000):
    """
    Resizes the image to fit within the specified dimensions while maintaining aspect ratio.
    """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale < 1:  # Only resize if the image is larger than the max dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image, scale
    return image, 1  # No resizing needed


def select_roi_and_calculate_average_color(image, prompt):
    # Resize the image to fit the screen
    resized_image, scale = resize_to_fit_screen(image)

    # Display the image and let the user select the ROI
    roi = cv2.selectROI(prompt, resized_image)
    cv2.destroyAllWindows()

    if not any(roi):
        raise ValueError("No region selected.")

    x, y, w, h = roi

    # Adjust ROI coordinates back to the original image scale if resized
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    # Extract the selected region from the original image
    selected_region = image[y:y+h, x:x+w]

    # Calculate the average color of the region
    average_color_per_channel = np.mean(selected_region, axis=(0, 1))
    average_color = tuple(map(int, average_color_per_channel[::-1]))  # Convert to RGB format

    return average_color




def mix_many_colors(colors, proportions):
    proportion_total = 0
    current_color = (0, 0, 0)
    for idx, (color, proportion) in enumerate(zip(colors, proportions, strict=True)):
        color_to_be_added = color
        color_proportion = proportion
        proportion_total += color_proportion

        proportion_of_new_color_to_previous_colors = color_proportion/proportion_total
        if np.isnan(proportion_of_new_color_to_previous_colors):
            proportion_of_new_color_to_previous_colors = 0.5
        # print((current_color, color_to_be_added, proportion_of_new_color_to_previous_colors))
        current_color = mixbox.lerp(current_color, color_to_be_added, proportion_of_new_color_to_previous_colors)

    return [x/256 for x in current_color]


def vary_color(color1, color2):
    # print(colors)
    proportions_mix = values.copy()
    # print(values)
    proportions_mix[scatterplot_x_idx] = color1
    proportions_mix[scatterplot_y_idx] = color2
    # print(proportions_mix)
    return mix_many_colors(colors, proportions_mix)

def onclick(event):
    if event.inaxes == scatter_ax:
        ix, iy = event.xdata, event.ydata
        # print(event)
        # print(ix, iy)
        sliders[scatterplot_x_idx].set_val(int(ix))
        sliders[scatterplot_y_idx].set_val(int(iy))
        update(1)
    return 1



def generate_repeated_numbers(n):
    """
    Generate a list of repeated numbers, each equal to 100/n.

    Args:
        n (int): The number of times to repeat the value (must be > 0).

    Returns:
        list: A list containing `n` numbers, each equal to 100/n.

    Raises:
        ValueError: If `n` is less than or equal to 0.
    """
    if n <= 0:
        raise ValueError("The input must be a positive integer.")

    value = 100 / n
    return [value] * n


# Update function for sliders and scatter plot
def update(val):
    # Get slider values and normalize them to sum to 100
    global values
    global scatterplot_x_idx
    global scatterplot_y_idx
    values = np.array([slider.val for slider in sliders])
    normalized_values = 100 * values / np.sum(values)

    # Update the pie chart
    ax.clear()
    wedges, texts, autotexts = ax.pie(
        normalized_values, labels=labels, autopct='%1.1f%%', startangle=90, colors=pie_colors
    )
    for wedge, autotext in zip(wedges, autotexts):
        autotext.set_color('white')
        autotext.set_weight('bold')

    combined_color = mix_many_colors(colors, values)
    ax.text(0, 0, [int(x*256) for x in combined_color] , ha='center', va='center', fontsize=16, backgroundcolor=combined_color)


    # update the scatter plot
    selected_axes = [i for i, checkbox in enumerate(checkboxes) if checkbox.get_status()[0]]
    scatter_ax.clear()  # Clear scatter axes before updating
    if len(selected_axes) == 2:

        scatterplot_x_idx, scatterplot_y_idx = selected_axes[:2]
        # disable slider
        # sliders[scatterplot_x_idx].active = False
        # sliders[scatterplot_x_idx].ax.set_facecolor('gray')
        # sliders[scatterplot_y_idx].active = False
        # sliders[scatterplot_y_idx].ax.set_facecolor('gray')

        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros((len(y), len(x), 3))  # Initialize a 3D array for RGB colors
        for i in range(len(y)):
            for j in range(len(x)):
                Z[i, j] = vary_color(X[i, j], Y[i, j])

        scatter_ax.set_xlabel(labels[scatterplot_x_idx])
        scatter_ax.set_ylabel(labels[scatterplot_y_idx])
        scatter_ax.xaxis.label.set_color([x/256 for x in colors[scatterplot_x_idx]])
        scatter_ax.yaxis.label.set_color([y/256 for y in colors[scatterplot_y_idx]])

        scatter_ax.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')

    fig.canvas.draw_idle()



if __name__ == '__main__':
    image = cv2.imread(sys.argv[-1])
    if image is None:
        raise FileNotFoundError("Error: Unable to load image. Check the file path.")


    num_colors = int(input("How many colors are there? "))
    print("Please choose each color region")

    colors = [select_roi_and_calculate_average_color(image, f"Select area for color {i+1}") for i in range(num_colors)]

    # Initial data
    values = generate_repeated_numbers(len(colors))
    labels = [str(x) for x in colors]
    pie_colors = [[x/256 for x in y] for y in colors]

    # Create a slider and checkbox for each slice
    num_slices = len(values)
    sliders = []
    slider_axes = []
    checkbox_axes = []
    checkboxes = []

    scatterplot_x_idx = 0
    scatterplot_y_idx = 0

    # Create the figure with two subplots
    fig, (ax, scatter_ax) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.1, bottom=0.4)  # Leave space for sliders and checkboxes

    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%', startangle=90, colors=pie_colors
    )

    # Adjust colors and appearance
    for wedge, autotext in zip(wedges, autotexts):
        autotext.set_color('white')
        autotext.set_weight('bold')


    for i in range(num_slices):
        ax_slider = plt.axes([0.1, 0.3 - i * 0.07, 0.65, 0.03])  # Adjust slider position
        slider = Slider(
            ax_slider, label=f'{labels[i]}', valmin=0, valmax=100, valinit=values[i]
        )
        sliders.append(slider)
        slider_axes.append(ax_slider)

        ax_checkbox = plt.axes([0.8, 0.3 - i * 0.07, 0.15, 0.03])  # Adjust checkbox position
        checkbox = CheckButtons(ax_checkbox, ["compare"], [False])
        checkboxes.append(checkbox)
        checkbox_axes.append(ax_checkbox)

    # Add event listeners
    for slider in sliders:
        slider.on_changed(update)

    for checkbox in checkboxes:
        checkbox.on_clicked(update)

    fig.canvas.mpl_connect('button_press_event', onclick)
    update(1)
    plt.show()
