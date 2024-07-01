import os
import sys
import numpy as np
import pygame
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import pygame_gui

# Suppress Pygame's welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Function to read array from file
def read_array_from_file(input_file):
    try:
        return np.load(input_file, allow_pickle=False)
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        return None
    except Exception as e:
        print(f"Couldn't load the array from {input_file}: {e}")
        return None

# Function to save array to file
def save_array_to_file(arr, output_file):
    try:
        np.save(output_file, arr, allow_pickle=False)
        return True, None
    except Exception as e:
        return False, f"Couldn't save the array to {output_file}: {e}"

# Function to save array as image
def save_array_as_image(arr, output_file, cmap="coolwarm", dpi=200):
    try:
        cmap_func = plt.get_cmap(cmap)
        img = Image.fromarray(np.uint8(cmap_func(arr) * 255))
        img.save(output_file, dpi=(dpi, dpi))
        return True
    except Exception as e:
        print(f"Couldn't save the image: {e}")
        return False

# Function to update a pixel
def update_pixel(x, y, array, mouse_button, brush_size):
    for i in range(max(0, y - brush_size // 2), min(array.shape[0], y + brush_size // 2 + 1)):
        for j in range(max(0, x - brush_size // 2), min(array.shape[1], x + brush_size // 2 + 1)):
            if mouse_button == 1:
                array[i, j] = 1
            elif mouse_button == 3:
                array[i, j] = 0
    return array

# Function to render the array
def render(array, screen, drawing_rect):
    fig = plt.figure(figsize=(drawing_rect.width / 100, drawing_rect.height / 100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(array, vmin=0, vmax=1, cmap='coolwarm', origin='upper')
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf.shape = (fig.canvas.get_width_height()[::-1] + (4,))
    img = pygame.image.frombuffer(buf, fig.canvas.get_width_height(), "RGBA")

    scaled_img = pygame.transform.scale(img, (drawing_rect.width, drawing_rect.height))
    screen.blit(scaled_img, drawing_rect.topleft)
    plt.close(fig)

# Function to save and exit
def save_and_exit(array, parsed_args):
    pygame.quit()
    save_array_to_file(array, parsed_args.output_file)
    if parsed_args.output_png_file:
        save_array_as_image(array, parsed_args.output_png_file)
    exit()

# Function to calculate position in array
def calc_position(array, x, y, drawing_rect):
    if not drawing_rect.collidepoint(x, y):
        return None, None
    rel_x = x - drawing_rect.left
    rel_y = y - drawing_rect.top
    array_x = rel_x * array.shape[1] / drawing_rect.width
    array_y = rel_y * array.shape[0] / drawing_rect.height

    # Debugging information
    print(f"Mouse position: ({x}, {y})")
    print(f"Relative position in drawing_rect: ({rel_x}, {rel_y})")
    print(f"Calculated array position: ({array_x}, {array_y})")
    print(f"Array shape: {array.shape}")
    print(f"Drawing rect size: ({drawing_rect.width}, {drawing_rect.height})")
    print(f"Drawing rect top left: ({drawing_rect.left}, {drawing_rect.top})")

    return int(array_x), int(array_y)

# Main function
def main(args):
    parser = argparse.ArgumentParser(description='Create a new array using a GUI interface.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--rows', type=int, required=True, help='Number of rows for the array')
    parser.add_argument('--cols', type=int, required=True, help='Number of columns for the array')
    parser.add_argument('--input_file', type=str, help='File to start from')
    parser.add_argument('--output_png_file', type=str, help='Path to the output png file')
    parsed_args = parser.parse_args(args)

    print("Be aware this has known issues currently")

    pygame.init()
    width, height = 1200, 800  # Initial width and height
    control_panel_width = 300  # Width for the control panel
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption(f'Draw Generator ({parsed_args.rows}x{parsed_args.cols})')

    manager = pygame_gui.UIManager((width, height))

    brush_size_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, 50), (control_panel_width - 20, 30)),
                                                               start_value=1, value_range=(1, 64), manager=manager)
    brush_size_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 20), (control_panel_width - 20, 30)),
                                                   text='Brush Size', manager=manager)

    img_width_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, 150), (control_panel_width - 20, 30)),
                                                              start_value=parsed_args.cols, value_range=(10, 4096),
                                                              manager=manager)
    img_width_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 120), (control_panel_width - 20, 30)),
                                                  text='Image Width', manager=manager)

    img_height_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, 250), (control_panel_width - 20, 30)),
                                                               start_value=parsed_args.rows, value_range=(10, 4096),
                                                               manager=manager)
    img_height_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 220), (control_panel_width - 20, 30)),
                                                   text='Image Height', manager=manager)

    if parsed_args.input_file:
        array = read_array_from_file(parsed_args.input_file)
    else:
        array = np.zeros((parsed_args.rows, parsed_args.cols), dtype=float)

    mouse_down = False
    mouse_button = None

    clock = pygame.time.Clock()
    try:
        while True:
            time_delta = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exiting...")
                    save_and_exit(array, parsed_args)
                elif event.type == pygame.VIDEORESIZE:
                    print(f"Resizing to {event.w}x{event.h}")
                    width, height = event.w, event.h
                    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                    manager.set_window_resolution((width, height))
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    print("Mouse down")
                    mouse_down = True
                    mouse_button = event.button
                    drawing_rect = pygame.Rect(control_panel_width + 10, 10, width - control_panel_width - 20, height - 20)
                    x, y = calc_position(array, *event.pos, drawing_rect)
                    if x is not None and y is not None:
                        brush_size = int(brush_size_slider.get_current_value())
                        array = update_pixel(x, y, array, mouse_button, brush_size)
                elif event.type == pygame.MOUSEBUTTONUP:
                    print("Mouse up")
                    mouse_down = False
                    mouse_button = None
                elif event.type == pygame.MOUSEMOTION and mouse_down:
                    print("Mouse motion")
                    drawing_rect = pygame.Rect(control_panel_width + 10, 10, width - control_panel_width - 20, height - 20)
                    x, y = calc_position(array, *event.pos, drawing_rect)
                    if x is not None and y is not None:
                        brush_size = int(brush_size_slider.get_current_value())
                        array = update_pixel(x, y, array, mouse_button, brush_size)

                manager.process_events(event)

            new_width = int(img_width_slider.get_current_value())
            new_height = int(img_height_slider.get_current_value())

            if array.shape != (new_height, new_width):
                array = np.zeros((new_height, new_width), dtype=float)

            screen.fill((255, 255, 255))  # Fill the screen with white before drawing

            drawing_rect = pygame.Rect(control_panel_width + 10, 10, width - control_panel_width - 20, height - 20)
            render(array, screen, drawing_rect)

            manager.update(time_delta)
            manager.draw_ui(screen)
            pygame.display.update()
    except KeyboardInterrupt:
        save_and_exit(array, parsed_args)

if __name__ == "__main__":
    main(sys.argv[1:])
