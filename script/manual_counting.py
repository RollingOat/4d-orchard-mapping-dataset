
import cv2
import os


# Mouse callback function to draw a dot
def draw_dot(event, x, y, flags, param):
    global dot_count, image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a small circle (dot) at the location of the click
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red dot with a radius of 5
        dot_count += 1
        print(f"Dot count: {dot_count}")

image_folder = ""  # Replace 'your_image_folder' with the path to your image folder
save_folder = ""  # Replace 'your_image_folder' with the path to your image folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# load all images in the folder
# Get the screen resolution
screen_width = 1920  # Replace with your screen's width resolution
screen_height = 1080  # Replace with your screen's height resolution

image_names = os.listdir(image_folder)
image_names = [os.path.join(image_folder, image_name) for image_name in image_names]

for image_name in image_names:
    # Initialize the counter for the number of dots
    dot_count = 0
    image = cv2.imread(image_name)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", screen_width, screen_height)  # Set window size to the resized image
    cv2.setMouseCallback("Image", draw_dot)

    while True:
        # Display the image
        cv2.imshow("Image", image)
        
        # Wait for the 'q' key to be pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("number of fruits counted: ", dot_count)
            break

    
    # Clean up and close the window
    cv2.destroyAllWindows()

    # Print the final number of dots
    print(f"Total number of dots drawn: {dot_count}")

