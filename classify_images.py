import os
import cv2
import shutil


def copy_image_to_category_folder(images_path, image, image_type):
    """Copies the image into the required category folder."""
    image_folder = 'other'
    if image_type == 'w':
        image_folder = 'water'
    # suspecting that the photos to be classified are in a folder inside the main folder
    shutil.copyfile(os.path.join(images_path, image), os.path.join(images_path, "./../", image_folder, image))


def classify_images(images_path):
    """Loops through and displays the images inside the given folder, then
    classifies the images based on the pressed key(s).
    """
    for image in os.listdir(images_path):
        # if not a file, we continue to the next item
        if not os.path.isfile(os.path.join(images_path, image)):
            continue
        # The solution below is written to work with any number of categories
        # and one photo can be classified as several categories.
        # (for example, one photo can contain a car and a building as well)
        while True:
            img = cv2.imread(os.path.join(images_path, image))
            try:
                cv2.imshow("Photos", img)
            except:
                # if the file is not a valid image, cv2 throws an error so we just continue to the next one
                break
            # waiting for any key
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k in [ord('w'), ord('o')]:  # add more characters for more categories
                # if the characters are expected, we classify the image
                copy_image_to_category_folder(images_path, image, chr(k))
                # we continue with the same image as it can have multiple categories
                continue
            # pressing any other key means that the photo is not required anymore, we delete it
            os.remove(os.path.join(images_path, image))
            # breaking the loop, getting the next image
            break


classify_images("/path/to/holiday_photos/2023_best_trip_ever")
