# COMP 590-175 ISP Assignment

## How to use:

    - The project runs through the main.py file and can be run directly from the file. All you have to do to run the program is press the run button on the top right of the display window, and the program will do all of the work for you.

    - Note that the program runs very slowly and could take a few minutes to complete. 

    - The program will automatically save the images at each step of the imaging process, so you can view the results of each step of the imaging process.

    - After the compression step is complete, the program will prompt the user to select a white patch of the image to perform manual white balancing. For the best results, the user should select the top left and bottom right corner of some white patch from the top left quadrant of the image.

    - After selecting the patch (which should take just two mouse clicks on the image), the program will save the final images in the final_images folder within the data folder, and the program will teminate.

## Steps of the imaging pipeline:

    - Linearization: Converts the 2D array into a linear image and combats any dark noise or oversaturated pixels that the image may have by clipping the pixel values to be in the range [0, 1].

    - White balancing: The program will perform the white balancing with the white world version, gray world version, and a camera presets version. The camera presets version of the white balancing will multiply each pixel by a predetermined scaling factor. 

    - Demosaicing: The process of performing a bilinear demosaicing transformation corrects the pixels of the image based on the Bayer pattern. Since the pixels we have for our image are RGGB, we have to factor this into the calculation to make the image appear less green.

    - Color space correction: Converts the RGB pixels into SRGB space, which makes the colors of the image look much more natural. 

    - Brightness adjustment and gamma encoding: The image is brightened by linearly scaling it and clipping all intensity values for each pixel. 

    - Manual white balancing: By selecting a patch of the image that looks white, we can compare this with the pixels in our image and asjust the brightness factor for each pixel.
