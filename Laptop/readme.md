This is the readme for the Laptop folder.

The main file we care about in this file is mainScript.py. This is to be run from the laptop and is the core of our functionality.
This file contains MQTT functionality, OpenCV + MediaPipe functionality, as well as the GUI integration.
The GUI demands the main thread on macOS, so the GUI owns the main thread, and MQTT and OpenCV have their own seperate threads.

Next, we have scriptNoGUI.py. This script has the entire computer vision + MQTT functionality just without the GUI as the name suggests.
This is used if you want to visualize the analysis we are performing for a specific exercise since you need the main thread to display an 
image. To reiterate, this is just used for developing new workout classification methods and to visualize our analysis.

Finally, there is the environment.yml file. This is used to create our environment using a single command. This makes setting up the 
project much easier and allows us to set in stone which library versions we are using for this project in case future updates of these 
libraries deprecate any of our functionality.
