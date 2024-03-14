# Team3

Welcome to the Github page for piTrainer!

Below is a overall guide to each of the folders in our github, and each folder will have an individual readme for more detail.

Guide:

**ESP32-IMU**

This folder holds the entire necessary code base to have an esp32 successfully reading off of the berryIMU, as well as the implementation for the MQTT service.

It is updated to function specifically on our custom PCB.

You can open this entire folder using the Arduino IDE in order to upload it to our PCB.

**Laptop**

This folder holds the code running from your laptop.

There is also a yml file in this folder in order to allow you to use our environment without needing to download any packages.

**RPI**

This folder holds things necessary for your RPI. 

The data folder also holds a few csvs for the type of data we recieve from the IMU.

The mosquitto.conf file is a configuration file for the MQTT broker on the RPI, this must be moved into /etc/mosquitto/ and replace the standard config file.

You can find more info on setting up the broker in the user manual if needed.

**User Manual**

This is a pdf that holds detailed instructions for both set up, as well as usage of piTrainer.
