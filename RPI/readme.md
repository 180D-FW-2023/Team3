This is the readme for the RPI specific folder

In this folder, the main thing to care about is mainScript.py.

This script enables the RPI and acts as the center of communication between the IMU and the Laptop.
It uses MQTT to accept data stream from the IMU, and the result stream from the Laptop and integrates these two together to determine
errors.

There is also the mosquitto.conf file. This file belongs in /etc/mosquitto/ and enables the type of broker we want the RPI to be running.
There are more detailed instructions for setting up the broker in the user manual.

There is also a small data folder that holds CSV files that display the type of data the RPI is recieving from the IMU/ESP32.
