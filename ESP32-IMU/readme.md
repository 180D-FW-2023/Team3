This is the readme file for the ESP32-IMU folder

This folder holds the entire set of files necessary for the building and uploading of the main arduino-BerryIMU.ino script that is
uploaded and ran on our IMU.

Most of this code is from ozzmaker's open source code for the BerryIMU on Arduino, although some functionality was removed and some was added.

Our PCB was not compatible with their method of detecting the IMU as well as the version of the IMU, so the dependency on this was removed
and replaced with hard coding in the version of our IMU. We also removed the portion of the code responsible for the gyroscope functionality as
this was bugged and just resulting in a infinitely exploding value readings from the gyroscope.

Past this, WIFI and MQTT functionality were added to this code to make communication with our RPI possible. We also added a simple low pass 
filter to the data to try and remove and extraneous readings from the noisy accelerometer.
