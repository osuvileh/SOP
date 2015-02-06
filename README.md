# SOP
Video Object recognition and classification system from extracted video frames.
OS: Debian/ raspbian
Hardware: Raspberry Pi Model B

Required packages for running outside target environment:
sudo apt-get install python-pip
#SimpleCV
sudo apt-get install ipython python-opencv python-scipy python-numpy python-setuptools python-pip
sudo pip install https://github.com/sightmachine/SimpleCV/zipball/master

sudo apt-get install python-pygame
sudo pip install svgwrite
#ImageMagick
sudo apt-get install build-essential checkinstall libx11-dev libxext-dev zlib1g-dev libpng12-dev libjpeg-dev 
sudo apt-get build-dep imagemagick
wget http://www.imagemagick.org/download/ImageMagick-6.9.0-4.tar.gz
tar -xzvf ImageMagick-6.9.0-4.tar.gz
cd ImageMagick-6.9.0-4
sudo checkinstall
