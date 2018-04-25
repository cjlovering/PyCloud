sudo apt-get update
sudo apt-get -y install python3-pip
pip3 install -r requirements.txt
python3 src/download/download_mnist.py
python3 src/download/download_snli.py