#cp /home/juan.vallado/src/data_processing_layers_changed.py /home/appuser/TensorMask/.
#cp /home/juan.vallado/src/data_processing.py /home/appuser/.

#sudo python3 /home/juan.vallado/src/data_processing_layers_changed.py
sudo rm -Rf /home/appuser/output/*
sudo python3 /home/juan.vallado/src/data_processing.py
sudo rm -Rf /home/appuser/output/training_eval
sudo python3 /home/juan.vallado/src/inference.py
cd /home/appuser/output/
sudo rm -Rf */
