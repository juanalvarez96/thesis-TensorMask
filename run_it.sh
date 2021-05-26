ROOT_DIR=$PWD
DATA_DIR=/home/juan.vallado/data
CODE_DIR=$ROOT_DIR/src

export UID=$(id -u)
export GID=$(id -g)

username=$(whoami)


grep "^${username}:" /etc/passwd > .passwd.$$
grep "^${username}:" /etc/group > .group.$$

nvidia-docker  run   \
	-e CUDA_VISIBLE_DEVICES=0 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
	-v $DATA_DIR:/home/$username/data \
	-v $CODE_DIR:/home/$username/src \
	-v $ROOT_DIR/output:/home/$username/output \
	--name juan_thesis_tensormask \
	-it image_juan \
	bash




rm .passwd.$$
rm .group.$$
