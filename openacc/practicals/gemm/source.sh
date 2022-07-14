export CPATH=`echo $CPATH |sed "s@$CRAY_NVIDIA_PREFIX/compilers/include@@"`
export CPATH=$CRAY_NVIDIA_PREFIX/cuda/11.2/targets/x86_64-linux/include:$CPATH
