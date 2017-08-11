source /home/helcl/virtualenv/tensorflow-1.0-cpu/bin/activate
echo $epochs
python3 model.py --exp layer-size-$layer_size --epochs $epochs --layer_size $layer_size
