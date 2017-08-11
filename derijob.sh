source /home/helcl/virtualenv/tensorflow-1.0-cpu/bin/activate
for layer_size in 10 20 50 100; do
	for epochs in 15 20 25 35; do
		qsub -cwd -j y -v epochs=$epochs -v layer_size=$layer_size submit-deri.sh
		sleep 100
	done
done
