file=$1
chunk_size=$2
tmpdir=chunks
nl=`cat $1 | wc -l`
chunk_count=`expr $nl / $chunk_size - 1`
echo "Splitting $nl lines into $chunk_count parts."
rm -rf $tmpdir
mkdir $tmpdir
for i in `seq 0 $chunk_count`; do
	offset=`expr $i \* $chunk_size`
	head -n $offset $file | tail -n $chunk_size > $tmpdir/chunk.$i
done
