dir=$1
of=$2
nof=`ls $dir | wc -l`
seq 0 `expr $nof - 1` > perm
sort -r perm -o perm
cat perm | while read idx; do cat $dir/chunk.$idx >> $of; done
