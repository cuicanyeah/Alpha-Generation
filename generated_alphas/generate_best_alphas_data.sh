# This file generate alpha's predictions for the test period
while getopts i:o: flag
do
    case "${flag}" in
        i) input_dir=${OPTARG};; # best_alphas
        o) output_dir=${OPTARG};; # save_preds_only_244
    esac
done

input_dir=${1:-best_alphas}
output_dir=${2:-save_preds_only_244}

mkdir $(pwd)/$output_dir

home=$(pwd)

path=$home/$input_dir

best_alpha_files="$(ls $path)"

for alpha in $best_alpha_files
do
	./run_baseline.sh -m /$home/$input_dir/$alpha -a MY_ALPHA -s 10000000000 -n 50 -c 0.6 -d 1000099 -g /$home/$output_dir/output_data_$alpha
	# timeout 2m bash ./run_baseline.sh -a MY_ALPHA -p $path  -m $path/$alpha -s 100000 -v "${best_valid_returns}" -t "${best_test_returns}"
done
