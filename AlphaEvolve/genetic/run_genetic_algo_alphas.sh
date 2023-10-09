while getopts r:o: flag
do
    case "${flag}" in
        r) rounds=${OPTARG};;
        o) output_dir=${OPTARG};;
    esac
done

home=$(pwd)

if [ -z "$output_dir" ]
then
	echo "No output_dir argument supplied"
else
	echo "delete repeated results"
	rm -rf ${home}/${output_dir}
fi

mkdir -p ${home}/${output_dir}

declare -A best_valid_returns_=()
declare -A best_test_returns_=()

for i in $(seq 1 $((rounds)))
do
	best_valid_returns_[$i]=""
    best_test_returns_[$i]=""        
done

for round in $(seq 1 $((rounds)))
do 
	rm -rf $(pwd)/${output_dir}/genetic_baseline_${round}
	mkdir -p $(pwd)/${output_dir}/genetic_baseline_${round} 
	if [ $round -eq 1 ]
	then
		python $home/genetic_alpha.py --data_path "$(pwd)" --output_path "$(pwd)/${output_dir}/genetic_baseline_${round}" --ith_round $round
        best_valid_returns_[$round]="$(pwd)/${output_dir}/genetic_baseline_${round}/genetic_baseline_${round}th_round_validate_returns.pkl"
        best_test_returns_[$round]="$(pwd)/${output_dir}/genetic_baseline_${round}/genetic_baseline_${round}th_round_test_returns.pkl"
	else
		best_valid_returns=""
		best_test_returns=""
		for i in  $(seq 1 $(($round - 1)))
		do
       		ith_best_valid_returns=${best_valid_returns_[${i}]}
        	best_valid_returns+="${ith_best_valid_returns};"
        	ith_best_test_returns=${best_test_returns_[${i}]}
        	best_test_returns+="${ith_best_test_returns};"
    	done
		python $home/genetic_alpha.py --data_path "$(pwd)" --output_path "$(pwd)/${output_dir}/genetic_baseline_${round}" --ith_round $round --cutoffs_valid $best_valid_returns --cutoffs_test $best_test_returns 
        best_valid_returns_[$round]="$(pwd)/${output_dir}/genetic_baseline_${round}/genetic_baseline_${round}th_round_validate_returns.pkl"
    	best_test_returns_[$round]="$(pwd)/${output_dir}/genetic_baseline_${round}/genetic_baseline_${round}th_round_test_returns.pkl"
	fi
done
