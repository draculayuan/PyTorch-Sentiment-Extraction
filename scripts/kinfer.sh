config=${1}
out_path=${2}
job_name=${3}
iter=${4}

python -u tools/kfold_infer.py --config=${config} --out_path=${out_path} --job_name=${job_name} --iter=${iter}
