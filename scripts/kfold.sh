config=${1}

if [ $# -eq 2 ]; then
  job_name=${2}
else
  job_name=kfold
fi

python -u tools/kfold_trainval.py --config=${config} --job_name=${job_name}
