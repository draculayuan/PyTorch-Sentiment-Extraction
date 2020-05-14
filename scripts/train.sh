config=${1}

if [ $# -eq 2 ]; then
  job_name=${2}
else
  job_name=training
fi

python -u tools/train.py --config=${config} --job_name=${job_name}
