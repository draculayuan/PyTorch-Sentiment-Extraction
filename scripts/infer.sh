config=${1}
checkpoint=${2}
out_path=${3}

python -u tools/infer.py --config=${config} --checkpoint=${checkpoint} --out_path=${out_path}
