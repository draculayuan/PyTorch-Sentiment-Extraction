config=${1}
checkpoint=${2}

python -u tools/test.py --config=${config} --checkpoint=${checkpoint}
