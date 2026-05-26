
# add root folder to python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# compile custom operators
cd libs/pointops2
rm -rf build
python setup.py install
cd -
