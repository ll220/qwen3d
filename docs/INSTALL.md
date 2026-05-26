## Conda Installation
Note: If you are using micromamba, set:
```
alias conda='micromamba'
```

Note: The instructions below install a standalone CUDA installation in your conda enviorment instead of using the system installation. You may need to use a different CUDA version based on your system drivers, GPU, etc. You can skip this step and instead just set `CUDA_HOME`, e.g., `export CUDA_HOME='/usr/local/cuda-12.4'`.

```
conda create -n qwen3d python=3.10
conda activate qwen3d
conda install cuda cuda-nvcc -c nvidia/label/cuda-12.4.1 # Optional
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH" # Optional
export CUDA_HOME=$CONDA_PREFIX # Optional

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install flash-attn==2.6.3 --no-build-isolation
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
pip install spacy
python -m spacy download en_core_web_sm
python -c 'import nltk; nltk.download("stopwords")'

pip install -r docs/requirements.txt
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
bash docs/init.sh # Warning: You may need to set CUDA_HOME and TORCH_CUDA_ARCH_LIST (see below)
```


## Troubleshooting

To support multiple GPU architectures, you will need to set `TORCH_CUDA_ARCH_LIST`. For example:
```
export TORCH_CUDA_ARCH_LIST="8.0 8.6"
```

See this [guide](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for more details. 