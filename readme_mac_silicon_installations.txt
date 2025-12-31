conda create -n openfgl_graphfl python=3.11 -y
conda activate openfgl_graphfl

conda install pytorch torchvision cpuonly -c pytorch -y

# check:
python -c "import torch; print(torch.__version__)"

pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

#if needed: pip install "sympy==1.13.1" --force-reinstall

pip install torch-geometric
# check:
python -c "import torch, torch_sparse, torch_geometric; print('OK', torch.__version__)"

pip install -r docs/new_requirements.txt
pip install scikit-network pymetis munkres

pip install dtaidistance[dtaidistance-c]

# final check:
python -c "import numpy, torch, torch_sparse, torch_geometric; print('OK', numpy.__version__, torch.__version__)"

#benchmark on single run:
python run_table6_graphfl.py --seeds 0 --datasets PROTEINS --groups fgl

#full run:
python run_table6_graphfl.py --seeds 0 1 2 --datasets IMDB-BINARY IMDB-MULTI COLLAB 2>&1 | tee table6_full_log.txt