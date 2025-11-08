#####################################################################################
#### Option A (Ben bu yontemle yaptim)
#####################################################################################

# 1. Create a clean environment with Python 3.10
conda create -n openfgl python=3.10 -y

# 2. Activate the new environment
conda activate openfgl

# 3. Install core PyTorch/Torchvision (CPU-only)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install PyTorch Geometric (Pulls compatible sparse/scatter dependencies)
pip install torch_geometric

# 5. Install all main dependencies from the cleaned list
pip install -r new_requirements.txt

# 6. Fix the NumPy version conflict (CRITICAL step)
pip install numpy==1.26.4

# 7. Reinstall remaining packages (This ensures everything links to NumPy 1.26.4)
pip install -r new_requirements.txt

# 8. Install the final missing packages
# Your groupmates will need to run these commands for the final dependencies we found:
pip install scikit-network
pip install pymetis
pip install munkres


#####################################################################################
#### Option B (Sonra bu yml dosyasini olusturdum burdan yuklemek isterseniz diye)
#####################################################################################

# This command reads the file and creates a new environment named 'openfgl'
conda env create -f environment.yml

# Activate the newly created environment
conda activate openfgl