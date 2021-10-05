

test:
	python -m unittest obf.dataloader.test_loader
	python -m unittest obf.dataloader.test_ae


create:
	conda env create -f environment.yml

format:
	yapf -ri --style="{based_on_style: google, indent_width: 2}" .

export:
	conda env export > environment_obf.yml


update:
	conda env update --prefix ./env --file environment.yml  --prune


package:
	# We mainly use python 3.9 and PyTorch 1.9 for the development. Other versions should also
	# work well under our code. 
	# In addition to PyTorch, these packages are also need: 
	#   numpy scipy scikit-learn termcolor tqdm matplotlib pandas tensorboardX
	#   jupyterlab (if you need to use Jupyter Notebook)
	# Their version should not matter too much. If you encountered version problems, please use 
	# the provided "environment_obf.yml" file to install the exact same version.
	conda install python=3.9
	conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
	conda install numpy scipy scikit-learn termcolor tqdm matplotlib pandas tensorboardX
	conda install -c conda-forge jupyterlab


