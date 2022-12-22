## Create a environment
https://srinivas1996kumar.medium.com/adding-custom-kernels-to-a-jupyter-notebook-in-visual-studio-53e4d595208c

https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

https://towardsdatascience.com/managing-virtual-environment-with-pyenv-ae6f3fb835f8

https://docs.jupyter.org/en/latest/running.html

https://code.visualstudio.com/docs/datascience/jupyter-notebooks

 python -m venv ons-env

 .\ons-env\Scripts\activate
 
 pip install jupyter
 pip install ipykernel

 python -m ipykernel install --user --name ons-env  --display-name "ons-env"

 jupyter notebook