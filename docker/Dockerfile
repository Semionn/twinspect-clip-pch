FROM continuumio/anaconda3:2023.07-1
LABEL authors="semionn"

RUN conda create --name twinspect_env python=3.10

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "twinspect_env", "/bin/bash", "-c"]

RUN conda init bash
RUN echo "conda activate twinspect_env" > ~/.bashrc
RUN conda install faiss-cpu -c conda-forge
RUN conda install -y poetry==1.4.0

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y nano git
RUN apt-get update && apt-get install -y build-essential libtag1-dev
RUN git clone https://github.com/iscc/twinspect
RUN sed -i 's/python = "^3.11"/python = "^3.10"/' twinspect/pyproject.toml
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN apt-get update && apt-get install -y libutfcpp-dev libmagic-dev
RUN cd twinspect && sed '13 i\
urllib3="^1.26.16"\
' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml

#SHELL ["conda", "run", "-n", "twinspect_env", "/bin/bash", "-c"]
#RUN cd twinspect && poetry lock && poetry install

# first command in bash:
# conda init bash
# . ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate twinspect_env

#  cd twinspect/
#  pip install --ignore-installed --no-deps PyMuPDF==1.23.0
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/twinspect_env/lib/python3.10/site-packages/fitz_new/
# pip install dhash 'transformers[torch]' scikit-learn
#  poetry lock

#  poetry install
#  poetry install # to finish completely in case of unfinished installation
#  poetry run python -m twinspect

# python3
# from twinspect.algos.clip_pch import test
# test()

ENTRYPOINT ["/bin/bash"]
