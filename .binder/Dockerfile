FROM zivgitlab.wwu.io/pymor/pymor/binder:2021-2-1

# binder wants to set the NB_ vars anyways, so we use it to service both setups
ARG NB_USER
ARG NB_UID

USER root
RUN rm -rf /pymor && useradd -d /pymor --shell /bin/bash -u ${NB_UID} -o -c "" -m ${NB_USER}

ENV USER=${NB_USER} \
    HOME=/pymor

COPY . ${HOME}
RUN chown -R ${NB_USER} /pymor && python -m pip install --no-cache-dir notebook jupyterlab pymor rise jupytext ipywidgets
USER ${NB_USER}

ENTRYPOINT []
WORKDIR /pymor
