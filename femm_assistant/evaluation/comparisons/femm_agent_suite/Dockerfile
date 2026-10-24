FROM research-extender-femm:dev-connection10-20261024
USER root
WORKDIR /suite
COPY . /suite
RUN pip install --no-cache-dir '.[dev]' && pip freeze > /suite/evaluation-dependencies.txt
ENV HOME=/tmp FEMM_AGENT_HOME=/tmp/femm-agent-state FEMM_AGENT_WORK=/testwork/sessions FEMM_DIR=/opt/femm/bin FEMM_AGENT_NO_GUI=1 MPLCONFIGDIR=/tmp/matplotlib
USER assistant
CMD ["python", "-m", "pytest", "-q"]
