from invoke import task

VENV_PREFIX = "pipenv run"


@task
def init_dev(ctx):
    ctx.run("export SYSTEM_VERSION_COMPAT=1 &&" "pipenv install --dev")


@task
def format(ctx):
    ctx.run(f"{VENV_PREFIX} isort --atomic .")
    ctx.run(f"{VENV_PREFIX} black .")


@task(optional=["clean"])
def doc_build(ctx, clean=True):
    """Build documentation locally"""
    argument = ""
    if clean:
        argument += " --clean"

    ctx.run(f"{VENV_PREFIX} mkdocs build {argument}")
