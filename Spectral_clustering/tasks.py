from invoke import task
from main import main

@task
def run(c, k=None, n=None, Random=True):
    main(k, n, Random)