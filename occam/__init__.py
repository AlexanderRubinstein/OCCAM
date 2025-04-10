import os


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

os.environ["PROJECT_ROOT_PROVIDED_FOR_STUNED"] = PROJECT_ROOT
