import os
from stuned.utility.utils import PROJECT_ROOT_ENV_NAME


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

os.environ[PROJECT_ROOT_ENV_NAME] = PROJECT_ROOT
