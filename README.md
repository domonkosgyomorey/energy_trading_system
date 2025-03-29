# energy_trading_system

 Self sustaining energy trading system simulator with blockchain technology

## Setup & Run

install the [uv](https://pypi.org/project/uv/) library for faster package installation

```bash
pip install uv
```

Create the environment for this project

```bash
uv venv
```

Activate the previously created environment

```bash
.venv/Scripts/activate
```

Install the dependencies

```bash
uv pip install -r requirements.txt
```

Run the simulation

```bash
python main.py
```


## Development Policy

### Policy for the dependencies

1. The dependency must be stable and supported for the used python version
2. The dependency must be available from the pip package manager
3. The dependency must be downloaded at least 50k times
4. The dependency must be MIT licence
5. If there are no licence, check for licence on the project repository
6. The dependencies must be put inside the requirements.txt
7. The dependency version must be noted ( We should manually upgrade if it is required )

### Policy for commiting to this repository

1. Each feature should be developed on a different branch
2. After a feature is done, it must be fall trought a pull request to the dev branch
3. After some major upgrade on the dev branch are preceeded on the dev branch we should create a pull request to main
4. After each merging our branch into the dev, we must delete that branch

### Extension to must use

> Use vscode preferably

1. black ( code formatter )
2. isort ( import sorter )
3. MyPy type checker

## VSCode user preferences setup

### For mypy

```json
"mypy-type-checker.importStrategy": "fromEnvironment",
"[python]": {
	"editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        },
},
"isort.args": [
        "--profile",
        "black"
],
```
