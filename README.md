# Gelato-Env

Gelato-Env is an RL environment modelling a single Gelateria that sells multiple flavours of ice cream. 
However, it can also be interpreted generically as a single store that sells multiple products.

The environment is based on the [OpenAI Gym](https://www.gymlibrary.dev/) framework, with additional features to support the modelling of a store.

## Design
The design of the Gelato-Env environment is shown in the following diagram:
![Design of Gelato-Env](imgs/GelatoEnv_diagram.png)

## Usage
A pre-trained sales prediction model is needed to make use of the environment. There is no default model provided, but the user can train their own model with reference to the sales prediction model training script.

## Training the Sales Prediction Model
The training script provides a simple sales prediction model using a fairly simple MLP.





## Logging
This environment has integration with [wandb](https://docs.wandb.ai) to log the training process, so the user will need to set up a wandb account and enter their `WANDB_API_KEY` when prompted.
It is possible to turn off the wandb logging in the configuration, but the progress of training will only be visible in the console.

## Installation


### Linux

It is convenient to make use of `pipx` to install general helper packages:

```bash
python -m venv $HOME/.venvs
source $HOME/.venvs/bin/activate
pip install pipx
pipx install black
pipx install isort
pipx install ruff
pipx install pre-commit
```

Use the Makefile to install the repo and its dependencies:

```bash
make setup
```

#