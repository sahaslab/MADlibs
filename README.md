# MAD-libs
Study on the distribution of generated outputs from ***M***usicGen, ***A***CE-Step, and ***D***iffRhythm

## Dependencies
Run `pip install -r requirements.txt` to install dependencies.

If you add a new dependency, please add it to requirements.txt

## Hydra Pipeline
Uses Hydra for all runtime configuration.

Generate prompts:
`python3 run_prompts.py prompt_strategy=madlibs`

Generate audio (reuses existing prompts file):
`python run_generation.py generator=musicgen`

Run MIR:
`python run_mir.py analysis=full_pipeline`

## Formatting
Please format your code contributions by running `ruff format`

## Testing
Please run pytest suite with `PYTHONPATH=. pytest`
