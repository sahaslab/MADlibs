# MAD-libs
Study on the distribution of generated outputs from ***M***usicGen, ***A***CE-Step, ***D***iffRhythm, and ***L***yria

## Dependencies
Run `pip install -r requirements.txt` to install dependencies.

If you add a new dependency, please add it to one of the `requirements.txt` depending on which part(s) it's most relevant to.

`nltk` is a required dependency for prompt generation.
If you set `prompt_strategy.use_nltk_pos_tagger=true`, download the tagger once before running:
`python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"`

## Hydra Pipeline

### Prompt Generation
There are 1-2 layers depending on your configuration:
1. **Base generator** (creates candidate prompts)
2. **Optional ranker** (over-generates then keeps the best/diverse subset)

Think of it like: `base strategy` -> maybe `rank` -> final prompts

### Base Generators

#### `blank`
Repeats one fixed string (by default, empty string)

Command:
`python3 run_prompts.py prompt_strategy=blank`

#### `keyword_mix`
Randomly mixes style, mood, and instrument words from small lists

Command:
`python3 run_prompts.py prompt_strategy=keyword_mix`

#### `madlibs` (w/ 3 diff internal modes)
Uses part-of-speech slots and then fills them with lexicon words. Modes are:

- **Original templates** (`induction_method=none`)
  Uses only the hand-written template list

  Command:
  `python3 run_prompts.py prompt_strategy=madlibs prompt_strategy.induction_method=none`

- **Induced templates** (`induction_method=templates`)
  Learns template structures from caption corpus, then samples from those learned templates

  Command:
  `python3 run_prompts.py prompt_strategy=madlibs prompt_strategy.induction_method=templates`

- **PCFG** (`induction_method=pcfg`)
  Learns a probabilistic grammar over POS sequences and then samples syntax from that grammar

  Command:
  `python3 run_prompts.py prompt_strategy=madlibs_pcfg`

#### `dependency_graph`
Builds lightweight dependency-role templates from captions (subject/object-ish roles around a root token), then resamples those templates

Command:
`python3 run_prompts.py prompt_strategy=dependency_graph`

#### `llm` (w/ 2 diff internal modes)
Uses an LLM directly. Modes are:

- **"Seed" generation** (`generation_mode=seed`)
  Asks the model for one prompt per request (basically unconditional/system-prompt-conditional prompt sampling)

  Command:
  `python3 run_prompts.py prompt_strategy=llm prompt_strategy.generation_mode=seed`

- **Caption alternates** (`generation_mode=caption_alternates`)
  For each source caption, generates `K` alternates by slotting and replacing. Output count is approximately `K * N`, where:
  - `K = prompt_strategy.alternates_per_caption`
  - `N = number of filtered source captions`

  Command:
  `python3 run_prompts.py prompt_strategy=llm prompt_strategy.generation_mode=caption_alternates`

### Ranked Wrappers
Rankers generate many candidates from a base strategy, score them with a caption-trained n-gram LM + diversity/repetition terms, then keep a diverse top subset

#### `ranked_madlibs`
Wraps `madlibs` base strategy

Default command:
`python3 run_prompts.py prompt_strategy=ranked_madlibs`

You can set `madlibs` configuration options with `prompt_strategy.base_strategy.*` for example:
`python3 run_prompts.py prompt_strategy=ranked_madlibs prompt_strategy.base_strategy.induction_method=none`

#### `ranked_dependency_graph`
Wraps a dependency-graph base strategy.

Command:
`python3 run_prompts.py prompt_strategy=ranked_dependency_graph`

### Typical Workflow

1. Generate prompts:
`python3 run_prompts.py prompt_strategy=madlibs`

2. Generate audio (reuses existing prompts file):
`python run_generation.py generator=musicgen`

3. Run analysis:
`python run_mir.py analysis=full_pipeline`

### Lyria
Lyria can be run with:
`python run_generation.py generator=lyria`

This generator uses the Google `google-genai` realtime music session under the hood and saves fixed-length `.wav` files for each prompt. Set either `GOOGLE_API_KEY` or `GEMINI_API_KEY` before running.

## Formatting
Please format code contributions by running `ruff format`.

## Testing
Please run the pytest suite with `PYTHONPATH=. pytest`.
