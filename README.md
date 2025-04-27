# HuggingFace Argents Course - GAIA Challenge

## About <a name = "about"></a>

Final project for the Hugging Face AI Agents Course. This repository contains the code for an AI agent designed and evaluated against a subset of the GAIA benchmark. Goal: Achieve >= 30% score.

## Getting Started <a name = "getting_started"></a>

### Installing

All dependencies are managed with UV, so you can install them with the following command:

```bash
uv sync
```

## Usage <a name = "usage"></a>

The agent is designed to be run as a command line tool. You can run it with the following command:

```bash
python -m src.main <COMMAND> [OPTIONS]
```

Where `<COMMAND>` is one of the following:

- `generate_answers`: Generate answers and save them to the results folder.
- `submit_answers`: Submit the answers to the HuggingFace Evaluation API.
- `draw_workflow`: Draw the workflow of the agent and save it as `workflow.html`.

## Tools <a name = "tools"></a>

The agent uses the following tools:

- `google_search`: Search the web for information.
- `wikipedia_search`: Search Wikipedia for Wikipedia pages.
- `youtube_search`: Answers a question based on a Youtube video.
- `decode_text`: Tries to decode a text using the language model.

## Results <a name = "results"></a>

The results of the agent are saved in the `results` folder. Note that the generation runs are NOT deterministic,
so the results may vary between runs. In the `results` folder, you will find results for the following models:

- `gemini-2.0-flash` - 16/20 (80%) correct answers
- `gemini-2.5-pro-preview-03-25` - 15/20 (75%) correct answers
- `emini-2.5-flash-preview-04-17` - 11/20 (55%) correct answers
