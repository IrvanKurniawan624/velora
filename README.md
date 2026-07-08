# Velora AI Inference Pipeline

A lightweight production-ready AI inference pipeline skeleton built with Python and `uv`.

## Project Overview

This repository initializes a clean, modular Python-based AI inference pipeline designed for AI hackathons. It defines a structured layout to read inputs from `/input/tasks.json`, process them via lightweight models (decoupled from heavy framework dependencies), and write results to `/output/results.json`.

## Tech Stack

- **Language:** Python 3.12+
- **Package Manager:** [uv](https://github.com/astral-sh/uv)
- **Core Libraries:** `pydantic`, `pydantic-settings`, `httpx`, `openai`
- **Development Tooling:** `pytest`, `ruff`

## Folder Structure

```
.
├── app/                  # Main application package
│   ├── core/             # Core logic and base orchestrators
│   ├── models/           # Custom model wrappers
│   ├── schemas/          # Pydantic schemas for request/response
│   ├── services/         # External service integrations
│   ├── utils/            # Helper utilities (file IO, logging)
│   ├── prompts/          # Prompt templates and management
│   ├── config.py         # Configuration using pydantic-settings
│   ├── main.py           # Application entry point
│   └── __init__.py
├── docs/                 # Documentation files
├── benchmarks/           # Model evaluation and benchmark datasets/tests
├── input/                # Task inputs (e.g., tasks.json)
├── output/               # Output results (e.g., results.json)
├── tests/                # Pytest test suite
├── .env.example          # Environment variables example template
├── .gitignore            # Git ignore file
├── pyproject.toml        # uv project configuration
└── README.md             # Project documentation
```

## Setup Instructions

Ensure you have Python 3.12+ and `uv` installed.

1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd velora
   ```

2. **Initialize Environment:**
   Create `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Add your API keys to the `.env` file:
   ```env
   FIREWORKS_API_KEY=your_key_here
   FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
   ```

3. **Install Dependencies:**
   Create a virtual environment and install all packages using `uv`:
   ```bash
   uv sync
   ```

## How to Run

### Run the Application

You can execute the pipeline entry point:
```bash
uv run python -m app.main
```

### Run Tests

Run the test suite using pytest:
```bash
uv run pytest
```

### Run Linter and Formatter

Format and lint the codebase with Ruff:
```bash
uv run ruff check
uv run ruff format
```
