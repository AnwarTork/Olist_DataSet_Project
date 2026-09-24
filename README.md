# Olist MLOps Project

MLOps project for predicting late order delivery using the Olist dataset.

## Project Structure

- `app/` - FastAPI application
- `config/` - configuration files
- `data/` - raw, processed and split data
- `models/` - trained model artifacts
- `notebooks/` - exploratory and development notebooks
- `src/` - reusable Python modules
- `tests/` - automated tests
- `requirements/` - runtime and development dependencies

## Environment

Python 3.9

## Installation

Create a virtual environment:

```bash
python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements/requirements-dev.txt
Configuration

Copy .env.example to .env and configure the PostgreSQL connection.

Project parameters are stored in:

config/config.yaml

Secrets are stored in .env.

Current Model Task

Binary classification:

0 = On-time
1 = Late

The target column is:

is_late

Workflow

Data preparation → Labeling → Train/Validation/Test split → EDA → Feature Engineering → Model Training → Evaluation → API → Monitoring