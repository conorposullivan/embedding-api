# Embedding API Assignment

Candidate name: Conor O'Sullivan

## Prerequisites

It it recommended to use a computer that has 8GB RAM or more.

This API was created using Python 3.8

## Setup

The encoder can be downloaded from the following url: https://tfhub.dev/google/universal-sentence-encoder/4

To extract the contents use the following command:

```
tar xzf universal-sentence-encoder_4.tar.gz
```

Move the contents so the 'Models' directory resembles the following structure:

```
/models
├── universal-sentence-encoder_4/
│   ├── assets/
│   ├── saved_model.pb
│   ├── variables
```

Set the following environment variable:

```
export FLASK_APP=api
```

Python dependencies can be installed using the following command:

```
pip install -r requirements.txt
```

## Run application

The API will run on port 5000 by running the following command:

```
python3 -m flask run --no-debugger --no-reload
```

## Run tests

Tests can be run using the following command:

```
pytest
```

Note: The API must be running for all tests to pass.
