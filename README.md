# Generative AI Demo for Clinicians

This repository contains resources for a demo of generative AI applications using Python and LangChain, tailored for clinicians.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)


## Introduction

This project demonstrates the use of generative AI models to assist clinicians in various tasks, such as generating medical case summaries. The demo leverages the LangChain framework and integrates with Google Generative AI models.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/dermatologist/llm-demo.git
    cd llm-demo
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Agent (Example)

To run the agent (or any script), execute:
```sh
python agent.py
```