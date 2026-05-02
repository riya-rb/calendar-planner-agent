# Calendar Planner Agent

An autonomous calendar planning system that generates and evaluates optimized schedules using LLM-based agents. The project includes an interactive UI for real-time planning as well as an evaluation pipeline to benchmark agent performance against baseline strategies and prompting methods.

---

## Overview

The Calendar Planner Agent is designed to help users organize events into structured weekly schedules. It uses LLM-based reasoning to:

- Generate optimized calendar plans under constraints
- Compare multiple prompting strategies
- Benchmark agent performance against rule-based baselines
- Evaluate schedule quality on synthetic datasets

The system consists of two main components:

### 1. Interactive UI (Streamlit App)
A web-based interface for generating and visualizing schedules in real time.

### 2. Evaluation Pipeline
A reproducible benchmarking framework that compares:
- LLM-based agent vs baseline approaches
- Different prompting strategies

---

## Project Structure

```bash
calendar-planner-agent/
│
├── app.py                  # Streamlit UI entry point
├── eval/
│   └── evaluate.py         # Evaluation pipeline (agent vs baseline)
│
├── agent/                  # LLM-based scheduling agent logic
├── baseline/               # Rule-based or heuristic baselines
├── prompts/               # Prompt engineering strategies
├── data/                  # Synthetic dataset generation / samples
├── utils/                 # Helper functions
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/riya-rb/calendar-planner-agent.git
cd calendar-planner-agent
```
### 2. Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
## Running the UI

Run the interactive scheduling interface:
```bash
streamlit run app.py
```
This will:

- Start a local server
- Open the UI in your browser
- Allow interactive schedule generation and visualization

## Run Evaluation

1. Execute the benchmarking pipeline:
```bash
python3 eval/evaluate.py
```
This script will:

- Run agent-based scheduling experiments
- Compare against baseline methods
- Evaluate different prompting strategies
- Output performance metrics

2. Run the batch evaluator to compare scenarios pass/fail:
```bash
python3 -c "from main import run_agent; run_agent(batch_mode=True)"
```

## Data
- All data is synthetically generated

## System Design

The system follows a modular pipeline:

- User input / synthetic task generation
- LLM-based scheduling agent produces a plan
- Baseline system generates comparison schedule
- Evaluation module scores both outputs
- Metrics are aggregated and reported
- System is interactive via a Streamlit UI

## Future Improvements
- Integration with Google Calendar / Outlook APIs
- Personalization based on user behavior history
- Follow up conversation/discussion in case of scheduling failure
