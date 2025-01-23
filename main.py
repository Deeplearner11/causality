from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import numpy as np
import pandas as pd
from lingam import DirectLiNGAM
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import configparser
from json_gen import initialize_model, generate_response  # Importing from json_gen.py
from tasks_func import (  # Importing from tasks_func.py
    run_csl_cgl_task,
    run_cel_ate_task,
    run_cel_hte_task,
    run_cel_ma_task,
    run_cpl_opo_task,
    execute_task,
    determine_method
)
from inference import generate_summary  # Importing from inference.py

def main():
    # Load configuration
    config = configparser.ConfigParser()
    config.read('app.config')

    # Initialize the model
    model_pipe = initialize_model(config)

    while True:
        # Take input query from the user
        prompt = input("Enter your question (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        
        # Generate JSON response
        response_json = generate_response(model_pipe, prompt)
        print("Generated JSON:", response_json)
        
        try:
            json_obj = json.loads(response_json)
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
            continue
        task_function_mapping = {
            ("CSL", "CGL"): run_csl_cgl_task,
            ("CEL", "ATE"): run_cel_ate_task,
            ("CEL", "HTE"): run_cel_hte_task,
            ("CEL", "MA"): run_cel_ma_task,
            ("CPL", "OPO"): run_cpl_opo_task,
            }

        # Use execute_task to handle the json_obj and get the answer
        answer = execute_task(json_obj)
        print("Task Result:", answer)

        # Generate summary
        problem = json_obj["causal_problem"]
        method = determine_method(json_obj)
        summary = generate_summary(problem, prompt, method, answer, n_sentences=4)
        print("Summary:", summary)

if __name__ == "__main__":
    main()
