import pandas as pd
import numpy as np
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
import configparser
# Define the functions for each causal task
def run_csl_cgl_task(nodes, dataset):
    """
    Learn the causal DAG using LinGAM and plot the resulting graph.
    
    Parameters:
    - file_path: str, path to the CSV file containing the data.
    - variables: list of str, the column names of the variables to include in the analysis.
    
    Returns:
    - model: DirectLiNGAM object, the fitted LinGAM model.
    - adj_matrix: np.ndarray, the adjacency matrix representing the causal structure.
    """
    # Load the dataset
    df = pd.read_csv(dataset)
    
    # Select the relevant variables
    df = df[nodes]
    
    # Apply DirectLiNGAM to the data
    model = DirectLiNGAM()
    model.fit(df)
    
    # Get the adjacency matrix (causal structure)
    adj_matrix = model.adjacency_matrix_
    
    # Print adjacency matrix
    #print("Adjacency Matrix:")
    #print(adj_matrix)
    
    # Plot the learned DAG
    G = nx.DiGraph(adj_matrix)
    labels = {i: df.columns[i] for i in range(len(df.columns))}
    G = nx.relabel_nodes(G, labels)
    # Initialize a list to store significant causal relationships
    significant_relationships = []
    adjacency_matrix = adj_matrix
    variable_names = nodes
    K=2
    
    # Iterate through the upper triangle of the adjacency matrix (excluding diagonal)
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if i != j and adjacency_matrix[i, j] != 0:  # Avoid self-loops and check non-zero entries
                significant_relationships.append((adjacency_matrix[i, j], variable_names[i], variable_names[j]))
    
    # Sort the relationships by significance (assuming higher values indicate higher significance)
    significant_relationships.sort(reverse=True, key=lambda x: abs(x[0]))
    
    # Select the top K relationships (or fewer if less than K relationships exist)
    top_relationships = significant_relationships[:K]
    
    # Create summary sentences
    if len(top_relationships) == 0:
        summary = "There are 0 pairs of significant causal relationships."
    else:
        summary = f"There are {len(top_relationships)} pairs of significant causal relationships."
        for value, x, y in top_relationships:
            summary += f" The {x} would causally influence the {y}."
    print(summary)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Layout for nodes
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=12, font_weight="bold", arrowstyle="->", arrowsize=10)
    plt.title("Learned Causal DAG using LinGAM")
    plt.show()
    
    return summary


def run_cel_ate_task(treatment, response, dataset):
    """
    Calculate the Average Treatment Effect (ATE) for continuous treatment and response using the Inverse Probability Weighting (IPW) method.
    
    Parameters:
    - df (pd.DataFrame): The dataset containing treatment and response.
    - treatment_col (str): The name of the treatment variable.
    - response_col (str): The name of the response variable.
    
    Returns:
    - ATE (float): The estimated Average Treatment Effect.
    """
    # Load the dataset
    df = pd.read_csv(dataset)
    treatment_1 =treatment
    response_1 = response
    treatment = df[treatment[0]]
    response = df[response[0]]

    # Step 1: Estimate the propensity score for continuous treatment
    # Since we are not using covariates, we calculate the propensity score directly from treatment
    treatment_mean = treatment.mean()

    # Handle cases where treatment_mean is 0 to avoid division by zero
    treatment_mean = max(treatment_mean, 1e-8)

    # Step 2: Calculate weights
    weights = treatment / treatment_mean + (1 - treatment) / (1 - treatment_mean)

    # Step 3: Compute weighted outcomes
    weighted_outcomes = weights * response

    # Step 4: Calculate the ATE
    ate = np.mean(weighted_outcomes)
   
    print(f"The average treatment effect of setting {treatment_1[0]} as 1 on the {response_1[0]} is {ate:.4f}.")
    ate_summary = f"The average treatment effect of setting {treatment_1[0]} as 1 on the {response_1[0]} is {ate:.4f}."
    return ate_summary

def run_cel_hte_task(treatment, response, condition, dataset):
    """
    Function to implement T-Learner for Conditional Average Treatment Effect (CATE).

    Parameters:
    - X: pd.DataFrame
        Features (covariates).
    - treatment: pd.Series or np.array
        Treatment variable.
    - response: pd.Series or np.array
        Outcome variable.
    - condition: tuple
        Tuple containing (condition_variable_name, condition_value).
    - model: sklearn-like model
        Machine learning model to be used.
    - test_size: float (default=0.2)
        Proportion of data to be used as test set.
        
    Returns:
    - treatment_effect: np.array
        Estimated treatment effects.
    - mae: float
        Mean Absolute Error on the test set.
    """
    # Initialize a model
    model = RandomForestRegressor()
    treatment_1 = treatment
    response_1 = response
    test_size=0.2
    X = pd.read_csv(dataset)
    treatment = X[treatment[0]]  # Extract treatment as Series
    response = X[response[0]]    # Extract response as Series


    # Extract condition variable name and value
    condition_variable, condition_value,condition_type = condition
    # Convert condition_value to numeric type if possible
    try:
        condition_value = float(condition_value)  # Try converting to float
    except ValueError:
        pass  # If conversion fails, leave it as string (may cause issues if the column is numeric)
    # Filter the data based on the condition_type
    if condition_type == '>':
       condition_mask = X[condition_variable] > condition_value
    elif condition_type == '<':
       condition_mask = X[condition_variable] < condition_value
    elif condition_type == '=':
       condition_mask = X[condition_variable] == condition_value
    elif condition_type == '>=':
       condition_mask = X[condition_variable] >= condition_value
    elif condition_type == '<=':
       condition_mask = X[condition_variable] <= condition_value
    else:
       raise ValueError("Invalid condition type provided. Expected '>', '<', '=', '>=', or '<='.")
    X_conditioned = X[condition_mask]
    treatment_conditioned = treatment[condition_mask]
    response_conditioned = response[condition_mask]

    # Split the conditioned data into training and test sets
    X_train, X_test, treatment_train, treatment_test, response_train, response_test = train_test_split(
        X_conditioned, treatment_conditioned, response_conditioned, test_size=test_size, random_state=42
    )

    # Create two copies of the model for treated and control groups
    model_treated = clone(model)
    model_control = clone(model)

    # Train the model for the treated group (when treatment > median)
    treated_mask = treatment_train > np.median(treatment_train)
    model_treated.fit(X_train[treated_mask], response_train[treated_mask])

    # Train the model for the control group (when treatment <= median)
    control_mask = ~treated_mask
    model_control.fit(X_train[control_mask], response_train[control_mask])

    # Predict outcomes for both treated and control models
    y_pred_treated = model_treated.predict(X_test)
    y_pred_control = model_control.predict(X_test)

    # Calculate treatment effect as the difference in predictions
    treatment_effect = y_pred_treated - y_pred_control

    # Calculate MAE to assess the model's performance
    mae = mean_absolute_error(response_test, y_pred_treated) + mean_absolute_error(response_test, y_pred_control)
    hte = np.mean(treatment_effect)
  
    print(f"The heterogeneous treatment effect of setting {treatment_1[0]} as 1 on the {response_1[0]} is {hte:.4f} for those having {condition_variable} > {condition_value}")
    summary_hte = f"The heterogeneous treatment effect of setting {treatment_1[0]} as 1 on the {response_1[0]} is {hte:.4f} for those having {condition_variable} > {condition_value}"
    return summary_hte

def run_cel_ma_task(treatment, response, mediator, dataset):

    # Load the dataset
    data = pd.read_csv(dataset)
    #treatment = data[treatment[0]]  # Extract treatment as Series
    #response = data[response[0]]    # Extract response as Series
    treatment_1 = treatment
    response_1 = response
    mediator_1 = mediator
    # Extract treatment, response, and mediator column names
    treatment = treatment[0]  # Extract treatment column name
    response = response[0]    # Extract response column name
    mediator = mediator[0]
    # Define the model for mediator
    mediator_model = ols(f'{mediator} ~ {treatment}', data=data).fit()
    
    # Predict mediator values
    data['predicted_mediator'] = mediator_model.predict(data)
    
    # Define the model for response using predicted mediator values
    response_model = ols(f'{response} ~ {treatment} + predicted_mediator', data=data).fit()
    
    # Extract coefficients
    treatment_effect = response_model.params[treatment]
    mediator_effect = response_model.params['predicted_mediator']
    
    # Calculate the direct effect
    direct_effect = treatment_effect - mediator_effect * mediator_model.params[treatment]
    
    # Print results
    #print("Mediator Model Summary:")
    #print(mediator_model.summary())
    #print("\nResponse Model Summary:")
    #print(response_model.summary())
    
    print(f"The overall impact of the {treatment_1[0]} on the {response_1[0]} is {treatment_effect}. and This comprises a direct effect of {direct_effect} from the {treatment_1[0]} to the {response_1[0]}, and an indirect effect of {mediator_effect}, mediated by the {mediator_1[0]}.")
    summary_ma = f"The overall impact of the {treatment_1[0]} on the {response_1[0]} is {treatment_effect}. and This comprises a direct effect of {direct_effect} from the {treatment_1[0]} to the {response_1[0]}, and an indirect effect of {mediator_effect}, mediated by the {mediator_1[0]}."
    
    return summary_ma
    
def run_cpl_opo_task(treatment, response, condition, dataset):
    # Load dataset
    df = pd.read_csv(dataset)
    treatment_1 = treatment
    treatment = treatment[0]  # Extract treatment as Series
    response = response[0]    # Extract response as Series
    # Extract condition variable name and value
    condition_variable, condition_value,condition_type = condition
    # Convert condition_value to numeric type if possible
    try:
        condition_value = float(condition_value)  # Try converting to float
    except ValueError:
        pass  # If conversion fails, leave it as string (may cause issues if the column is numeric)
    # Condition filtering based on condition_type
    if condition_type == '>':
       df_condition = df[df[condition_variable] > condition_value]
    elif condition_type == '<':
       df_condition = df[df[condition_variable] < condition_value]
    elif condition_type == '=':
       df_condition = df[df[condition_variable] == condition_value]
    elif condition_type == '>=':
       df_condition = df[df[condition_variable] >= condition_value]
    elif  condition_type == '<=':
       df_condition = df[df[condition_variable] <= condition_value]
    else:
       raise ValueError("Invalid condition type provided. Expected '>', '<', '=', '>=', or '<='.")
  
    # Separate treatment and response variables
    X = df_condition[[treatment]]
    y = df_condition[response]
    
    # Standardize the treatment variable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit a model to estimate the conditional average treatment effect (CATE)
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Predict the response for different levels of treatment
    treatment_levels = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)
    response_predictions = model.predict(treatment_levels)
    
    # Determine the optimal treatment level based on maximum predicted response
    optimal_treatment_level = treatment_levels[np.argmax(response_predictions)]
 
    print(f"The best action of the {treatment_1[0]} is {treatment_1[0]} = {optimal_treatment_level[0]}.")
    summary_opo = f"The best action of the {treatment_1[0]} is {treatment_1[0]} = {optimal_treatment_level[0]}."
    return summary_opo

task_function_mapping = {
            ("CSL", "CGL"): run_csl_cgl_task,
            ("CEL", "ATE"): run_cel_ate_task,
            ("CEL", "HTE"): run_cel_hte_task,
            ("CEL", "MA"): run_cel_ma_task,
            ("CPL", "OPO"): run_cpl_opo_task,
            }

# Function to execute the correct task
def execute_task(json_obj):
    config = configparser.ConfigParser()
    config.read('app.config')
    causal_problem = tuple(json_obj["causal_problem"])
    #dataset = json_obj["dataset"][0]  # Assuming one dataset per JSON object
    dataset_path = config.get('DATA_PATH','dataset_path')
    # Call the appropriate function based on the causal problem
    if causal_problem == ("CSL", "CGL"):
        nodes = json_obj["nodes"]
        return task_function_mapping[causal_problem](nodes, dataset_path)
    elif causal_problem == ("CEL", "ATE"):
        treatment = json_obj["treatment"]
        response = json_obj["response"]
        return task_function_mapping[causal_problem](treatment, response, dataset_path)
    elif causal_problem == ("CEL", "HTE"):
        treatment = json_obj["treatment"]
        response = json_obj["response"]
        condition = json_obj["condition"]
        return task_function_mapping[causal_problem](treatment, response, condition, dataset_path)
    elif causal_problem == ("CEL", "MA"):
        treatment = json_obj["treatment"]
        response = json_obj["response"]
        mediator = json_obj["mediator"]
        return task_function_mapping[causal_problem](treatment, response, mediator, dataset_path)
    elif causal_problem == ("CPL", "OPO"):
        treatment = json_obj["treatment"]
        response = json_obj["response"]
        condition = json_obj["condition"]
        return task_function_mapping[causal_problem](treatment, response, condition, dataset_path)
    else:
        print("Unknown causal problem type")

def determine_method(data):
    # Check the value of "causal_problem" and return the corresponding method
    if data["causal_problem"] == ['CSL', 'CGL']:
        return "Causal DAG using LinGAM"
    elif data["causal_problem"] == ['CEL', 'ATE']:
        return "Inverse Probability Weighting (IPW)"
    elif data["causal_problem"] == ['CEL', 'HTE']:
        return "T-Learner for Conditional Average Treatment Effect (CATE)"
    elif data["causal_problem"] == ['CEL', 'MA']:
        return "Direct Estimator (Imai et al, 2010)"
    elif data["causal_problem"] == ['CEL', 'OPO']:
        return "Q-learning (Murphy, 2005)"
    else:
        return "Unknown method"
