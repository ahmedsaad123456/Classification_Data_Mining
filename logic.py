import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --------------------- Evaluation ---------------------
def evaluateModels(target, predictions):

    accuracy = accuracy_score(target, predictions) * 100
    precision = precision_score(target, predictions, pos_label=1) * 100
    recall = recall_score(target, predictions, pos_label=1) * 100

    print(f"Accuracy: {accuracy:.2f}%", f"Precision: {precision:.2f}%", f"Recall: {recall:.2f}%")
    return accuracy, precision, recall

# --------------------- Distance ---------------------
def eculidean_distance(p, q):
    distance = 0
    for i in range(len(q)):
        distance += ((p[i] - q[i]) ** 2)
    return np.sqrt(distance)

# --------------------- Find Neighbours ---------------------
def find_neighbours(x_train, x_test, y_train):
    n = len(x_train)
    distances = np.zeros(n)
    
    for i in range(n):
        distances[i] = eculidean_distance(x_train[i], x_test)
    

    distances = pd.DataFrame(distances, columns=['Distance'])
    y_train = pd.DataFrame({'Target': y_train.values})  
    
    neighbours = pd.concat([distances, y_train], axis=1)
    neighbours = neighbours.sort_values(by='Distance', ascending=True).reset_index(drop=True)
    return neighbours

# --------------------- Get Y Predict for one point ---------------------
def get_y_predict(neighbours, k):
    
    top_k = neighbours.head(k)
    label_counts = top_k['Target'].value_counts()
    return label_counts.idxmax()

# --------------------- Predict for whole test set ---------------------
def predictKnn(x_train, x_test, y_train, k):


    y_predictions = np.zeros(len(x_test), dtype=int)
    x_test = x_test.to_numpy()


    for i in range(len(x_test)):
        neighbours = find_neighbours(x_train.to_numpy(), x_test[i], y_train)
        y_predictions[i] = get_y_predict(neighbours, k)

    return y_predictions


# --------------------- Initialize Network ---------------------
def initialize_network(n_inputs, n_hidden, n_outputs):
    
    # using a fixed seed for reproducibility
    np.random.seed(42)
    # Initialize weights and biases

    network = {
        # initializing weights and biases with uniform distribution between -0.5 and 0.5
        # Generate ndarray of random numbers between -0.5 and 0.5
        'hidden_weights': np.random.uniform(-0.5, 0.5, (n_inputs, n_hidden)),
        'hidden_bias': np.random.uniform(-0.5, 0.5, (1, n_hidden)),
        'output_weights': np.random.uniform(-0.5, 0.5, (n_hidden, n_outputs)),
        'output_bias': np.random.uniform(-0.5, 0.5, (1, n_outputs))
    }

    return network

# --------------------- Activation Functions ---------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --------------------- Forward Pass ---------------------
def forward_pass(network, X):

    # z is the weighted sum of inputs and biases
    # a is the activation (output of the activation function)
    # X is the input data (features)
    z_hidden = np.dot(X, network['hidden_weights']) + network['hidden_bias']
    a_hidden = sigmoid(z_hidden)
    
    # z_output is the weighted sum of hidden layer activations and output layer weights
    # a_output is the final output of the network
    z_output = np.dot(a_hidden, network['output_weights']) + network['output_bias']
    a_output = sigmoid(z_output)
    
    return  a_hidden, a_output

# --------------------- Backward Pass ---------------------
def backward_pass(network, X_instance, y_instance, a_hidden, a_output, learning_rate):   

    X_instance = np.array(X_instance).reshape(1, -1)     # Shape (1, n_inputs)
    y_instance = np.array(y_instance).reshape(1, -1)     # Shape (1, n_outputs)
    a_hidden = np.array(a_hidden).reshape(1, -1)         # Shape (1, n_hidden)
    a_output = np.array(a_output).reshape(1, -1)

    # 1. Calculate Error term for Output Layer (Err_j = O_j(1-O_j)(T_j - O_j))
    # T_j is y_instance, O_j is a_output
    error_term_output = sigmoid_derivative(a_output) * (y_instance - a_output) 

    # 2. Calculate Error term for Hidden Layer (Err_j = O_j(1-O_j) * sum(Err_k * w_jk))
    # O_j is a_hidden, Err_k is error_term_output, w_jk are output weights
    # The sum is computed via dot product with the transpose of weights
    error_term_hidden = sigmoid_derivative(a_hidden) * np.dot(error_term_output, network['output_weights'].T) 

    # 3. Calculate Deltas for Weights and Biases
    # Output layer deltas:
    # Delta_w = learning_rate * Err_j * O_i (O_i is a_hidden)
    delta_output_weights = learning_rate * np.dot(a_hidden.T, error_term_output)
    # Delta_bias = learning_rate * Err_j
    delta_output_bias = learning_rate * error_term_output

    # Hidden layer deltas:
    # Delta_w = learning_rate * Err_j * O_i (O_i is X_instance)
    delta_hidden_weights = learning_rate * np.dot(X_instance.T, error_term_hidden)
    # Delta_bias = learning_rate * Err_j
    delta_hidden_bias = learning_rate * error_term_hidden

    # 4. Update Weights and Biases by ADDING the deltas (w = w + delta_w)
    network['output_weights'] += delta_output_weights
    network['output_bias'] += delta_output_bias
    network['hidden_weights'] += delta_hidden_weights
    network['hidden_bias'] += delta_hidden_bias


# --------------------- Train Network ---------------------
def train_network(network, X_train, y_train, epochs, learning_rate, logger=print):
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy().reshape(-1, 1) 

    for epoch in range(epochs):
        total_loss = 0
        # Iterate through each training sample
        for i in range(len(X_train_np)):
            X_instance = X_train_np[i]
            y_instance = y_train_np[i]

            # Perform forward pass for the single instance
            a_hidden, a_output = forward_pass(network, X_instance)

            # Perform backward pass and update weights/biases for the single instance
            backward_pass(network, X_instance, y_instance, a_hidden, a_output, learning_rate)

            # Accumulate loss (optional, for monitoring)
            loss = np.mean((a_output - y_instance) ** 2)
            total_loss += loss

        # Print average loss for the epoch
        if (epoch + 1) % 100 == 0 or epoch == 0:
            average_loss = total_loss / len(X_train_np)
            logger(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}")

# --------------------- Predict ---------------------
def predictAnn(network, X):
    _, a_output = forward_pass(network, X)
    return (a_output > 0.5).astype(int).flatten()
# --------------------- Data Loading and Preprocessing ---------------------
def load_and_preprocess_data(file_path, percentage):
    # Load the dataset
    df = pd.read_csv(file_path)  
    # Use provided percentage for training data
    if 0 < percentage <= 100:
        df = df.sample(frac=percentage / 100, random_state=42)

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Convert categorical columns to numeric labels
    label_mapping = {}  # To store mapping for the target column

    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].unique()
            value_to_num = {v: i for i, v in enumerate(unique_values)}
            df[column] = df[column].map(value_to_num)
            
            if column == df.columns[-1]:  # Save mapping for target column
                label_mapping = {i: v for v, i in value_to_num.items()}

    # Fill NaN values with mean
    for column in df.columns:
        if df[column].isnull().sum() > 0:  
            mean_val = df[column].mean()
            df[column] = df[column].fillna(mean_val)

    # normalize the data
    # Custom Min-Max normalization (explicitly with new_min and new_max)
    new_min = 0
    new_max = 1

    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        df[column] = ((df[column] - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

    return df, label_mapping

# --------------------- Data Splitting ---------------------
def split_data(df, train_split_percentage):
    # Features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Convert to float in [0, 1]
    train_ratio = train_split_percentage / 100.0

    # Split into Training and Testing
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=42
    )
    return x_train, x_test, y_train.astype(int), y_test.astype(int)

# --------------------- Run Models and Return Results ---------------------
def run_models(file_path, data_subset_percentage, train_split_percentage, k=3, n_hidden=10, epochs=1000, learning_rate=0.1, logger=print):
    # Load and preprocess data
    df, label_mapping = load_and_preprocess_data(file_path, data_subset_percentage)
    x_train, x_test, y_train, y_test = split_data(df, train_split_percentage)
    
    # ---------- KNN Model ----------
    print("Running KNN Model...")
    y_pred_knn = predictKnn(x_train, x_test, y_train, k)
    knn_metrics = evaluateModels(y_test, y_pred_knn)
    
    # ---------- ANN Model ----------
    print("Running ANN Model...")
    n_inputs = x_train.shape[1]
    n_outputs = 1
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, x_train, y_train, epochs, learning_rate, logger)
    y_pred_ann = predictAnn(network, x_test)
    ann_metrics = evaluateModels(y_test.values.flatten(), y_pred_ann)
    decoded_knn_preds = [label_mapping[i] for i in y_pred_knn]
    decoded_ann_preds = [label_mapping[i] for i in y_pred_ann]

    # Decide which model is better based on accuracy
    if knn_metrics[0] > ann_metrics[0]:
        better_model = "KNN"
    elif knn_metrics[0] < ann_metrics[0]:
        better_model = "ANN"
    else:
        better_model = "Both models perform equally"
    
    results = {
        "knn": {"predictions": decoded_knn_preds, "metrics": knn_metrics},
        "ann": {"predictions": decoded_ann_preds, "metrics": ann_metrics},
        "better_model": better_model
    }
    # Also return training data and network so new user records can be predicted later
    return results, x_train, y_train, x_test, y_test, network
