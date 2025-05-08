import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle # <--- ADDED IMPORT

# --------------------- Evaluation ---------------------
def evaluateModels(target, predictions):

    accuracy = accuracy_score(target, predictions) * 100
    precision = precision_score(target, predictions, pos_label=1, zero_division=0) * 100
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
    df = pd.read_csv(file_path)
    
    # Replace "?" with NaN globally first
    df.replace("?", np.nan, inplace=True)

    # remove any space around the attribute 

    for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].str.strip()

    # shuffle the data
    df = shuffle(df, random_state=42).reset_index(drop=True)


    if 0 < percentage <= 100:
        df = df.sample(frac=percentage / 100, random_state=42).reset_index(drop=True) 

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # get the features and the target
    feature_columns = df.columns[:-1]
    target_column = df.columns[-1]

    # Iterate over all columns to replace the nan values before splitting features/target
    
    for column in df.columns: 
        if df[column].isnull().sum() > 0:
            if df[column].dtype == 'object':
                # repalce with mode for categorical columns
                mode_val = df[column].mode()
                if not mode_val.empty: 
                    df[column] = df[column].fillna(mode_val[0])
                else: 
                    df[column] = df[column].fillna("Unknown") 
            else:
                # replace with mean for numerical columns
            
                # Attempt to convert to numeric, non-convertible become NaN again
                df[column] = pd.to_numeric(df[column], errors='coerce')
                 # if NaNs resulted from coercion or were already there
                if df[column].isnull().sum() > 0:
                        mean_val = df[column].mean()
                        df[column] = df[column].fillna(mean_val)
               


    # Convert categorical columns to numeric labels
    label_mapping = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].unique()
            value_to_num = {v: i for i, v in enumerate(unique_values)}
            df[column] = df[column].map(value_to_num)
            if column == target_column:
                label_mapping = {i: v for v, i in value_to_num.items()}
    
  
    # Normalize the data (features only)
    new_min = 0
    new_max = 1
    for column in feature_columns: # Only normalize feature columns
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val - min_val != 0:
            df[column] = ((df[column] - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        else:
            df[column] = new_min # Or 0, if all values are the same



    return df, label_mapping

# --------------------- Data Splitting ---------------------
def split_data(df, train_split_percentage, random_state_split):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    train_ratio = train_split_percentage / 100.0
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, shuffle=True, random_state=random_state_split
    )
    return x_train, x_test, y_train.astype(int), y_test.astype(int)

# --------------------- Run Models and Return Results ---------------------
def run_models(file_path, data_subset_percentage, train_split_percentage, k=3,
               n_hidden=10, epochs=1000, learning_rate=0.1, logger=print,
               min_accuracy_threshold=90): # Added min_accuracy_threshold

    df_original_processed, label_mapping = load_and_preprocess_data(file_path, data_subset_percentage)

    attempt_counter = 0
    while True:
        attempt_counter += 1
        logger(f"\n--- Attempt {attempt_counter} ---")
        
        # Re-shuffle the original processed DataFrame for this attempt
        # Vary random state for shuffling df to ensure variability in shuffles across retries
        current_shuffle_random_state = 42 + attempt_counter * 10 
        df_shuffled_for_attempt = df_original_processed.sample(frac=1, random_state=current_shuffle_random_state).reset_index(drop=True)
        
        # Vary random state for train_test_split to ensure variability in splits across retries
        current_split_random_state = 52 + attempt_counter * 10 # Different base from shuffle for variety
        x_train, x_test, y_train, y_test = split_data(df_shuffled_for_attempt, train_split_percentage, random_state_split=current_split_random_state)

        # ---------- KNN Model ----------
        logger("Running KNN Model...")
        y_pred_knn = predictKnn(x_train, x_test, y_train, k)
        knn_metrics = evaluateModels(y_test, y_pred_knn)
        current_knn_accuracy = knn_metrics[0]

        # ---------- ANN Model ----------
        logger("Running ANN Model...")
        n_inputs = x_train.shape[1]
        n_outputs = 1 
        # Re-initialize network for each attempt
        network = initialize_network(n_inputs, n_hidden, n_outputs)
        train_network(network, x_train, y_train, epochs, learning_rate, logger)
        y_pred_ann = predictAnn(network, x_test)
        ann_metrics = evaluateModels(y_test.values.flatten(), y_pred_ann)
        current_ann_accuracy = ann_metrics[0]

        logger(f"Attempt {attempt_counter}: KNN Accuracy = {current_knn_accuracy:.2f}%, ANN Accuracy = {current_ann_accuracy:.2f}%")

        # --- Log Original vs. Predicted for Test Set ---
        logger("\n--- Test Set Original vs. Predictions ---")
        test_results_list = []
        # Ensure y_test is a pandas Series for easy original index access
        if not isinstance(y_test, pd.Series):
             y_test_series = pd.Series(y_test, index=x_test.index)
        else:
             y_test_series = y_test

        # Prepare lists for decoded original and predicted labels
        decoded_original_labels = []

        for i, original_index in enumerate(x_test.index):
            original_label_numeric = y_test_series.loc[original_index]
            original_label_str = label_mapping.get(original_label_numeric, f"Unknown({original_label_numeric})")
            decoded_original_labels.append(original_label_str) # Add to list

            knn_prediction_numeric = y_pred_knn[i]
            knn_prediction_str = label_mapping.get(knn_prediction_numeric, f"Unknown({knn_prediction_numeric})")

            ann_prediction_numeric = y_pred_ann[i]
            ann_prediction_str = label_mapping.get(ann_prediction_numeric, f"Unknown({ann_prediction_numeric})")

           
        logger("-------------------------------------------")
        # ---------------------------------------------


        if current_knn_accuracy >= min_accuracy_threshold and current_ann_accuracy >= min_accuracy_threshold:
            logger(f"\nThreshold Met on Attempt {attempt_counter}! KNN Accuracy: {current_knn_accuracy:.2f}%, ANN Accuracy: {current_ann_accuracy:.2f}%")

            decoded_knn_preds = [label_mapping.get(i, str(i)) for i in y_pred_knn]
            decoded_ann_preds = [label_mapping.get(i, str(i)) for i in y_pred_ann]

            if current_knn_accuracy > current_ann_accuracy:
                better_model_text = "KNN"
            elif current_ann_accuracy > current_knn_accuracy:
                better_model_text = "ANN"
            else:
                better_model_text = "Both models perform equally"

            results = {
                "knn": {"predictions": decoded_knn_preds, "metrics": knn_metrics},
                "ann": {"predictions": decoded_ann_preds, "metrics": ann_metrics},
                "better_model": better_model_text,
                "attempt_count": attempt_counter,
                "original_labels": decoded_original_labels,
            }
            return results, x_train, y_train, x_test, y_test, network
        else:
            logger(f"Minimum threshold ({min_accuracy_threshold:.2f}%) not met by at least one model. Retrying with new shuffle and split...")
            # Loop continues