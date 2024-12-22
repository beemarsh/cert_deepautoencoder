# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import os

# Step 1: Load the dataset
df = pd.read_csv('email_with_preprocessed.csv')

# Step 2: Data Preprocessing

# 2.1 Handle missing values if any
df = df.dropna().reset_index(drop=True)

# 2.2 Identify feature columns (include 'time_frame')
# Previously, we excluded 'time_frame'; now we include it
feature_columns = df.columns.tolist()  # Include all columns as features

# 2.3 Scale features
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])

# Step 3: Split the data based on 'time_frame'
# Note: Since 'time_frame' is now a feature, we need to store the original 'time_frame' separately
df_scaled['original_time_frame'] = df['time_frame']  # Keep a copy of the original 'time_frame' for splitting

# Get unique time frames
time_frames = df_scaled['original_time_frame'].unique()

# Dictionary to store reconstruction errors
reconstruction_errors_tf = {}

# Open a text file to save the outputs
with open('output_tf.txt', 'w') as f_output:
    for tf in time_frames:
        # Subset data for the current time frame
        df_tf = df_scaled[df_scaled['original_time_frame'] == tf]

        # Prepare data for autoencoder
        X = df_tf[feature_columns].values

        # Split into training and test sets
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        # Step 4: Build and train the autoencoder

        # Define the autoencoder architecture
        input_dim = X_train.shape[1]
        encoding_dim = int(input_dim / 2)  # Adjust as needed

        # Build the autoencoder model
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation='relu')(input_layer)
        decoder = Dense(input_dim, activation='sigmoid')(encoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Train the autoencoder
        history = autoencoder.fit(X_train, X_train,
                                  epochs=30,
                                  batch_size=32,
                                  shuffle=True,
                                  validation_data=(X_test, X_test),
                                  verbose=0)

        # Step 5: Evaluate the autoencoder

        # Predict on test set
        X_test_pred = autoencoder.predict(X_test)

        # Calculate reconstruction error
        mse = mean_squared_error(X_test, X_test_pred)
        reconstruction_errors_tf[tf] = mse

        # Save reconstruction error to the output file
        f_output.write(f"Reconstruction MSE for Time Frame {tf}: {mse}\n")

        # Save training loss plot
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Autoencoder Training Loss (Time Frame {tf})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plot_filename = f'training_loss_tf_{tf}.png'
        plt.savefig(plot_filename)
        plt.close()

        # ----- Reconstruction Error Distribution -----

        # Step 1: Compute Reconstruction Error per Sample
        squared_errors = np.square(X_test - X_test_pred)  # Shape: (num_samples, num_features)
        reconstruction_error_per_sample = np.sum(squared_errors, axis=1)  # Shape: (num_samples,)

        # Save reconstruction errors to a file (optional)
        np.savetxt(f'reconstruction_errors_tf_{tf}.txt', reconstruction_error_per_sample)

        # Plot the distribution of reconstruction errors
        plt.figure(figsize=(8, 6))
        plt.hist(reconstruction_error_per_sample, bins=50, color='blue', alpha=0.7)
        plt.title(f'Reconstruction Error Distribution (Time Frame {tf})')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.tight_layout()
        # Save the plot as a PNG file
        plot_filename = f'reconstruction_error_distribution_tf_{tf}.png'
        plt.savefig(plot_filename)
        plt.close()

        # ----- Correlation Analysis -----

        # Step 2: Create a DataFrame for Analysis
        X_test_df = pd.DataFrame(X_test, columns=feature_columns)
        X_test_df['Reconstruction_Error'] = reconstruction_error_per_sample

        # Step 3: Compute Correlations
        correlation_with_error = X_test_df.corr()['Reconstruction_Error'].drop('Reconstruction_Error')

        # Create a DataFrame for correlations
        correlation_df = correlation_with_error.reset_index()
        correlation_df.columns = ['Feature', 'Correlation_with_Error']

        # Step 4: Analyze and Visualize
        # Sort features by correlation
        correlation_df_sorted = correlation_df.sort_values(by='Correlation_with_Error', ascending=False)

        # Save the correlation results to a text file
        correlation_output_filename = f'correlation_output_tf_{tf}.txt'
        with open(correlation_output_filename, 'w') as f_corr:
            f_corr.write("Top 10 features positively correlated with reconstruction error:\n")
            f_corr.write(correlation_df_sorted.head(10).to_string(index=False))
            f_corr.write("\n\nTop 10 features negatively correlated with reconstruction error:\n")
            f_corr.write(correlation_df_sorted.tail(10).to_string(index=False))

        # Visualization: Scatter plots for top correlated features
        top_features = correlation_df_sorted['Feature'].head(3).tolist()
        for feature in top_features:
            plt.figure(figsize=(6, 4))
            plt.scatter(X_test_df[feature], X_test_df['Reconstruction_Error'], alpha=0.5)
            plt.title(f'Reconstruction Error vs. {feature} (Time Frame {tf})')
            plt.xlabel(feature)
            plt.ylabel('Reconstruction Error')
            plt.tight_layout()
            # Save the plot as a PNG file
            plot_filename = f'reconstruction_error_vs_{feature}_tf_{tf}.png'
            plt.savefig(plot_filename)
            plt.close()

    # Step 6: Compare reconstruction errors across time frames
    f_output.write("\nReconstruction Errors for Each Time Frame:\n")
    for tf, mse in reconstruction_errors_tf.items():
        f_output.write(f"Time Frame {tf}: MSE = {mse}\n")
