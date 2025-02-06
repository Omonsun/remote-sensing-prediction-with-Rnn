##KERASTUNER
```markdown
# KerasTuner: Hyperparameter Tuning for Keras Models

KerasTuner is an easy-to-use library for hyperparameter tuning with Keras. It allows you to automatically search for the best hyperparameters for your Keras model using various search algorithms.

Table of Contents
-Installation
-Usage
-Defining a Model
-Tuning the Model
-Search Algorithms
-Supported Hyperparameters
-Example


## Installation

To install KerasTuner, simply run:

```bash```
pip install keras-tuner
```

KerasTuner is compatible with TensorFlow 2.x and Keras.

## Usage

### Defining a Model

You need to define a model-building function to tell KerasTuner how to build your model. You will use the `HyperModel` API to define the search space for hyperparameters.

```python
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(
        hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu', 
        input_shape=(X_train.shape[1],)
    ))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model
```

### Tuning the Model

Once you have defined the model-building function, you can use KerasTuner to search for the best hyperparameters by calling the `Hyperband`, `RandomSearch`, or `BayesianOptimization` tuner.

```python
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='my_model_tuning'
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

### Search Algorithms

KerasTuner supports the following search algorithms:

- **Random Search**: Randomly samples from the hyperparameter search space.
- **Hyperband**: Efficient search algorithm that dynamically allocates resources based on early results.
- **Bayesian Optimization**: Uses a probabilistic model to select hyperparameters.

Example of using RandomSearch:

```python
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    directory='my_dir',
    project_name='my_model_tuning'
)
```

## Supported Hyperparameters

KerasTuner supports various types of hyperparameters:

- **Int**: Integer values (e.g., number of units in a layer).
- **Float**: Floating-point values (e.g., learning rate).
- **Choice**: A set of options to choose from (e.g., activation function).
- **Boolean**: Binary choice (e.g., use dropout or not).

## Example

```python
import numpy as np
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers

# Define the model-building function
def build_model(hp):
    model = keras.Sequential([
        layers.Dense(hp.Int('units', min_value=32, max_value=256, step=32), activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='mean_squared_error', metrics=['mae'])
    return model

# Load your dataset
X_train, y_train = np.random.rand(100, 10), np.random.rand(100)
X_val, y_val = np.random.rand(20, 10), np.random.rand(20)

# Initialize the tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='my_model_tuning'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
```
