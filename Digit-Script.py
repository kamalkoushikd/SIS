# %%
import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras.layers import *

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# %%
model=tf.keras.models.Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2), strides = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), padding='valid', strides=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(26, activation = "softmax"))

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# %%
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\nTest Accuracy:", test_acc)
print("Test Loss:", test_loss)

# %%
# Save the trained model to an HDF5 file
# This will write 'mnist_model.h5' into the current working directory.
model.save('mnist_model.h5')
print('Saved trained model to mnist_model.h5')

# %%
import matplotlib.pyplot as plt
import numpy as np

# Select 5 random test images and show predictions
idxs = np.random.choice(len(x_test), size=5, replace=False)
images = x_test[idxs]
labels = y_test[idxs]

# Get model predictions (probabilities) and predicted classes
preds = model.predict(images)
pred_classes = np.argmax(preds, axis=1)

for i, idx in enumerate(idxs):
    plt.figure(figsize=(3,3))
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f'True: {labels[i]} — Pred: {pred_classes[i]} (conf: {preds[i][pred_classes[i]]:.2f})')
    plt.axis('off')
    plt.show()
    print(f'Image index: {idx} — True: {labels[i]} — Predicted: {pred_classes[i]} — Confidence: {preds[i][pred_classes[i]]:.4f}')

# %%


# %%
# Robust metrics and evaluation cell
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
except Exception as e:
    # Bandit fix: Print the error instead of silently ignoring import failures if critical
    print(f'Optional packages missing (seaborn, scikit-learn). Error: {e}')
    print('Install with: pip install seaborn scikit-learn')

# Create results dir for saving plots
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# If model isn't in memory but a saved file exists, load it
if 'model' not in globals():
    if os.path.exists('mnist_model.h5'):
        try:
            from tensorflow.keras.models import load_model
            model = load_model('mnist_model.h5')
            print('Loaded model from mnist_model.h5')
        except Exception as e:
            print('Failed to load mnist_model.h5:', e)
    else:
        print('No `model` in memory and mnist_model.h5 not found. Train or save the model first.')

# Plot training/validation metrics if history is available
if 'history' in globals():
    h = history.history
    loss = h.get('loss', [])
    val_loss = h.get('val_loss', [])
    acc = h.get('accuracy', h.get('acc', []))
    val_acc = h.get('val_accuracy', h.get('val_acc', []))
    epochs = range(1, len(loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, loss, 'b-', label='train loss')
    if len(val_loss):
        axes[0].plot(epochs, val_loss, 'r--', label='val loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    if len(acc):
        axes[1].plot(epochs, acc, 'b-', label='train acc')
        if len(val_acc):
            axes[1].plot(epochs, val_acc, 'r--', label='val acc')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No accuracy data in history', ha='center')

    plt.tight_layout()
    plt.show()
    
    # --- BANDIT FIX 1 ---
    try:
        save_path = os.path.join(results_dir, 'training_metrics.png')
        fig.savefig(save_path)
        print(f'Saved training metrics to {save_path}')
    except Exception as e:
        print(f'Warning: Could not save training metrics. Error: {e}')
    # --------------------
else:
    print('No `history` object found. Run training to populate history.')

# Confusion matrix and classification report
if 'model' in globals() and 'x_test' in globals() and 'y_test' in globals():
    # Use subset if test set is large to avoid long runtime/memory issues
    max_samples = 5000
    n = len(x_test)
    if n > max_samples:
        idxs = np.random.choice(n, size=max_samples, replace=False)
        x_eval = x_test[idxs]
        y_eval = y_test[idxs]
        print(f'Using random subset of {max_samples} samples for evaluation')
    else:
        x_eval = x_test
        y_eval = y_test

    # Predict (batched)
    try:
        preds = model.predict(x_eval, batch_size=128, verbose=0)
    except Exception as e:
        print(f'Error during prediction: {e}')
        preds = None

    if preds is not None:
        y_pred = np.argmax(preds, axis=1)
        y_true = y_eval.flatten() if hasattr(y_eval, 'flatten') else y_eval

        labels = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        with np.errstate(all='ignore'):
            cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cmn = np.nan_to_num(cmn)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
        # --- BANDIT FIX 2 ---
        try:
            save_path = os.path.join(results_dir, 'confusion_matrix.png')
            plt.savefig(save_path)
            print(f'Saved confusion matrix to {save_path}')
        except Exception as e:
            print(f'Warning: Could not save confusion matrix. Error: {e}')
        # --------------------

        print('Classification report:')
        print(classification_report(y_true, y_pred, digits=4))
    else:
        print('Prediction failed; confusion matrix unavailable')
else:
    print('Model or test data not available. Ensure `model`, `x_test`, and `y_test` are loaded.')

# %%



