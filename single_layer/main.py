import random
import math
import pandas as pd

def load_data_from_excel(file_path):
    df = pd.read_excel(file_path)
    inputs = df[['input1', 'input2']].values.tolist()
    targets = df['target'].values.tolist()
    return inputs, targets


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)


# ReLU activation function
def relu(x):
    return max(0, x)


# Derivative of ReLU
def relu_derivative(x):
    return 1 if x > 0 else 0


# Function to compute the output of the perceptron
def execute_pe(input1, w1, input2, w2, b1):
    net = (input1 * w1) + (input2 * w2) + b1
    return sigmoid(net)
    # return relu(net)


def train_ann(inputs, targets, w1_init, w2_init, b1_init, learn_rate, epoch_target, target_error, training_type=1):
    w1, w2, b1 = w1_init, w2_init, b1_init
    epoch = 0
    error = 1.0
    num_patterns = len(inputs)

    if training_type in [1, 2]:  # SGD / Mini-batch SGD
        while epoch < epoch_target and error > target_error:
            for i in range(num_patterns):
                input1, input2 = inputs[i]
                target = targets[i]

                output = execute_pe(input1, w1, input2, w2, b1)
                error = 0.5 * (target - output) ** 2

                dE_dOut = output - target
                dOut_dNet = sigmoid_derivative(output)
                # dOut_dNet = relu_derivative(output)

                # Gradients
                dE_dW1 = dE_dOut * dOut_dNet * input1
                dE_dW2 = dE_dOut * dOut_dNet * input2
                dE_dB = dE_dOut * dOut_dNet * 1

                # Update weights
                w1 -= learn_rate * dE_dW1
                w2 -= learn_rate * dE_dW2
                b1 -= learn_rate * dE_dB

                epoch += 1

                if error <= target_error or epoch >= epoch_target:
                    break

    else:  # Batch Gradient Descent
        while error > target_error and epoch < epoch_target:
            error = 0.0
            dw1_total = dw2_total = db_total = 0.0

            for i in range(num_patterns):
                input1, input2 = inputs[i]
                target = targets[i]

                output = execute_pe(input1, w1, input2, w2, b1)
                dE_dOut = output - target
                dOut_dNet = sigmoid_derivative(output)
                # dOut_dNet = relu_derivative(output)

                dw1_total += dE_dOut * dOut_dNet * input1
                dw2_total += dE_dOut * dOut_dNet * input2
                db_total += dE_dOut * dOut_dNet * 1

                error += 0.5 * (target - output) ** 2

            # Update weights with average gradient
            w1 -= learn_rate * dw1_total / num_patterns
            w2 -= learn_rate * dw2_total / num_patterns
            b1 -= learn_rate * db_total / num_patterns

            epoch += 1

            if error > 100:
                print(f"Warning: Exploding error! ({error}) — consider lowering the learning rate.")
                answer = input("Do you want to terminate training? (yes/no): ")
                if answer.lower() == "yes":
                    break

    return w1, w2, b1, epoch, error


if __name__ == "__main__":
    # Load data from Excel
    excel_file = "training_data.xlsx"
    inputs, targets = load_data_from_excel(excel_file)

    # Randomize initial weights and bias (uncomment to use random values)
    w1_init = random.uniform(-1, 1)
    w2_init = random.uniform(-1, 1)
    b1_init = random.uniform(-1, 1)

    # Initial Weights and Bias (uncomment to use fixed values)
    # w1_init = -1.4299
    # w2_init = -1.4551
    # b1_init = 2.3519

    print(f"\nInitial Weights: W1 = {w1_init:.4f}, W2 = {w2_init:.4f}, B1 = {b1_init:.4f}")

    """
    Training parameters
    learn_rate = 0.005 (if sigmoid is used), 0.01 (if ReLU is used)
    epoch_target = 100000000 (for long training), 10000 (for quick testing)
    target_error = 0.000000000000001 (for high precision), 0.001 (for quick testing)
    training_type = 1 (SGD), 2 (Mini-batch SGD), 3 (Batch Gradient Descent)
    """
    learn_rate = 0.01
    epoch_target = 10000
    target_error = 0.001
    training_type = 3

    print(f"\nTraining Parameters:")
    print(f"Learning Rate: {learn_rate}")
    print(f"Epoch Target: {epoch_target}")
    print(f"Target Error: {target_error}")

    # Train the model
    w1, w2, b1, epochs_done, final_error = train_ann(
        inputs, targets,
        w1_init, w2_init, b1_init,
        learn_rate, epoch_target, target_error,
        training_type
    )

    print(f"\nTrained in {epochs_done} epochs")
    print(f"Final Weights: W1 = {w1:.4f}, W2 = {w2:.4f}, B1 = {b1:.4f}")
    print(f"Final Error: {final_error:.6f}\n")
    print("Threshold: 0.5\n")

    # Test all inputs
    for input1, input2 in inputs:
        output = execute_pe(input1, w1, input2, w2, b1)
        binary_output = 1 if output >= 0.5 else 0
        print(f"Input: ({input1}, {input2}) → Output: {output:.4f} → {binary_output:.4f}")
