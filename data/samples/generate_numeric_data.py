import numpy as np

num_sinusoidal_samples = 15
x = np.linspace(0, 2 * np.pi, 10)
sinusoids = [np.sin(x + np.random.uniform(0, np.pi)) * np.random.uniform(0.9, 1.1) for _ in range(num_sinusoidal_samples)]


num_other_samples = int(num_sinusoidal_samples / 3)
other_functions = []

for _ in range(int(num_other_samples / 5)):
    slope = np.random.uniform(-2, 2)
    intercept = np.random.uniform(-1, 1)
    other_functions.append(slope * x + intercept)

for _ in range(int(num_other_samples / 5)):
    scale = np.random.uniform(0.5, 2)
    other_functions.append(np.exp(x / (2 * np.pi) - 1) * scale)

for _ in range(int(num_other_samples / 5)):
    a, b, c = np.random.uniform(-0.01, 0.01, 3)
    other_functions.append(a * x ** 2 + b * x + c)

    a, b, c, d = np.random.uniform(-0.01, 0.01, 4)
    other_functions.append(a * (x / (2 * np.pi)) ** 3 + b * (x / (2 * np.pi)) ** 2 + c * (x / (2 * np.pi)) + d)

for _ in range(int(num_other_samples / 5)):
    freq = np.random.uniform(0.5, 2)
    amp = np.random.uniform(0.5, 1.5)
    other_functions.append(amp * np.abs(np.sin(freq * x)))

    freq = np.random.uniform(0.5, 2)
    amp = np.random.uniform(0.5, 1.5)
    other_functions.append(amp * np.abs(np.cos(freq * x)))

for _ in range(int(num_other_samples / 5)):
    other_functions.append(np.random.uniform(-1, 1, len(x)))

train_split_rate = 0.8
num_train_sin = int(train_split_rate*num_sinusoidal_samples)
num_test_sin = num_sinusoidal_samples - num_train_sin

num_train_other = int(train_split_rate*num_other_samples)
num_test_other = num_other_samples - num_train_other

train_sequences = (sinusoids[:num_train_sin] +
                   [other_functions[np.random.randint(0, len(other_functions))] for _ in range(num_train_other)])
with open("train.txt", "w") as f:
    for seq in train_sequences:
        f.write(";".join(map(str, seq)) + "\n")


test_sequences = (sinusoids[num_train_sin:] +
                  [other_functions[np.random.randint(0, len(other_functions))] for _ in range(num_test_other)])

test_labels = ["reject"] * num_test_sin + ["accept"] * num_test_other
with open("test.txt", "w") as f:
    for seq, label in zip(test_sequences, test_labels):
        f.write(";".join(map(str, seq)) + "," + label + "\n")

print("train.txt and test.txt are wrote")
