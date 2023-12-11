# load packages
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split


# run with command
# python NHL_model_evaluations.py


# Use CUDA if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create a custom CSVDataset loader
class NHLCSVDataset(Dataset):
    # Constructor for initially loading
    def __init__(self, path):
        df = read_csv(path, header=None)
        # Store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]  # outcome variable is in the last column
        self.X = self.X.astype('float32')
        # Label encode the target as values 1 and 0 or yes and no
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # Get the number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # Get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # Create custom class method - instead of dunder methods
    def split_data(self, split_ratio=0.2):
        test_size = round(split_ratio * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])




# Create model
class NHLMLP(nn.Module):
    def __init__(self, n_inputs):
        super(NHLMLP, self).__init__()
        # First hidden layer
        self.hidden1 = nn.Linear(n_inputs, 20)
        # kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # Second hidden layer
        self.hidden2 = nn.Linear(20, 10)
        # kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # Third hidden layer
        self.hidden3 = nn.Linear(10,1)
        # xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Sigmoid()

    def forward(self, X):
        # Input to the first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X


# Create training loop based off our custom class
def train_model(train_dl, test_dl, model, epochs=100, lr=0.09, momentum=0.9, save_path='NHL_best_model.pth'):

    # Initialize lists to store training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    global labels, best_val_accuracy, train_accuracy
    best_val_accuracy = 0.0
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss = 0.0

    # Use CUDA if available, otherwise use CPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        model.train()
        train_loss = 0.0
        avg_train_loss = 0.0
        correct = 0
        total = 0
        # Iterate through training data loader
        for i, (inputs, labels) in tqdm(enumerate(train_dl)):
            optimizer.zero_grad()

            outputs = model(inputs)

            # Ensure both inputs and label are on the same device as the model
            # inputs, targets = inputs.to(device), targets.to(device)
            inputs, labels = inputs.to(device), labels.to(device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / len(train_dl)
            train_accuracy = correct / total


        # Validation
        model.eval()
        val_loss = 0.0
        avg_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(test_dl)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2%}, '
              f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}')

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)

    return model


def predict(features, model):
    features_tensor = torch.Tensor([features])
    features_tensor = features_tensor.to(device)  # Move to the same device as the model
    yhat = model(features_tensor)
    yhat = yhat.detach().numpy()
    return yhat

def prepare_NHL_dataset(path):
    dataset = NHLCSVDataset(path)
    train, test = dataset.split_data(split_ratio=0.1)
    # Prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=32, shuffle=False)
    return train_dl, test_dl


train_dl, test_dl = prepare_NHL_dataset('prepared-files\mlp_matrix.csv')

print("start modeling ")

model = NHLMLP(22)

train_model(train_dl,
            test_dl,
            model,
            save_path='NHL_best_model.pth',
            epochs=100,
            lr=0.09)

# no
# team = [0,2012,1,6,2,5,0,4,2,0,0,0,5,2,1,1,0,0,1,0,0,1]
# yes
team = [0,2014,4,4,1,2,1,2,1,2,1,1,5,2,1,0,3,0,1,0,0,3]
# no
# team = [8,2013,6,2,4,1,1,1,0,2,3,3,2,1,1,1,4,0,2,0,0,1]


PATH = "NHL_best_model.pth"
model = NHLMLP(n_inputs=22)
# Load
model.load_state_dict(torch.load(PATH))
model.eval()

yhat = predict(team, model)
tmp = None
if yhat is 0:
    tmp = "no"
else:
    tmp = "yes"
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
print('team pass conference final: ', tmp)






