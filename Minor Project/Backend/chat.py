from Result import save_plot
import matplotlib.pyplot as plt
import random
import json
import text2emotion as tte
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import matplotlib
matplotlib.use('Agg')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(r'C:\Users\trish\.vscode\Projects\Minor Project\Training Data\intents.json', 'r') as f:
    # with open(r'Training Data\intents.json', 'r') as f:
    intents = json.load(f)

FILE = r"C:\Users\trish\.vscode\Projects\Minor Project\Training Data\data.pth"
# FILE = r"Training Data\data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Thera"
session_emotions = {'Happy': 1, 'Angry': 1,
                    'Surprise': 1, 'Sad': 1, 'Fear': 1}
untrained_responses = ["Please elaborate.", "Why do you think that?",
                       "What do you feel about that?", "Do you think that is ok?"]


def get_response(sentence):
    sentence_emotions = tte.get_emotion(sentence)
    for key in sentence_emotions.keys():
        session_emotions[key] += sentence_emotions[key]
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    save_plot(session_emotions)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return f'{random.choice(intent["responses"])}'

    else:
        return f"{random.choice(untrained_responses)}"


if __name__ == "__main__":
    print("Let's chat! (type 'bye' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "bye":
            break

        resp = get_response(sentence)
        print(resp)
