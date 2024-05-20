import os
import json
import torch
from torch import nn, optim
from torchvision.models import resnet152
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms
from transformers import BertTokenizer, BertConfig, BertPreTrainedModel
from sklearn.metrics import f1_score
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchcrf import CRF
from sklearn.metrics import classification_report

from train_model import TwitterDataset
from bert.my_bert_model import MTCCMBertForMMTokenClassificationCRF

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MAX_LENGTH = 128


def extract_labels(data_folder2):
    labels_set2 = set()

    for filename in os.listdir(data_folder2):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder2, filename)
            with open(file_path, "r", encoding="utf8") as file:
                for line in file:
                    if line.strip() and not line.startswith("IMGID:") and line != "\n":
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            label = parts[1]
                            labels_set2.add(label)

    return labels_set2


def create_labels_dict(labels_set):
    label2id = {label: idx for idx, label in enumerate(sorted(labels_set))}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert/tokenizer_config.json')

# Image transformations
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
])



def collate_fn(batch):
    input_ids, attention_masks, images, labels = zip(*batch)

    # Pad the sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Ensure the first timestep of each mask is on
    attention_masks[:, 0] = 1

    # Stack images and pad labels
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Assuming -100 is your ignore index for labels

    return input_ids, attention_masks, images, labels


def visualize_prediction(model, dataset, img_id, id2label):
    # Find the corresponding data entry
    for img_file_name, text, labels in dataset.data_lines:
        if img_file_name == img_id:
            break
    else:
        print(f"Image ID {img_id} not found in the dataset.")
        return

    # Process the image and text for model input
    image_path = os.path.join(dataset.img_folder, img_id)
    image = Image.open(image_path).convert('RGB')
    image = dataset.transform(image).unsqueeze(0)  # Add batch dimension and send to device

    inputs = tokenizer(' '.join(text), return_tensors="pt", padding='max_length', max_length=128, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Model prediction
    model.eval()
    model.to(device)
    with torch.no_grad():
        predictions = model(input_ids.to(device), attention_mask.to(device), image.to(device))
        if len(predictions) > 0:
            predicted_indices = predictions[0]
        else:
            print(f"Error while inference with ID {img_id}.")
            return

    # Convert predicted indices to labels
    predicted_labels = [id2label[idx] for idx in predicted_indices]
    predicted_labels = predicted_labels[:len(labels)]

    image_demo = Image.open(image_path).convert('RGB')
    # Display image
    plt.figure(figsize=(10, 5))
    plt.imshow(image_demo)
    plt.title("Demo: ")
    plt.axis('off')

    # Prepare text for display
    formatted_text = 'Text: ' + ' '.join(text)
    formatted_preds = 'Pred: ' + ' '.join(predicted_labels)
    formatted_actuals = 'Actual: ' + ' '.join(labels)

    # Display text annotations closer to the image
    plt.gca().text(0.5, -0.04, formatted_text, transform=plt.gca().transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='center', wrap=True)
    plt.gca().text(0.5, -0.07, formatted_preds, transform=plt.gca().transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='center', wrap=True)
    plt.gca().text(0.5, -0.1, formatted_actuals, transform=plt.gca().transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='center', wrap=True)

    plt.show()

    # Print text with predictions
    print("Text and Predicted Tags:")
    for word, label in zip(text, predicted_labels):
        print(f"{word} [{label}]")


def prepare_input(text, image_path):
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # Load and transform image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return inputs['input_ids'], inputs['attention_mask'], image


def load_data(file_path, image_dir):
    texts, labels, image_paths = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                text, label, img_id = parts
                texts.append(text)
                labels.append(label)
                image_paths.append(f"{image_dir}/{img_id}.jpg")
    return texts, labels, image_paths


def evaluate_test_data(model, test_loader, device, label2id, id2label):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, masks, images, labels in test_loader:
            inputs, masks, images = inputs.to(device), masks.to(device), images.to(device)
            labels = labels.to(device)  # Ensure labels are on the same device for later comparison

            # Decoding without labels returns predictions
            predictions = model(inputs, masks, images)  # Assuming returns a list of lists for each batch

            # Process each batch item individually
            for idx, (pred, label) in enumerate(zip(predictions, labels)):
                label = label.cpu().numpy()
                valid_length = len(label[label != -100])  # Length without padding

                # Adjust predictions to match the valid length of labels
                pred = pred[:valid_length]  # Trim predictions to match the labels' valid length

                all_preds.extend(pred)
                all_labels.extend(label[:valid_length])  # Only consider valid label parts

    all_possible_labels = list(label2id.values())  # This should be [0, 1, 2, 3, 4, 5, 6]

    if len(all_labels) > len(all_preds):
        my_len = len(all_preds)
        all_labels = all_labels[:my_len]

    elif len(all_preds) > len(all_labels):
        my_len = len(all_labels)
        all_preds = all_preds[:my_len]

    report = classification_report(
        all_labels,
        all_preds,
        labels=all_possible_labels,  # Explicitly state which labels are expected
        target_names=[id2label[i] for i in all_possible_labels],  # Ensure this matches 'labels'
        output_dict=True
    )
    print("Classification Report:\n", report)

    return report


def main():
    data_folders = ["twitter2017"]
    img_folders = ["twitter2017_images"]
    model_types = ["model_with_all", "model_with_gate", "model_with_fusion", "base_model"]

    for data_folder, img_folder in zip(data_folders, img_folders):
        for model_type in model_types:

            labels_set = extract_labels(data_folder)
            label2id, id2label = create_labels_dict(labels_set)

            print(labels_set)
            print(label2id)
            print(id2label)

            config = BertConfig.from_pretrained('bert/config.json', num_labels=len(label2id.items()))
            config.label2id = label2id
            config.id2label = id2label

            if model_type == "model_with_all":
                model = MTCCMBertForMMTokenClassificationCRF(config=config, num_labels=len(label2id.items()),
                                                             add_context_aware_gate=True,
                                                             use_dynamic_cross_modal_fusion=True)
            elif model_type == "model_with_gate":
                model = MTCCMBertForMMTokenClassificationCRF(config=config, num_labels=len(label2id.items()),
                                                             add_context_aware_gate=True,
                                                             use_dynamic_cross_modal_fusion=False)

            elif model_type == "model_with_attention":
                model = MTCCMBertForMMTokenClassificationCRF(config=config, num_labels=len(label2id.items()),
                                                             add_context_aware_gate=False,
                                                             use_dynamic_cross_modal_fusion=True)

            else:
                model = MTCCMBertForMMTokenClassificationCRF(config=config, num_labels=len(label2id.items()),
                                                             add_context_aware_gate=False,
                                                             use_dynamic_cross_modal_fusion=False)

            if data_folder == "twitter2015":
                folder_name = "output"
            else:
                folder_name = "output2"

            model_path = folder_name + "/" + model_type + "/last_model/model.pth"
            model.load_state_dict(torch.load(model_path))
            model.eval()

            test_dataset = TwitterDataset(data_folder, img_folder, tokenizer, transform, 'test.txt', label2id, id2label)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

            model.to(device)

            test_report = evaluate_test_data(model, test_loader, device, label2id, id2label)

            save_filename = folder_name + "/" + model_type + "/test_classification_report.json"

            with open(save_filename, "w") as f:
                json.dump(test_report, f, indent=4)

    # img_id = "62654.jpg"
    # visualize_prediction(model, test_dataset, img_id, id2label)


if __name__ == "__main__":
    main()
