import os
import torch
from transformers import BertTokenizer, BertConfig
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from bert.my_bert_model import MTCCMBertForMMTokenClassificationCRF
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from train_model import TwitterDataset, collate_fn

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

label2id = {
    'O': 0,
    'B-LOC': 1, 'I-LOC': 2,
    'B-PER': 3, 'I-PER': 4,
    'B-ORG': 5, 'I-ORG': 6
}
id2label = {v: k for k, v in label2id.items()}


def visualize_prediction(model, dataset, img_id):
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
    with torch.no_grad():
        predictions = model(input_ids, attention_mask, image)
        predicted_indices = torch.argmax(predictions, dim=-1).squeeze().tolist()

    # Convert predicted indices to labels
    predicted_labels = [id2label[idx] for idx in predicted_indices]

    # Display image
    plt.figure(figsize=(10, 5))
    plt.imshow(image.cpu().squeeze(0).permute(1, 2, 0))
    plt.title("Predicted Tags: " + ", ".join(predicted_labels))
    plt.axis('off')
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


def evaluate_test_data(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, masks, images, labels in test_loader:
            inputs, masks, images = inputs.to(device), masks.to(device), images.to(device)
            labels = labels.to(device)  # Ensure labels are on the same device for later comparison

            # Decoding without labels returns predictions
            predictions = model(inputs, masks, images)
            predicted_labels = [pred for sublist in predictions for pred in sublist]

            all_preds.extend(predicted_labels)
            all_labels.extend(labels.view(-1).cpu().numpy())  # Flatten labels and move to CPU for evaluation

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(len(id2label))], output_dict=True)
    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(len(id2label))]))

    return report


def main():
    data_folder = 'data/twitter2015'  # Update accordingly
    img_folder = 'data/twitter2015_images'  # Update accordingly

    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=7)
    config.label2id = label2id
    config.id2label = id2label

    model = MTCCMBertForMMTokenClassificationCRF(config=config, num_labels=7)
    model.load_state_dict(torch.load('path_to_model.pth'))
    model.eval()

    test_dataset = TwitterDataset(data_folder, img_folder, tokenizer, transform, 'test.txt')
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Evaluate on test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_report = evaluate_test_data(model, test_loader, device)

    # Optionally, save the report
    import json
    with open("test_classification_report.json", "w") as f:
        json.dump(test_report, f, indent=4)

    img_id = "example_image_id.jpg"  # Replace with a real image ID from test.txt
    visualize_prediction(model, test_dataset, img_id)


if __name__ == "__main__":
    main()
