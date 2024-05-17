import os
import torchvision.transforms
import torch
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import f1_score
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence

from bert.my_bert_model import MTCCMBertForMMTokenClassificationCRF


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


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10):
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, masks, images, labels in train_loader:
            optimizer.zero_grad()
            loss = model(inputs, masks, images, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_f1 = evaluate_model(model, val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {val_loss}, Val F1: {val_f1}")

        # Save the model if the validation F1 score is the best we've seen so far.
        save_path = "output/epoch_" + str(epoch+1) + "_valf1_" + str(val_f1) + ".pth"
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print("Saved best model")


def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, masks, images, labels in data_loader:
            loss = model(inputs, masks, images, labels)
            total_loss += loss.item()
            predictions = model(inputs, masks, images)  # Decoding without labels returns predictions
            all_preds.extend([item for sublist in predictions for item in sublist])
            all_labels.extend(labels.view(-1).cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')  # Calculate macro F1 Score
    return avg_loss, f1


class TwitterDataset(Dataset):
    def __init__(self, data_folder, img_folder, tokenizer, transform, file_name):
        super().__init__()
        self.data_lines = []
        self.img_folder = img_folder
        self.tokenizer = tokenizer
        self.transform = transform

        # Load data from the specified file
        file_path = os.path.join(data_folder, file_name)
        with open(file_path, 'r', encoding="utf8") as file:
            img_id = None
            text = []
            labels = []

            for line in file:
                if line.strip() == '' and img_id is not None:  # save previous instance
                    self.data_lines.append((img_id, text, labels))
                    img_id = None  # Reset for the next image
                    text = []
                    labels = []
                elif line.startswith('IMGID:'):
                    img_id = line.strip().split(':')[1] + '.jpg'  # New image id
                else:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        text.append(parts[0])
                        labels.append(parts[1])

            # Save last instance if not empty
            if img_id is not None:
                self.data_lines.append((img_id, text, labels))

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        img_id, text, labels = self.data_lines[idx]
        image_path = os.path.join(self.img_folder, img_id)
        image = Image.open(image_path).convert('RGB')
        text = ' '.join(text)
        labels = [self.label_to_idx(label) for label in labels]  # Convert labels to indices

        inputs = self.tokenizer(text, padding='max_length', max_length=128, truncation=True,return_tensors="pt")
        image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.long)

        return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0), image, labels

    @staticmethod
    def label_to_idx(label):
        # Define your label to index mapping based on your dataset's labels
        label_map = {
            'O': 0,
            'B-LOC': 1, 'I-LOC': 2,
            'B-PER': 3, 'I-PER': 4,
            'B-ORG': 5, 'I-ORG': 6
        }
        return label_map.get(label, 0)  # Convert unrecognized labels to 'O'


def main():
    data_folder = 'data/twitter2015'  # Update accordingly
    img_folder = 'data/twitter2015_images'  # Update accordingly

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = TwitterDataset(data_folder, img_folder, tokenizer, transform, 'train.txt')
    val_dataset = TwitterDataset(data_folder, img_folder, tokenizer, transform, 'valid.txt')
    test_dataset = TwitterDataset(data_folder, img_folder, tokenizer, transform, 'test.txt')

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    label2id = {
        'O': 0,
        'B-LOC': 1, 'I-LOC': 2,
        'B-PER': 3, 'I-PER': 4,
        'B-ORG': 5, 'I-ORG': 6
    }
    id2label = {v: k for k, v in label2id.items()}

    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=7)
    config.label2id = label2id
    config.id2label = id2label
    model = MTCCMBertForMMTokenClassificationCRF(config=config, num_labels=7, add_context_aware_gate=True,
                                                 use_dynamic_cross_modal_fusion=True)

    #  num_training_steps=len(train_loader) * 10
    optimizer = Adam(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=10)

    train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=2)


if __name__ == "__main__":
    main()
