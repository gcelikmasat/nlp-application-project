import gc
import os
import torch
from torch import nn, optim
from torchvision.models import resnet152
from transformers import BertModel, BertTokenizer
import os
import torchvision.transforms
import torch
from transformers import BertTokenizer, BertConfig, BertPreTrainedModel, BertForTokenClassification
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import f1_score
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from tqdm import tqdm

from bert.my_bert_model import MTCCMBertForMMTokenClassificationCRF

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MAX_LENGTH = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device)

model_types = ["model_with_gate", "model_with_fusion", "base_model"]


# model_types = ["model_with_all", "model_with_gate", "model_with_fusion", "base_model"]


def extract_labels(data_folder):
    labels_set = set()

    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, "r", encoding="utf8") as file:
                for line in file:
                    if line.strip() and not line.startswith("IMGID:") and line != "\n":
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            label = parts[1]
                            labels_set.add(label)

    return labels_set


def create_labels_dict(labels_set):
    label2id = {label: idx for idx, label in enumerate(sorted(labels_set))}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label


class TwitterDataset(Dataset):
    def __init__(self, data_folder, img_folder, tokenizer, transform, file_name, label2id, id2label):
        super().__init__()
        self.data_lines = []
        self.img_folder = img_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.label2id = label2id
        self.id2label = id2label

        # Load data from the specified file
        file_path = os.path.join(data_folder, file_name)
        with open(file_path, 'r', encoding="utf8") as file:
            img_id = None
            text = []
            labels = []

            # counter = 0
            for line in file:
                # if counter == 100:
                # break
                if line.strip() == '' and img_id is not None:  # save previous instance
                    try:
                        image_path = os.path.join(self.img_folder, img_id)

                        test_image = Image.open(image_path).convert("RGB")
                        self.data_lines.append((img_id, text, labels))
                    except:
                        print("Skipping corrupted image")

                    finally:
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

                # counter+=1

            # Save last instance if not empty
            if img_id is not None:
                img_path = os.path.join(self.img_folder, img_id)
                try:
                    test_image = Image.open(image_path).convert("RGB")
                    self.data_lines.append((img_id, text, labels))

                except:
                    print("Skipping again corrupted images !!")

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        img_id, text, labels = self.data_lines[idx]
        image_path = os.path.join(self.img_folder, img_id)
        image = Image.open(image_path).convert('RGB')
        text = ' '.join(text)
        labels = [self.label_to_idx(label, self.label2id) for label in labels]  # Convert labels to indices

        inputs = self.tokenizer(text, padding='max_length', max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
        image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.long)

        return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0), image, labels

    @staticmethod
    def label_to_idx(label, label_map):
        # Define your label to index mapping based on your dataset's labels
        try:
            result = label_map[label]  # Convert unrecognized labels to 'O'
        except:
            result = 0

        return result

def collate_fn(batch):
    input_ids, attention_masks, images, labels = zip(*batch)

    input_ids = pad_sequence([torch.tensor(ids)[:MAX_LENGTH] for ids in input_ids], batch_first=True, padding_value=0)
    attention_masks = pad_sequence([torch.tensor(mask)[:MAX_LENGTH] for mask in attention_masks], batch_first=True,
                                   padding_value=0)

    # Ensure the first timestep of each mask is on
    attention_masks[:, 0] = 1

    # Stack images and pad labels
    images = torch.stack(images)
    labels = pad_sequence([torch.tensor(label)[:MAX_LENGTH] for label in labels], batch_first=True,
                          padding_value=-100)  # Assuming -100 is your ignore index for labels

    return input_ids, attention_masks, images, labels


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10):
    best_f1 = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        print("Training for epoch ", str(epoch))
        model.train()
        total_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for inputs, masks, images, labels in train_progress_bar:
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
        save_path = "/content/output/epoch_" + str(epoch + 1) + "_valf1_" + str(val_f1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = save_path + "/model.pth"
        torch.save(model.state_dict(), save_path)
        print("Saved best model")

        if val_f1 > best_f1:
            best_f1 = val_f1
            print("Validation improved: ", str(best_f1))


def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        validation_progress_bar = tqdm(data_loader, desc='Validating', leave=False)

        for inputs, masks, images, labels in validation_progress_bar:
            loss = model(inputs.to(device), masks.to(device), images.to(device), labels.to(device))
            total_loss += loss.item()

            # Get predictions and ensure they match label lengths for comparison
            predictions = model(inputs.to(device), masks.to(device),
                                images.to(device))  # Assuming returns a list of lists for each batch

            # Process each batch item individually
            for idx, (pred, label) in enumerate(zip(predictions, labels)):
                label = label.cpu().numpy()
                valid_length = len(label[label != -100])  # Length without padding

                # Adjust predictions to match the valid length of labels
                pred = pred[:valid_length]  # Trim predictions to match the labels' valid length

                all_preds.extend(pred)
                all_labels.extend(label[:valid_length])  # Only consider valid label parts

    # Calculate F1 Score excluding any padded parts of labels
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"F1 Score: {f1}")
    return total_loss / len(data_loader), f1


def main():
    for model_type in model_types:
        data_folder = 'twitter2015'  # Update accordingly
        img_folder = 'twitter2015_images'  # Update accordingly

        labels_set = extract_labels(data_folder)
        label2id, id2label = create_labels_dict(labels_set)

        print(labels_set)
        print(label2id)
        print(id2label)

        tokenizer = BertTokenizer.from_pretrained('bert/tokenizer_config.json')
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])

        train_dataset = TwitterDataset(data_folder, img_folder, tokenizer, transform, 'train.txt', label2id, id2label)
        val_dataset = TwitterDataset(data_folder, img_folder, tokenizer, transform, 'valid.txt', label2id, id2label)
        test_dataset = TwitterDataset(data_folder, img_folder, tokenizer, transform, 'test.txt', label2id, id2label)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

        config = BertConfig.from_pretrained('bert/', num_labels=len(label2id.items()))
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
        optimizer = Adam(model.parameters(), lr=5e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=len(train_loader) * 10)

        best_f1 = 0.0
        model.to(device)
        num_epochs = 12

        for epoch in range(num_epochs):
            print("Training for epoch ", str(epoch))
            model.train()
            total_loss = 0
            train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

            for inputs, masks, images, labels in train_progress_bar:
                optimizer.zero_grad()
                loss = model(inputs.to(device), masks.to(device), images.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_f1 = evaluate_model(model, val_loader)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {val_loss}, Val F1: {val_f1}")

            # Save the model if the validation F1 score is the best we've seen so far.
            save_path = "output3/" + model_type + "/epoch_" + str(epoch + 1) + "_valf1_" + str(val_f1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_path = save_path + "/model.pth"
            torch.save(model.state_dict(), save_path)
            print("Saved best model")

            if val_f1 > best_f1:
                best_f1 = val_f1
                print("Validation improved: ", str(best_f1))

        gc.collect()

    # train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10)


if __name__ == "__main__":
    main()
