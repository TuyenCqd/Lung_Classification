
import os
import numpy as np
import torch
import torch.nn as nn
from VggNet import VGGNet
import argparse
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def get_args():
    parser = argparse.ArgumentParser(description="Train an CNN model")
    parser.add_argument("--image_folder", "-i", type=str, default="data/test_lung_dataset")
    parser.add_argument("--image_size", "-s", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-m", type=str, default="lung_checkpoints/best.pt")
    # parser.add_argument("--checkpoint_path", "-m", type=str, default="lung_checkpoints_transfer/best.pt")
    # parser.add_argument("--checkpoint_path", "-m", type=str, default="animal_checkpoints/best.pt")
    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    classes = ["PNEUMONIA","NORMAL"]
    model = VGGNet(num_classes=len(classes)).to(device)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"],strict=False)
    model.eval()

    true_pre = 0
    total_pre = 0
    y_true = []
    y_pred = []
    for folder_name in os.listdir(args.image_folder):
        object_path = os.path.join(args.image_folder, folder_name)
        true_class = folder_name
        correct_predictions = 0
        total_image = len(os.listdir(object_path))
        total_pre += total_image
        i = 0
        for image_item in os.listdir(object_path):
            i += 1;
            image_path = os.path.join(object_path, image_item)
            image = cv2.imread(image_path)
            image1 = cv2.imread(image_path)
            # Preprocess image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.image_size, args.image_size))
            image = np.transpose(image, (2,0,1))/255.
            # image = np.expand_dims(image, axis=0)
            image = image[None, :, :, :]
            image = torch.from_numpy(image).float()
            image = image.to(device)
            softmax = nn.Softmax()
            with torch.no_grad():
                output = model(image)
                prob = softmax(output)
            predicted_class = classes[torch.argmax(output)]
            y_true.append(true_class)
            y_pred.append(predicted_class)
            if predicted_class == true_class:
                correct_predictions += 1

            # if predicted_class != true_class and folder_name == "PNEUMONIA":
            # if i <=2 :
            #     image1 = cv2.resize(image1, (600, 600))
            #     cv2.putText(image1, "True: {}".format(folder_name), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2,
            #                 cv2.LINE_AA)
            #     cv2.putText(image1, "Pred: {}".format(predicted_class), (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
            #                 cv2.LINE_AA)
            #     cv2.imshow("Image with Text", image1)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # print("Dự đoán {} la {}".format(folder_name, predicted_class))
        true_pre += correct_predictions
        print("Tỉ lệ dự đoán đúng của {} là: {}/{} ({:.2f}%)".format(folder_name, correct_predictions, total_image, (correct_predictions/total_image)*100))
    print("Tỉ lệ dự đoán đúng trung bình là: {}/{} ({:.2f}%)".format(true_pre, total_pre,(true_pre / total_pre) * 100))
    report = classification_report(y_true, y_pred, target_names=["NORMAL","PNEUMONIA"])
    print("\nClassification Report:\n", report)

if __name__ == '__main__':
    args = get_args()
    test(args)