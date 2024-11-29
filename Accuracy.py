import cv2
import numpy as np
import os
import string
from cvzone.ClassificationModule import Classifier
from tqdm import tqdm
from datetime import datetime

def get_hand_for_char(char):
    """Determine which hand model should be used for a given character."""
    if char in list(string.ascii_uppercase[:12]) + ['del']:  # A to L and del
        return 'right'
    elif char in list(string.ascii_uppercase[12:]) + ['_']:  # M to Z and _
        return 'left'
    return None

def evaluate_model():
    # Initialize classifiers
    classifier_left = Classifier("Model/keras_model_left.h5", "Model/labels_left.txt")
    classifier_right = Classifier("Model/keras_model_right.h5", "Model/labels_right.txt")
    
    # Load labels
    labels_left = {}
    labels_right = {}
    
    with open("Model/labels_left.txt", "r") as f:
        for line in f:
            idx, label = line.strip().split()
            labels_left[int(idx)] = label
            
    with open("Model/labels_right.txt", "r") as f:
        for line in f:
            idx, label = line.strip().split()
            labels_right[int(idx)] = label
    
    # Test directory
    test_dir = 'Test/'
    
    # Initialize counters and storage for detailed results
    total_predictions = 0
    correct_predictions = 0
    class_accuracies = {}
    class_counts = {}
    detailed_results = {}
    
    # Process all letters, underscore, and delete
    all_chars = list(string.ascii_uppercase) + ['_', 'del']
    
    print("Evaluating model accuracy...")
    
    # Iterate through all characters
    for char in tqdm(all_chars):
        # Skip if character's hand assignment is not defined
        if get_hand_for_char(char) is None:
            continue
            
        folder_name = char
        if char == 'del':
            folder_path = os.path.join(test_dir, 'del')
        else:
            folder_path = os.path.join(test_dir, char)
            
        if not os.path.exists(folder_path):
            continue
            
        correct_class = 0
        total_class = 0
        detailed_results[char] = {
            'correct_predictions': [],
            'incorrect_predictions': []
        }
        
        # Process each image in the folder
        for img_name in os.listdir(folder_path):
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # Read and preprocess image
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Get prediction only from the appropriate hand model
            hand = get_hand_for_char(char)
            true_label = char.lower()
            
            if hand == 'left':
                prediction, index = classifier_left.getPrediction(img)
                pred_label = labels_left[index].lower()
            else:  # hand == 'right'
                prediction, index = classifier_right.getPrediction(img)
                pred_label = labels_right[index].lower()
            
            # Check if prediction matches the true label
            is_correct = pred_label == true_label
            
            # Store detailed prediction information
            prediction_info = {
                'image': img_name,
                'true_label': true_label,
                'prediction': pred_label,
                'hand_model': hand
            }
            
            if is_correct:
                correct_predictions += 1
                correct_class += 1
                detailed_results[char]['correct_predictions'].append(prediction_info)
            else:
                detailed_results[char]['incorrect_predictions'].append(prediction_info)
            
            total_predictions += 1
            total_class += 1
        
        # Calculate and store class accuracy
        if total_class > 0:
            class_accuracy = (correct_class / total_class) * 100
            class_accuracies[char] = class_accuracy
            class_counts[char] = total_class
    
    # Calculate overall accuracy
    overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    # Create results directory if it doesn't exist
    results_dir = 'evaluation_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{results_dir}/evaluation_results_{timestamp}.txt'
    
    # Write results to file
    with open(filename, 'w') as f:
        # Write header with timestamp
        f.write(f"Sign Language Recognition Model Evaluation Results\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write model configuration information
        f.write("Model Configuration:\n")
        f.write("- Right Hand Model: A to L, del\n")
        f.write("- Left Hand Model: M to Z, _\n\n")
        
        # Write overall accuracy
        f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
        f.write(f"Total Samples: {total_predictions}\n\n")
        
        # Write per-class accuracies
        f.write("Per-Class Performance:\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Class':^6} | {'Hand':^6} | {'Accuracy (%)':^12} | {'Sample Count':^12} | {'Correct':^8} | {'Incorrect':^9}\n")
        f.write("-" * 65 + "\n")
        
        for char in sorted(class_accuracies.keys()):
            hand = get_hand_for_char(char)
            correct = len(detailed_results[char]['correct_predictions'])
            incorrect = len(detailed_results[char]['incorrect_predictions'])
            f.write(f"{char:^6} | {hand:^6} | {class_accuracies[char]:^12.2f} | {class_counts[char]:^12} | {correct:^8} | {incorrect:^9}\n")
        
        # Write detailed prediction analysis
        f.write("\nDetailed Prediction Analysis:\n")
        f.write("=" * 50 + "\n")
        
        for char in sorted(detailed_results.keys()):
            f.write(f"\nClass: {char} (Using {get_hand_for_char(char)} hand model)\n")
            f.write("-" * 20 + "\n")
            
            # Write incorrect predictions for analysis
            if detailed_results[char]['incorrect_predictions']:
                f.write("Incorrect Predictions:\n")
                for pred in detailed_results[char]['incorrect_predictions']:
                    f.write(f"  Image: {pred['image']}\n")
                    f.write(f"  True Label: {pred['true_label']}\n")
                    f.write(f"  Prediction: {pred['prediction']}\n")
                    f.write("  ---\n")
    
    print(f"\nEvaluation results have been saved to: {filename}")
    return overall_accuracy, class_accuracies, class_counts, filename

if __name__ == "__main__":
    overall_acc, class_acc, counts, results_file = evaluate_model()
    
    # Print summary to console
    print("\nSummary of Results:")
    print(f"Overall Accuracy: {overall_acc:.4f}%")
    print(f"Total Classes Evaluated: {len(class_acc)}")
    print(f"Detailed results have been saved to: {results_file}")
