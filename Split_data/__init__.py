import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from package import shutil,random

class SplitData:
    def __init__(self,root_dir,split_ratio=0.8,dataset_train='dataset_processed_split/train',dataset_test='dataset_processed_split/test'):
        self.root_dir = root_dir
        self.split_ratio = split_ratio
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test

    def split(self):
        source_folder = self.root_dir

        train_folder = r"dataset_processed_split/train"
        test_folder = r"dataset_processed_split/test"
        os.makedirs(self.dataset_train, exist_ok=True)
        os.makedirs(self.dataset_test, exist_ok=True)
        for class_name in os.listdir(source_folder):
            class_path = os.path.join(source_folder, class_name)
            if not os.path.isdir(class_path):
                continue
            
            os.makedirs(os.path.join(self.dataset_train, class_name), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_test, class_name), exist_ok=True)
            
            files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            random.shuffle(files)
            
            split_idx = int(self.split_ratio * len(files))
            train_files = files[:split_idx]
            test_files = files[split_idx:]
            
            for f in train_files:
                shutil.copy(os.path.join(class_path, f), os.path.join(self.dataset_train, class_name, f))
            
            for f in test_files:
                shutil.copy(os.path.join(class_path, f), os.path.join(self.dataset_test, class_name, f))
            
            print(f"{class_name}: {len(train_files)} train, {len(test_files)} test")
        
        

# if __name__ == "__main__":
#     splitter = SplitData(root_dir=r"dataset_processed/train",split_ratio=0.8,dataset_train='dataset_processed_split/train',dataset_test='dataset_processed_split/test')
#     splitter.split()