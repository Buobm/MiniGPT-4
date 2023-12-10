"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from PIL import Image
import os

from minigpt4.datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


class OKVQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image_id']
        question = data['question']
        question_id = data['question_id']
        img_file = '{:0>12}.jpg'.format(img_id)
        image_path = os.path.join(self.root_path, img_file)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, question_id, img_id

class VizWizEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image']
        question = data['question']
        answers = data['answers']
        answers = '_'.join([answer['answer'] for answer in answers])
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] The question is '{question}' Based on the image, answer the question with a single word or phrase. and reply 'unanswerable' when the provided information is insufficient"
        return image, question, answers

class IconQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image_id = data['image_id']
        question = data['question']
        image_path = os.path.join(self.root_path, image_id, 'image.png')
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image).half().cuda()
        candidates = '_'.join(data['choices'])
        answer = data['answer']
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, candidates, answer

class GQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_id = ann["image"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["question"]
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        labels = ann["answer"]

        return image, question, labels

class HMEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_id = ann["img"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["text"]
        question = f"This is an image writting '{question}'. Is this image hateful? Answer yes or no. Answer:"
        labels = ann["label"]

        return image, question, labels

class VSREvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_path = os.path.join(self.root_path, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["caption"]
        question = f'[vqa] Based on the image, is this statement true or false? {question}'
        labels = 'true' if ann["label"] == 1 else 'false'

        return image, question, labels

class HoloAssistData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.count = 0

    def get_image(self,image_list):
        num = len(image_list)
        i = int(num/2.0)
        return image_list[i]

    
    def generate_question(self, data):
        # Generate a question based on the top 5 predictions
        predictions = data['predictions']
        options = [pred['label'] for pred in predictions]
        #options.append('None')  # Adding 'None' as an option
        options_str = ', '.join(options)
        question = f"You are now an action recognition assistent of few words. Based on the four images, where the top left is the first, top right the second, bottom left the third and bottom right the fourth image in a sequence, which of the following actions is being performed: {options_str} ? You have to respond only with the action name."
        #question = f"Based on the 4 images that start in the top left and end bottom right which of the following actions is being performed: {options_str}. Only tell me the correct action from the given list." #"If none of the descriptions fit say: None"
        #question = f"[vqa] Which action of the following options is the person in the image performing? {options_str}  .Only tell me the action from the list"
        return question

    def get_answer(self,data):
        correct_label = data['correct_label']
        options = [pred['label'] for pred in data['predictions']]
        if correct_label in options:
            answer = correct_label
        else:
            answer = 'None'
        all_options = options
        all_options.append(correct_label)
        return answer, all_options
    

    def load_images(self, image_names):
        indexes = [0, 5, 10, 15]
        images = [image_names[i] for i in indexes]
        selected_images = []

        for img_name in images:
            image_path = os.path.join(self.root_path, img_name)
            image = Image.open(image_path).convert('RGB')
            selected_images.append(image)
        return selected_images

    def create_image_grid(self, images, rows=2, cols=2) -> Image:
        assert len(images) == rows * cols
        w, h = images[0].size
        grid = Image.new('RGB', (cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, (w * (i % cols), h * (i // cols)))
        return grid
        
    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        
        images = self.load_images(data['image'])
        image = self.create_image_grid(images)
        if self.count == 0:
            image.save("image.jpg")
            self.count += 1
        question = self.generate_question(data)
        answers, options = self.get_answer(data)
        # img_id = self.get_image(data['image'])
        # image_path = os.path.join(self.root_path, img_id)
        # image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        if len(options) != 6:
            print(options)

        return image, question, answers, options