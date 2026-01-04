import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset,DatasetDict
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image


def get_chunk(dataset, chunk_id, chunk_size):
    """
    返回指定 chunk_id 对应的 HF Dataset 分块。
    """
    total = len(dataset)
    start = chunk_id * chunk_size
    end = min(start + chunk_size, total)
    if start >= total:
        raise ValueError(f"chunk_id {chunk_id} 超过最大 chunk 数量，数据集总长度={total}")
    return dataset.select(range(start, end))


class HF_COCOQA_Dataset(Dataset):
    def __init__(self, max_length=1000,model_name = "sam3"):
        # 加载 huggingface 数据集
        self.ds = load_dataset("Slicky325/coco_qa_dataset",split="train")
        # 为了演示流畅，可以只取前N条
        if max_length:
            self.ds = self.ds.select(range(max_length))
        if model_name == "sam3":
            h,w = 1008,1008
        if model_name == "mobilev4":
            h,w = 224,224
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        # 1. 处理图像
        image = item['image'].convert('RGB')
        image = self.transform(image)
        # 2. 提取三层文本信息
        # L1: Caption (宏观)
        prompt = (
            f"Caption: {item['caption']} "
            f"Question: {item['question']} "
            f"Answer: {str(item['answer'])}"
        )

        # 返回 Tensor 和 原始字符串
        return {
            "image": image,
            "text_prompt": prompt
        }


import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T
import random


class COCO_caption_Dataset(Dataset):
    def __init__(self, img_dir, ann_file, image_size = 224,model_name=None,chunk_id=0, chunk_size=None):
        """
        chunk_id: 当前是第几个分块 (从 0 开始)
        chunk_size: 每个分块包含多少张图片。如果为 None，则加载全部。
        """
        self.img_dir = img_dir
        print(f">>> Loading annotations from {ann_file}...")
        self.coco = COCO(ann_file)
        all_ids = list(sorted(self.coco.imgs.keys()))
        total_images = len(all_ids)
        # 2. 根据 chunk 逻辑进行切片
        if chunk_size is not None:
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, total_images)
            if start_idx >= total_images:
                print(f"Warning: Chunk ID {chunk_id} is out of range. Dataset will be empty.")
                self.ids = []
            else:
                self.ids = all_ids[start_idx:end_idx]
                print(f">>> Chunking Enabled: ID={chunk_id}, Size={chunk_size}")
                print(f">>> Selected indices [{start_idx}:{end_idx}] from {total_images} total images.")
        else:
            # 如果没指定 size，加载全部
            self.ids = all_ids
            print(f">>> Loading Full Dataset: {len(self.ids)} images.")

        print(f">>> Final Dataset Length: {len(self.ids)}")

        # 根据模型选择分辨率
        if model_name == "sam3":
            self.target_size = (1008, 1008)
        else:
            self.target_size = (image_size, image_size)

        from sentence_transformers import SentenceTransformer
        self.text_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        ).to('cuda')
        self.text_encoder.eval()


        self.transform = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 这里 idx 的范围是 0 ~ chunk_size-1
        img_id = self.ids[idx]

        # Captions
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        if len(anns) > 0:
            caption = random.choice(anns)['caption']
        else:
            caption = "an image"

        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']

        img_path = os.path.join(self.img_dir, file_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found {img_path}, skipping...")
            # 简单的递归跳过，防止训练中断
            return self.__getitem__((idx + 1) % len(self))

        image = self.transform(image)

        text_feats = self.text_encoder.encode(
            caption,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        prompt = text_feats
        return {
            "image": image,
            "prompt":prompt,
        }


def get_chunk(ds, chunk_id, chunk_size):
    # 简单的切片逻辑
    start = (chunk_id - 1) * chunk_size
    end = start + chunk_size
    # 防止越界
    total = len(ds)
    if start >= total:
        raise ValueError("Chunk index out of bounds")
    return ds.select(range(start, min(end, total)))

from collections import defaultdict

class Balanced_HF_VQA_Dataset(Dataset):
    def __init__(self, split='train', model_name="sam3"):
        print(f">>> Initializing Balanced Dataset ...")

        # 1. 加载 HuggingFace 数据集
        if split == "train":
            # 建议使用 stream=False 以便快速建立索引，如果内存不够再考虑 iterable
            self.ds = load_dataset(
                "parquet",
                data_files={"train": ["hf://datasets/lmms-lab/VQAv2/data/train-*.parquet"]},
                split="train"
            )
        elif split in ["validation", "val"]:
            self.ds = load_dataset("lmms-lab/VQAv2", split="validation")
        else:
            self.ds = load_dataset("lmms-lab/VQAv2", split=split)

        if model_name == "sam3":
            h, w = 1008, 1008
        else:
            h, w = 224, 224

        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # === Balancing Strategy ===
        print(">>> Scanning dataset for answer types...")
        self.type_indices = defaultdict(list)
        # 遍历元数据 (这一步很快，因为不加载图片)
        # VQAv2 的 answer_type 通常是: 'yes/no', 'number', 'other'
        all_answer_types = self.ds['answer_type']
        for idx, a_type in enumerate(tqdm(all_answer_types)):
            if a_type is None: a_type = 'other'
            self.type_indices[a_type].append(idx)

        print(">>> Original Distribution:")
        for k, v in self.type_indices.items():
            print(f"   {k}: {len(v)}",end="   ")
        # 5. 生成平衡后的虚拟索引列表
        # 策略：以数量最多的类别为基准，对其他类别进行随机重复采样 (Oversampling)
        max_len = max([len(v) for v in self.type_indices.values()])
        self.balanced_indices = []
        # 我们希望每个 Batch 里三种类型的比例大概是 1:1:1
        # 所以我们把三个列表都扩充到 max_len，然后交错混合
        types = ['yes/no', 'number', 'other']
        expanded_lists = {}
        for t in types:
            indices = self.type_indices.get(t, [])
            if not indices: continue  # 防止某个类别完全没有
            # 随机重复采样直到达到 max_len
            # 使用 random.choices 进行有放回采样
            expanded_lists[t] = random.choices(indices, k=max_len)
        # 交错合并 (Interleave)
        # [type1_0, type2_0, type3_0, type1_1, ...]
        for i in range(max_len):
            for t in types:
                if t in expanded_lists:
                    self.balanced_indices.append(expanded_lists[t][i])
        print(f">>> Balanced Dataset Length: {len(self.balanced_indices)} (Effective)")
    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        real_idx = self.balanced_indices[idx]
        item = self.ds[real_idx]
        image = item['image'].convert('RGB')
        image = self.transform(image)
        question = item['question']
        if 'multiple_choice_answer' in item:
            answer = item['multiple_choice_answer']
        elif 'answers' in item and len(item['answers']) > 0:
            answer = item['answers'][0]['answer']
        else:
            answer = "unknown"

        # 获取类型
        q_type = item.get('answer_type', 'other')
        #  === Prompt Engineering ===
        # 告诉 LLM 期待什么样的答案，防止它全部回答 Yes/No
        if q_type == 'yes/no':
            sys_prompt = "Answer with yes or no."
        elif q_type == 'number':
            sys_prompt = "Count the objects and answer with a number."
        else:
            sys_prompt = "Answer the question concisely."
        text_prompt = f"{sys_prompt} Question: {question} Answer: {answer}"

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "text_prompt": text_prompt,  # 包含了 EOS 逻辑将在 collate_fn 处理
            "original_type": q_type  # 用于调试
        }

class HF_VQA_Dataset(Dataset):
    def __init__(self, split='train',model_name = "sam3",chunk_id=1, chunk_size=50000):
        # 加载 huggingface 数据集

        if split == "train":
            self.ds = load_dataset(
                "parquet",
                data_files={"train": ["hf://datasets/lmms-lab/VQAv2/data/train-*.parquet"]},
                split="train"
            )
        elif split == "validation" or "val":
            self.ds = load_dataset("lmms-lab/VQAv2", split="validation")
        elif split == "testdev" or split == "test":
            self.ds = load_dataset("lmms-lab/VQAv2", split=split)
        else:
            raise ValueError(f"Unknown split: {split}. Supported: train, validation, test, testdev")
        if chunk_id and chunk_size is not None:
            self.ds = get_chunk(self.ds, chunk_id=chunk_id, chunk_size=chunk_size)
        if model_name == "sam3":
            h,w = 1008,1008
        if model_name == "mobilev4":
            h,w = 224,224
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        # 1. 处理图像
        image = item['image'].convert('RGB')
        image = self.transform(image)

        question = item['question']
        if 'multiple_choice_answer' in item:
            answer = item['multiple_choice_answer']
        elif 'answers' in item and len(item['answers']) > 0:
            answer = item['answers'][0]['answer']
        else:
            answer = "unknown"


        # 返回 Tensor 和 原始字符串
        return {
            "image": image,
            "question":question,
            "answer":answer,
            "question_type": item.get('question_type', ''),
            "image_id": item.get('image_id', -1)
        }

class VLMCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        texts = [item['text_prompt'] for item in batch]
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_len
        )
        return {
            "image": images,
            "input_ids": encodings['input_ids'],
            "attention_mask": encodings['attention_mask']
        }


import random
class Mixed_COCO_Dataset(Dataset):
    def __init__(self,qa_dataset,caption_dataset,ratio = 0.5):
        self.qa_dataset = qa_dataset
        self.caption_dataset = caption_dataset
        self.ratio = ratio
        self.length = len(self.qa_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        is_caption_task = random.random() < self.ratio
        if is_caption_task:
            cap_idx = idx % len(self.caption_dataset)
            item = self.caption_dataset[cap_idx]
            image = item["image"]
            caption = item["text_prompt"]

            if isinstance(caption,list):
                caption = caption[0]
            question = "Describe in detail."
            return {"image":image,"question":question,"answer":caption,"text_prompt":f"Question:{question} Answer:{caption}"}
        else:
            item = self.qa_dataset[idx]
            image = item["image"]
            q =  item['question']
            a = str(item['answer'])
            return  {"image":image,"question":q,"answer":a,"text_prompt":item['text_prompt']}

class Mixed_Counting_caption_Dataset(Dataset):
    def __init__(self,counting_dataset,caption_dataset,ratio = 0.5):
        self.co_dataset = counting_dataset
        self.caption_dataset = caption_dataset
        self.ratio = ratio
        self.length = len(self.co_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        is_caption_task = random.random() < self.ratio
        if is_caption_task:
            cap_idx = idx % len(self.caption_dataset)
            item = self.caption_dataset[cap_idx]
            image = item["image"]
            caption = item["text_prompt"]
            if isinstance(caption,list):
                caption = caption[0]
            question = "Describe in detail."
            count = -1
            return {"image":image,"question":question,"answer":caption,"task":"caption","text_prompt":f"Question:Describe in detail.\nAnswer:{caption}","count_gt":count}
        else:
            item = self.co_dataset[idx]
            image = item["image"]
            q =  item['question']
            a = str(item['answer'])
            count = item["count_gt"]
            return  {"image":image,"question":q,"answer":a,"task":"counting","text_prompt":f"Question: {q} Answer:{a} ","count_gt":count}



# if __name__ == "__main__":
#     # 配置你的路径 (注意 Windows 路径前面的 r)
#     IMG_DIR = r"D:\VSCode\osteoporosis\CADDM-master\data\COCO\train2014"
#     ANN_FILE = r"D:\VSCode\osteoporosis\RAG\annotations\captions_train2014.json"
#     instance_file = r"D:\VSCode\osteoporosis\RAG\annotations\instances_train2014.json"
#     ca_dataset = COCO_caption_Dataset(img_dir=IMG_DIR,
#                                    ann_file=ANN_FILE,
#                                    model_name="sam3",
#                                    )
#     dataset = Mixed_Counting_caption_Dataset(caption_dataset=ca_dataset,counting_dataset=co_dataset,ratio=0.5)
#     print(f"\nDataset Size: {len(dataset)}")
#     item = dataset[0]
#     print(f"Image Shape: {item['image'].shape}")
#     print(f"Text Prompt: {item['question']}")
#     print(f"Answer: {item['answer']}")
#     loader = DataLoader(dataset, batch_size=4, shuffle=True)
#     batch = next(iter(loader))
#     print(f"Batch Image Shape: {batch['image'].shape}")
#     print("Batch loaded successfully!")

