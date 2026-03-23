import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pandas as pd

class FashionProductDataset(Dataset):
    """Fashion Product Images Dataset加载器"""
    
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        
        # 加载元数据
        self.metadata = pd.read_csv(os.path.join(data_root, 'styles.csv'))
        
        # 过滤存在的图片
        self.valid_indices = []
        for idx in range(len(self.metadata)):
            img_path = self._get_image_path(self.metadata.iloc[idx]['id'])
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
        
        print(f"有效商品数量: {len(self.valid_indices)}")
        
        # 图像预处理（与CLIP模型对齐）
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _get_image_path(self, product_id):
        """获取图片路径"""
        return os.path.join(self.data_root, 'images', f"{product_id}.jpg")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]
        item = self.metadata.iloc[data_idx]
        
        # 加载图像
        img_path = self._get_image_path(item['id'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 构建文本描述
        text = self._build_text_description(item)
        
        # 商品元数据
        metadata = {
            'product_id': str(item['id']),
            'category': item.get('subCategory', 'unknown'),
            'price': float(item.get('price', 0)) if pd.notna(item.get('price')) else 0,
            'style': item.get('style', 'unknown'),
            'description': text
        }
        
        return {
            'image': image,
            'text': text,
            'metadata': metadata
        }
    
    def _build_text_description(self, item):
        """构建文本描述"""
        parts = []
        
        if 'subCategory' in item and pd.notna(item['subCategory']):
            parts.append(str(item['subCategory']))
        
        if 'gender' in item and pd.notna(item['gender']) and item['gender'] != 'Unisex':
            parts.append(str(item['gender']))
        
        if 'color' in item and pd.notna(item['color']):
            parts.append(str(item['color']))
        
        if 'style' in item and pd.notna(item['style']):
            parts.append(str(item['style']))
        
        if 'productDisplayName' in item and pd.notna(item['productDisplayName']):
            text = str(item['productDisplayName'])
        elif parts:
            text = ' '.join(parts)
        else:
            text = 'clothing'
        
        return text

def create_dataloader(data_root, batch_size=64, num_workers=4):
    """创建数据加载器"""
    dataset = FashionProductDataset(data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader