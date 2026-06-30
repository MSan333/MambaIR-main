#!/usr/bin/env python3
"""使用bicubic下采样从HR图像生成LR图像
用法: python scripts/data/generate_lr.py --hr-dir ./datasets/DIV2K/DIV2K_train_HR --scale 4
"""
import os
import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='从HR图像生成bicubic下采样的LR图像')
    parser.add_argument('--hr-dir', type=str, required=True, help='HR图像目录')
    parser.add_argument('--lr-dir', type=str, default=None, help='LR输出目录 (默认: 同级目录)')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 3, 4], help='下采样倍数')
    args = parser.parse_args()
    
    hr_dir = Path(args.hr_dir)
    if args.lr_dir:
        lr_dir = Path(args.lr_dir)
    else:
        lr_dir = hr_dir.parent / f'LR_bicubic' / f'X{args.scale}'
    
    lr_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"HR目录: {hr_dir}")
    print(f"LR目录: {lr_dir}")
    print(f"缩放倍数: x{args.scale}")
    
    img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    hr_images = sorted([f for f in hr_dir.iterdir() if f.suffix.lower() in img_extensions])
    
    print(f"找到 {len(hr_images)} 张HR图像")
    
    for i, hr_path in enumerate(hr_images, 1):
        lr_name = hr_path.stem + f'x{args.scale}' + '.png'
        lr_path = lr_dir / lr_name
        
        if lr_path.exists():
            continue
        
        img = Image.open(hr_path)
        w, h = img.size
        
        # 确保尺寸能被scale整除
        w_new = w - w % args.scale
        h_new = h - h % args.scale
        if w_new != w or h_new != h:
            img = img.crop((0, 0, w_new, h_new))
        
        lr_img = img.resize((w_new // args.scale, h_new // args.scale), Image.BICUBIC)
        lr_img.save(lr_path)
        
        if i % 100 == 0 or i == len(hr_images):
            print(f"  [{i}/{len(hr_images)}] 已处理")
    
    print(f"\n完成！LR图像保存至: {lr_dir}")


if __name__ == '__main__':
    main()
