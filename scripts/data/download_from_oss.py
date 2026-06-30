#!/usr/bin/env python3
"""从阿里云OSS下载超分辨率数据集到本地
用法: python scripts/data/download_from_oss.py [--data-dir ./datasets]
"""
import os
import sys
import argparse
from pathlib import Path

# OSS配置
OSS_ENDPOINT = 'oss-cn-hangzhou-zmf.aliyuncs.com'
OSS_BUCKET = 'lazada-ai-model'
OSS_PREFIX = 'ad/guoshauile.gsl/data/'
# 从环境变量读取认证信息（使用前请先设置环境变量）
# export OSS_ACCESS_ID="your_access_id"
# export OSS_ACCESS_KEY="your_access_key"
OSS_ACCESS_ID = os.environ.get('OSS_ACCESS_ID', '')
OSS_ACCESS_KEY = os.environ.get('OSS_ACCESS_KEY', '')


def main():
    parser = argparse.ArgumentParser(description='从OSS下载超分辨率数据集')
    parser.add_argument('--data-dir', type=str, default='./datasets',
                       help='本地数据保存目录 (默认: ./datasets)')
    parser.add_argument('--endpoint', type=str, default=OSS_ENDPOINT,
                       help=f'OSS Endpoint (默认: {OSS_ENDPOINT})')
    args = parser.parse_args()
    
    try:
        import oss2
    except ImportError:
        print("[错误] 请先安装oss2: pip install oss2")
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("  从OSS下载超分辨率数据集")
    print(f"  Bucket: {OSS_BUCKET}")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  OSS路径: {OSS_PREFIX}")
    print(f"  本地目录: {data_dir.absolute()}")
    print("=" * 50)
    
    # 连接OSS
    auth = oss2.Auth(OSS_ACCESS_ID, OSS_ACCESS_KEY)
    bucket = oss2.Bucket(auth, f'https://{args.endpoint}', OSS_BUCKET)
    
    # 列出所有文件
    print("\n[扫描] 列出OSS文件...")
    files = []
    for obj in oss2.ObjectIterator(bucket, prefix=OSS_PREFIX):
        if not obj.key.endswith('/'):
            files.append(obj.key)
    
    print(f"[发现] 共 {len(files)} 个文件")
    
    if not files:
        print("[警告] OSS路径下没有文件，请确认数据已上传")
        return
    
    # 下载
    downloaded = 0
    skipped = 0
    failed = 0
    
    for i, key in enumerate(files, 1):
        # 去除前缀，保留相对路径
        relative = key[len(OSS_PREFIX):]
        local_path = data_dir / relative
        
        # 跳过已存在
        if local_path.exists():
            skipped += 1
            continue
        
        # 创建目录
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            bucket.get_object_to_file(key, str(local_path))
            downloaded += 1
            if downloaded % 100 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] 已下载: {downloaded}, 跳过: {skipped}")
        except Exception as e:
            failed += 1
            print(f"  [失败] {relative}: {e}")
    
    print(f"\n[完成] 下载: {downloaded}, 跳过: {skipped}, 失败: {failed}")
    print(f"  数据目录: {data_dir.absolute()}")


if __name__ == '__main__':
    main()
