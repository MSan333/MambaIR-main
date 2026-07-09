#!/usr/bin/env python3
"""通过 SSH/SFTP 上传数据到云开发机"""
import os
import sys
import stat
import paramiko
import time

HOST = "connect.nmb1.seetacloud.com"
PORT = 20683
USER = "root"
PASSWORD = "v+9om9z7d74+"
REMOTE_BASE = "/root/autodl-tmp/data"

LOCAL_DIRS = [
    "/home/guoshuaile/pyproject/data/SR",
    "/home/guoshuaile/pyproject/data/DIV2K",
]


def human_size(n):
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def mkdir_p(sftp, remote_dir):
    """递归创建远程目录"""
    if remote_dir in ('/', ''):
        return
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        parent = os.path.dirname(remote_dir)
        mkdir_p(sftp, parent)
        sftp.mkdir(remote_dir)


def upload_dir(sftp, local_dir, remote_dir, file_count, file_idx, skipped):
    """递归上传目录（支持断点续传：跳过已存在且大小相同的文件）"""
    for entry in sorted(os.listdir(local_dir)):
        local_path = os.path.join(local_dir, entry)
        remote_path = remote_dir + '/' + entry

        if os.path.isdir(local_path):
            try:
                sftp.stat(remote_path)
            except FileNotFoundError:
                sftp.mkdir(remote_path)
            file_count, file_idx, skipped = upload_dir(sftp, local_path, remote_path, file_count, file_idx, skipped)
        else:
            file_size = os.path.getsize(local_path)
            file_idx[0] += 1
            # 断点续传：检查远程文件是否已存在且大小一致
            try:
                remote_stat = sftp.stat(remote_path)
                if remote_stat.st_size == file_size:
                    skipped[0] += 1
                    if skipped[0] % 100 == 0:
                        print(f"  [{file_idx[0]}/{file_count}] {entry}  跳过(已存在)  累计跳过{skipped[0]}")
                    continue
            except FileNotFoundError:
                pass
            except Exception:
                pass
            t0 = time.time()
            sftp.put(local_path, remote_path)
            dt = time.time() - t0
            speed = file_size / dt if dt > 0 else 0
            print(f"  [{file_idx[0]}/{file_count}] {entry}  {human_size(file_size)}  {human_size(speed)}/s  {dt:.1f}s")
    return file_count, file_idx, skipped


def count_files(local_dir):
    n = 0
    for _, _, files in os.walk(local_dir):
        n += len(files)
    return n


def main():
    print(f"连接 {HOST}:{PORT} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("连接成功!")

    sftp = ssh.open_sftp()

    # 创建远程基目录
    mkdir_p(sftp, REMOTE_BASE)
    print(f"远程目录 {REMOTE_BASE} 已就绪\n")

    for local_dir in LOCAL_DIRS:
        dir_name = os.path.basename(local_dir)
        remote_dir = REMOTE_BASE + '/' + dir_name
        total_files = count_files(local_dir)
        total_size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(local_dir) for f in fs)

        print(f"{'='*60}")
        print(f"上传: {local_dir}")
        print(f"  -> {remote_dir}")
        print(f"  文件数: {total_files}, 总大小: {human_size(total_size)}")
        print(f"{'='*60}")

        mkdir_p(sftp, remote_dir)
        t_start = time.time()
        _, _, skipped = upload_dir(sftp, local_dir, remote_dir, total_files, [0], [0])
        dt = time.time() - t_start
        print(f"  完成! 耗时 {dt:.0f}s, 跳过{skipped[0]}个已存在文件")

    sftp.close()
    ssh.close()
    print("全部上传完成!")


if __name__ == '__main__':
    main()
