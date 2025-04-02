#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import glob

def convert_pdf(pdf_path):
    """转换单个 PDF 文件为文本"""
    pdf_path = Path(pdf_path)
    output_dir = Path("converted_texts")
    output_dir.mkdir(exist_ok=True)
    
    # 构建输出文件路径，保持原文件名
    output_file = output_dir / f"{pdf_path.stem}.txt"
    
    print(f"处理文件: {pdf_path}")
    print(f"输出到: {output_file}")
    
    try:
        # 使用 pdftotext 进行转换
        result = subprocess.run([
            'pdftotext',
            '-layout',
            '-enc', 'UTF-8',
            str(pdf_path),
            str(output_file)
        ], capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0 and output_file.exists():
            print(f"✓ 成功转换: {pdf_path.name}")
            return True
        else:
            print(f"✗ 转换失败: {pdf_path.name}")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ 转换出错 {pdf_path.name}: {str(e)}")
        return False

def main():
    # 设置语言环境
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    
    # 获取当前目录下所有的 PDF 文件
    pdf_files = []
    for pdf_file in glob.glob('*.pdf'):
        pdf_files.append(pdf_file)
    
    if not pdf_files:
        print("当前目录没有找到 PDF 文件")
        print("当前目录:", os.getcwd())
        print("当前目录文件列表:")
        for f in os.listdir('.'):
            print(f)
        return
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件:")
    for pdf in pdf_files:
        print(f"- {pdf}")
    
    print("\n开始转换...")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(convert_pdf, pdf_files))
    
    # 统计结果
    success_count = sum(1 for r in results if r)
    print("\n转换统计：")
    print(f"总文件数：{len(pdf_files)}")
    print(f"成功转换：{success_count}")
    print(f"失败数量：{len(pdf_files) - success_count}")
    
    # 显示转换后的文件
    output_dir = Path("converted_texts")
    if output_dir.exists():
        print("\n转换后的文件列表：")
        for txt_file in output_dir.glob('*.txt'):
            print(f"- {txt_file.name}")

if __name__ == '__main__':
    main()