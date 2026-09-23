# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
Sample script for cv2.docRead(): runs a VLM OCR / document-understanding engine over a
page image and returns a unified, engine-agnostic result -- pages made of typed blocks
(title/text/table/form/signature), each carrying lines, table cells or form fields.

Local model types (paddleocr_vl, granite_docling) load ONNX weights from --model_dir,
following the upstream layout of the model's ONNX export:
    https://huggingface.co/PaddlePaddle/PaddleOCR-VL
    https://huggingface.co/onnx-community/granite-docling-258M-ONNX
Cloud model types (openai, anthropic, gemini, grok) call the provider's hosted vision API
instead -- pass the provider's model name as --model_dir and an API key via --api_key, or
via the provider's usual environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY,
GEMINI_API_KEY, GROK_API_KEY).

Usage:
    python docread.py --model_type=granite_docling \
                       --model_dir=<path-to-granite-docling-258M-ONNX-export> \
                       --input=<path-to-page-image>

    python docread.py --model_type=openai --model_dir=gpt-4o --input=<path-to-page-image>
'''

import argparse
import os

import cv2 as cv

MODEL_TYPES = {
    'paddleocr_vl': 0,
    'granite_docling': 1,
    'openai': 2,
    'anthropic': 3,
    'gemini': 4,
    'grok': 5,
}

API_KEY_ENV_VARS = {
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'gemini': 'GEMINI_API_KEY',
    'grok': 'GROK_API_KEY',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run cv2.docRead() on a page image.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', type=str, required=True,
                         help='Path to the input page image.')
    parser.add_argument('--model_type', type=str, required=True, choices=sorted(MODEL_TYPES),
                         help='Which VLM engine to use.')
    parser.add_argument('--model_dir', type=str, required=True,
                         help='Local model types: path to the local ONNX export directory. '
                              'Cloud model types: the provider model name, e.g. gpt-4o.')
    parser.add_argument('--engine', type=str, default='opencv', choices=('opencv',),
                         help='cv::dnn engine for local model types; ignored by cloud model types.')
    parser.add_argument('--device', type=str, default='cpu', choices=('cpu', 'cuda'),
                         help='Compute device for local model types; ignored by cloud model types.')
    parser.add_argument('--api_key', type=str, default='',
                         help='API key for cloud model types; falls back to the provider\'s '
                              'usual environment variable when omitted.')
    parser.add_argument('--prompt', type=str, default='',
                         help='Task prompt; an empty string uses the engine\'s default prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                         help='Maximum number of tokens the engine may generate.')
    parser.add_argument('--raw', action='store_true',
                         help='Skip structured parsing and print each page\'s raw engine output.')
    return parser.parse_args()


def print_block(block, indent):
    prefix = ' ' * indent
    print(f'{prefix}[{block.type}]')
    for line in block.get('lines', []):
        print(f'{prefix}  {line.text}')
    for cell in block.get('cells', []):
        header = ' (header)' if cell.is_header else ''
        print(f'{prefix}  [{cell.row},{cell.col}]{header} {cell.text}')
    for field in block.get('fields', []):
        print(f'{prefix}  {field.key}: {field.value}')


def print_result(result):
    m = result.metadata
    print(f'model={m.model} engine={m.engine} pages={m.pages} '
          f'tokens_used={m.tokens_used} inference_time_ms={m.inference_time_ms}')
    for page in result.pages:
        print(f'--- page {page.page_number} ({page.width}x{page.height}) ---')
        for block in page.blocks:
            print_block(block, indent=2)


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get(API_KEY_ENV_VARS.get(args.model_type, ''), '')

    result = cv.docRead(args.input, MODEL_TYPES[args.model_type], args.model_dir,
                         engine=args.engine, device=args.device, api_key=api_key,
                         prompt=args.prompt, max_new_tokens=args.max_new_tokens, raw=args.raw)

    if args.raw:
        for i, page_text in enumerate(result, start=1):
            print(f'--- page {i} (raw) ---')
            print(page_text)
    else:
        print_result(result)


if __name__ == '__main__':
    main()
