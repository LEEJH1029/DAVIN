"""Extract item attributes from the title and representative image with GPT. Finished items are skipped on restart."""
import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from extraction.common import call_gpt, catalog_items, parse_attributes, save_json
from extraction.prompts import ATTRIBUTE_SYSTEM_PROMPT, attribute_prompt


def extract(client, record, model, detail):
    try:
        with open(record['image_path'], 'rb') as stream:
            image = base64.b64encode(stream.read()).decode('ascii')
        messages = [{'role': 'system', 'content': ATTRIBUTE_SYSTEM_PROMPT},
                    {'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}', 'detail': detail}},
                        {'type': 'text', 'text': attribute_prompt(record['title'])}]}]
        attributes = parse_attributes(call_gpt(client, messages, model))
        return record['asin'], dict(title=record['title'], merged_attributes=attributes)
    except Exception as error:
        return record['asin'], dict(title=record['title'], merged_attributes=None, extraction_error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main_category', required=True)
    parser.add_argument('--items', default='data/item/metadata.json')
    parser.add_argument('--images_dirs', nargs='+', default=['data/item/images'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='gpt-4.1')
    parser.add_argument('--image_detail', default='auto', choices=['low', 'high', 'auto'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=100)
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    results = {}
    if os.path.exists(args.output):
        with open(args.output, encoding='utf-8') as stream:
            results = json.load(stream)
    pending = [r for r in catalog_items(args.items, args.main_category, args.images_dirs)
               if not results.get(r['asin'], {}).get('merged_attributes')]

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract, client, r, args.model, args.image_detail) for r in pending]
        for done, future in enumerate(tqdm(as_completed(futures), total=len(futures)), 1):
            asin, results[asin] = future.result()
            failures += results[asin]['merged_attributes'] is None
            if done % args.save_every == 0:
                save_json(results, args.output)
    save_json(results, args.output)
    print(f'Saved {args.output} ({failures} failures)')


if __name__ == '__main__':
    main()
