"""Extract user preferences from dialogues with GPT. Finished dialogues are skipped on restart."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from extraction.common import call_gpt, load_records, parse_preferences, save_json
from extraction.prompts import PREFERENCE_SYSTEM_PROMPT, preference_prompt


def extract(client, index, record, model):
    messages = [{'role': 'system', 'content': PREFERENCE_SYSTEM_PROMPT},
                {'role': 'user', 'content': preference_prompt(record['context'])}]
    base = dict(context=record['context'], gt_items=record.get('gt_items'))
    try:
        return index, dict(base, preferences=parse_preferences(call_gpt(client, messages, model)) or None)
    except Exception as error:
        return index, dict(base, preferences=None, extraction_error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='gpt-4.1')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=50)
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    records = load_records(args.input)
    results = load_records(args.output) if os.path.exists(args.output) else list(records)
    pending = [i for i, record in enumerate(results) if not record.get('preferences')]

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract, client, i, records[i], args.model) for i in pending]
        for done, future in enumerate(tqdm(as_completed(futures), total=len(futures)), 1):
            index, results[index] = future.result()
            failures += results[index]['preferences'] is None
            if done % args.save_every == 0:
                save_json(results, args.output)
    save_json(results, args.output)
    print(f'Saved {args.output} ({failures} failures)')


if __name__ == '__main__':
    main()
