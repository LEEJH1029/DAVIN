import json
import os
import re
import time

from davin.data import find_main_image


def call_gpt(client, messages, model, max_retries=4):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=0,
                                                      response_format={'type': 'json_object'})
            return response.choices[0].message.content
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def load_records(path: str) -> list:
    """Read a JSON array or a stream of JSON objects."""
    with open(path, encoding='utf-8') as stream:
        text = stream.read()
    decoder = json.JSONDecoder()
    position, values = 0, []
    while position < len(text):
        while position < len(text) and text[position] in ' \t\r\n':
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        values.append(value)
    return values[0] if len(values) == 1 and isinstance(values[0], list) else values


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f'{path}.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def catalog_items(metadata_path: str, main_category: str, images_dirs: list) -> list:
    with open(metadata_path, encoding='utf-8') as stream:
        metadata = json.load(stream)
    records = []
    for asin in sorted(a for a, m in metadata.items() if m.get('main_category') == main_category):
        image_path = find_main_image(images_dirs, asin)
        if image_path is not None:
            records.append(dict(asin=asin, title=(metadata[asin].get('title') or '').strip(), image_path=image_path))
    return records


def normalize_entries(raw) -> list:
    """Lowercase categories, strip values, and drop repeated (category, value) pairs."""
    out, seen = [], set()
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        category, value = entry.get('category'), entry.get('value')
        if not category or value in (None, ''):
            continue
        category, value = str(category).strip().lower(), str(value).strip()
        if (category, value.lower()) not in seen:
            seen.add((category, value.lower()))
            out.append(dict(category=category, value=value))
    return out


def _json_payload(text: str) -> str:
    text = (text or '').strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0]
    elif '```' in text:
        text = text.split('```')[1].split('```')[0]
    start, end = text.find('{'), text.rfind('}') + 1
    if start != -1 and end > start:
        text = text[start:end]
    return re.sub(r',\s*([}\]])', r'\1', text)


def parse_preferences(text: str) -> list:
    return normalize_entries(json.loads(_json_payload(text)).get('preferences'))


def parse_attributes(text: str):
    """Return the merged attributes, or None on failure. An empty merged list falls back to the title and image lists."""
    payload = _json_payload(text)
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r'(?<=\d)"', r'\\"', payload))      # Unescaped inch marks such as 5'2"
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    return (normalize_entries(obj.get('merged_attributes'))
            or normalize_entries(normalize_entries(obj.get('title_attributes')) + normalize_entries(obj.get('image_attributes')))
            or None)
