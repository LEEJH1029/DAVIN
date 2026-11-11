FORMAT_RULES = """- "category" names the KIND of property, not the property itself and never its value. Test it: "what kind of property is this?" — answer with the broadest name that is still correct, one that products of any kind could share ("color", not "dial color"; "feature", not "water resistance"). A name that refers to a part of the product, or to a specific characteristic, is too specific: the part belongs in the value, and any functional or quality characteristic is classified as "feature". The product type is always named exactly "product type". Use the same category name every time for the same kind of property.
- "value": exactly one property per entry, 1-4 lowercase words. A phrase that bundles several properties, or lists alternatives, becomes one entry per property or alternative. The product type value is only the generic product noun; every other detail is its own entry under its own category."""

PREFERENCE_SYSTEM_PROMPT = ("You are an expert at reading Reddit threads about products. The dialogue is a thread "
                            "(a submission, possibly followed by comments); the product of interest is the one the NEXT reply "
                            "in the thread would recommend or link. Identify what the user wants in that product.")

ATTRIBUTE_SYSTEM_PROMPT = ("You are an expert at describing products from their listing title and product image, "
                           "decomposing them into structured attributes.")

PREFERENCE_OUTPUT = '{"summary": "...", "preferences": [{"category": "...", "value": "..."}]}'

ATTRIBUTE_OUTPUT = ('{"summary": "...", "title_attributes": [{"category": "...", "value": "..."}], '
                    '"image_attributes": [{"category": "...", "value": "..."}], '
                    '"merged_attributes": [{"category": "...", "value": "..."}]}')


def preference_prompt(dialogue: str) -> str:
    return f"""Extract the user's preferences for the product they are looking for from the dialogue below.
Dialogue:
{dialogue}
- The dialogue is a Reddit thread. First decide which product the NEXT reply would recommend and state it in one sentence as "summary". Then extract preferences only for that product; ignore other products mentioned along the way (e.g., what the user already owns or is replacing), even with strong opinions attached. Focus on the most recent messages, but the target may be established anywhere in the thread.
- Extract every property the user wants in that product, whether stated directly or clearly implied by the context (for instance, "something for my daughter's birthday" implies target user: child). Do not add properties that merely seem likely but are not supported by what the user said. Always include the product type.
- Extract only properties of the product itself. Skip: what the user rejects; facts about the user that do not translate into a product property; where or how to buy; and judgments with no specific content behind them (a bare "comfortable" or "good" says nothing about the product). Ignore "[deleted]" comments and garbled characters.
Preference format:
{FORMAT_RULES}
Output ONLY a JSON object:
{PREFERENCE_OUTPUT}"""


def attribute_prompt(title: str) -> str:
    return f"""Extract the item's attributes from its title and image below.
Title: {title}
First state in one sentence what the item is, as "summary". Then work in three steps and output all three.
- Step 1, title only: every property the title states. Extract only properties of the product itself: skip marketing claims with no specific content behind them, listing codes, and anything about the seller or shipping.
- Step 2, image only: every property you can see — color, style, shape, pattern, finish, fit, visible material, legible brand text. Describe what the image shows on its own terms, as if the title did not exist: do not skip a property because the title already mentions it, and do not reword what you see to match the title.
- Step 3, merge: keep the union of both lists. If both state the same fact, keep it once with the more accurate wording. If they disagree, keep the version better supported by the evidence you can actually verify. Different values of one category are different facts — keep all of them.
Attribute format:
{FORMAT_RULES}
Output ONLY a JSON object:
{ATTRIBUTE_OUTPUT}"""
