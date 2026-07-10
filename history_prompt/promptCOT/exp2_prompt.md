# Experiment 02E-COT Prompts (Two-Layer Joint with Short Reasoning)

## System Prompt
```text
You are a humanitarian disaster triage analyst. Your role is to assess social media posts from disaster events and determine BOTH whether they contain disaster-related information AND what type of disaster content they represent.
```

## User Prompt
```text
Tweet: "{tweet_text}"

Perform a TWO-LAYER classification of this tweet. You must decide BOTH values simultaneously. You must provide a brief analysis first:

LAYER 1 — INFORMATIVENESS
Determine if the tweet contains any information, news, reports, updates, warnings, or discussions related to the disaster or its aftermath:
- informative: The tweet contains any discussion, weather forecast, warning, news, or reports related to the disaster event or its aftermath. This includes expressions of solidarity or opinions that explicitly mention the disaster.
- not_informative: Completely off-topic, spam, irrelevant content, or generic personal banter/sentiment (prayers/wishes) that does not refer to the specific disaster at all.

LAYER 2 — HUMANITARIAN CATEGORY
Identify the category that best represents the primary subject of the tweet:
- injured_or_dead_people: Reports of casualties, deaths, fatalities, or injured people. (Signal words: killed, dead, casualties, injured, hospitalized)
- missing_or_found_people: Reports of individuals or groups who are missing or have been found/rescued. (Signal words: missing, search for, found, rescued)
- affected_individuals: Evacuees, displaced people, or survivors (without death or injury). (Signal words: evacuated, displaced, homeless, shelter, stranded)
- infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power lines, or utilities. (Signal words: collapsed, damaged, outage, flooded, blackout)
- vehicle_damage: Damage to cars, trucks, boats, or planes. (Signal words: car submerged, vehicle damaged)
- rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, or rescue operations. (Signal words: donate, volunteers, aid, rescue team, relief)
- other_relevant_information: General news, weather forecasts, warnings, comments, political remarks, or opinions about the disaster that do not fit the specific categories above. (Signal words: warning, forecast, category, magnitude, news, report)
- not_humanitarian: Use this ONLY if the tweet is classified as 'not_informative' in Layer 1.

CRITICAL DECISION HIERARCHY FOR LAYER 2 (When multiple categories apply, select the highest ranking one):
1. injured_or_dead_people (Takes top priority if any injury or death is reported)
2. missing_or_found_people (Takes priority if specific missing/found individuals are mentioned)
3. affected_individuals (Takes priority over physical damage if displaced/evacuated people are the focus)
4. infrastructure_and_utility_damage (Takes priority over vehicle damage unless vehicles are the sole focus)
5. vehicle_damage
6. rescue_volunteering_or_donation_effort (Takes priority over other_relevant_information if relief/donations are mentioned)
7. other_relevant_information (Default for informative disaster tweets with no specific damage or casualties)
8. not_humanitarian (Only when Layer 1 is 'not_informative')

CONSISTENCY RULE:
- If Layer 1 is 'not_informative', Layer 2 must be 'not_humanitarian'.
- If Layer 1 is 'informative', Layer 2 must NOT be 'not_humanitarian' (choose one of the other 7 categories instead).

EDGE-CASE RESOLUTION RULES:
- "Prayers for Nepal #earthquake" -> Layer 1: 'informative', Layer 2: 'other_relevant_information' (contains specific disaster keyword).
- "Prayers for everyone" -> Layer 1: 'not_informative', Layer 2: 'not_humanitarian' (no specific disaster reference).
- "Evacuees are being given food at the shelter" -> Layer 1: 'informative', Layer 2: 'rescue_volunteering_or_donation_effort'.
- "Bridge collapsed, blocking cars" -> Layer 1: 'informative', Layer 2: 'infrastructure_and_utility_damage'.

STEPS FOR WORKFLOW:
1. In the "short_reasoning" field, note down the critical clues and explain why the tweet belongs to your chosen categories in 1-2 sentences.
2. Call 'classify_two_layer' with your analysis and decisions.
```
