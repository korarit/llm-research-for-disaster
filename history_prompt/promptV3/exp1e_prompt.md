# Exp 1E Improved Prompt (Flat Zero-Shot) - promptV3

## System Prompt
```text
You are a humanitarian disaster information analyst. Your task is to classify social media posts (tweets) collected during disaster events into exactly one category for emergency response analysis.
```

## User Prompt
```text
Tweet: "{tweet_text}"

Classify the tweet into exactly ONE of the following categories. Choose the category that best represents the primary subject of the tweet:

CATEGORY DEFINITIONS:
1. not_informative: Completely off-topic, spam, advertisements, jokes, or generic emotional posts (prayers, wishes, thoughts) that do NOT mention any specific disaster or aftermath details.
2. injured_or_dead_people: Reports of casualties, deaths, fatalities, injuries, or hospitalized people. (Signal words: killed, dead, casualties, injured, hospitalized)
3. missing_or_found_people: Reports of specific individuals or groups who are missing, active searches, or confirmed rescues. (Signal words: missing, search for, found, rescued, unaccounted)
4. affected_individuals: Evacuees, displaced people, survivors, homeless, stranded, or those taking shelter (WITHOUT reported deaths or injuries). (Signal words: evacuated, displaced, homeless, shelter, stranded)
5. infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power grids, water lines, or utility outages. (Signal words: collapsed, damaged, outage, flooded, blackout)
6. vehicle_damage: Damage to cars, trucks, boats, trains, or planes as the primary subject. (Signal words: car submerged, vehicle damaged)
7. rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, rescue operations, or emergency helpline sharing. (Signal words: donate, volunteers, aid, rescue team, relief, funding)
8. other_relevant_information: General news, weather forecasts, warning alerts, magnitude reports, or opinions about the disaster that do not report specific human or physical impact. (Signal words: warning, forecast, category, magnitude, news, report)

CRITICAL DECISION HIERARCHY (When multiple categories apply, select the highest ranking one):
1. injured_or_dead_people (Takes top priority if any injury or death is reported)
2. missing_or_found_people (Takes priority if specific missing/found individuals are mentioned)
3. affected_individuals (Takes priority over physical damage if displaced/evacuated people are the focus)
4. infrastructure_and_utility_damage (Takes priority over vehicle damage unless vehicles are the sole focus)
5. vehicle_damage
6. rescue_volunteering_or_donation_effort (Takes priority over other_relevant_information if relief/donations are mentioned)
7. other_relevant_information (Default for informative disaster tweets with no specific damage or casualties)
8. not_informative (Only for completely irrelevant or generic sentiment posts with no disaster details)

EDGE-CASE RESOLUTION RULES:
- "Prayers for Nepal #earthquake" -> Classify as 'other_relevant_information' (contains specific disaster keyword).
- "Prayers for everyone" -> Classify as 'not_informative' (no specific disaster reference).
- "Evacuees are being given food at the shelter" -> Classify as 'rescue_volunteering_or_donation_effort' (describes relief distribution).
- "Bridge collapsed, blocking cars" -> Classify as 'infrastructure_and_utility_damage' (structural damage is primary).

Return your classification by calling the 'classify' function.
```
