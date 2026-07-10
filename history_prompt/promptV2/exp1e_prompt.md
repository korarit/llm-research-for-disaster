# Exp 1E Optimized Prompt (Flat Zero-Shot)

## System Prompt
```text
You are a humanitarian disaster information analyst. Your task is to classify social media posts (tweets) collected during disaster events into exactly one category for emergency response analysis.
```

## User Prompt
```text
Tweet: "{tweet_text}"

Classify the tweet into exactly ONE of the following categories. Choose the category that best represents the primary subject of the tweet:

CATEGORY DEFINITIONS:
1. not_informative: Completely off-topic, spam, or irrelevant content that does not refer to the disaster or its aftermath at all.
2. injured_or_dead_people: Reports of casualties, deaths, fatalities, or injured people. (Signal words: killed, dead, casualties, injured, hospitalized)
3. missing_or_found_people: Reports of individuals or groups who are missing or have been found/rescued. (Signal words: missing, search for, found, rescued)
4. affected_individuals: Evacuees, displaced people, or survivors (without death or injury). (Signal words: evacuated, displaced, homeless, shelter, stranded)
5. infrastructure_and_utility_damage: Damage to buildings, roads, bridges, power lines, or utilities. (Signal words: collapsed, damaged, outage, flooded, blackout)
6. vehicle_damage: Damage to cars, trucks, boats, or planes. (Signal words: car submerged, vehicle damaged)
7. rescue_volunteering_or_donation_effort: Relief goods, donations, volunteering, aid distribution, or rescue operations. (Signal words: donate, volunteers, aid, rescue team, relief)
8. other_relevant_information: General news, weather forecasts, warnings, comments, political remarks, or opinions about the disaster that do not fit the specific categories above. (Signal words: warning, forecast, category, magnitude, news, report)

Return your classification by calling the 'classify' function.
```
