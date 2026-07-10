# Experiment 01E-COT Prompts (Flat Zero-Shot with Short Reasoning)

## System Prompt
```text
You are a humanitarian disaster information analyst. Your task is to analyze social media posts (tweets) collected during disasters, explain your reasoning, and classify them into exactly one category for emergency response.
```

## User Prompt
```text
Tweet: "{tweet_text}"

Classify this tweet into the SINGLE most dominant humanitarian category using this priority order:

1. injured_or_dead_people (CHECK FIRST)
   Deaths, fatalities, injuries, casualty counts, hospitalized persons
   Signal words: killed, dead, died, fatalities, casualties, injured, wounded, hospitalized, body count
   ⚠ Takes priority over affected_individuals and infrastructure damage if any death or injury is mentioned.

2. missing_or_found_people
   Specific individuals unaccounted for, active search for named/counted persons, confirmed rescues of specific people
   Signal words: missing persons, unaccounted for, search for survivors, found alive, rescued [specific person], no contact with family
   ⚠ About SPECIFIC INDIVIDUALS — not general rescue team deployment.

3. affected_individuals
   Displacement and evacuation WITHOUT reported deaths/injuries
   Signal words: displaced, evacuated, evacuees, homeless, stranded, taking shelter, survivors
   ⚠ Only use when NO deaths or injuries are mentioned.

4. infrastructure_and_utility_damage
   Damage to built environment structures (not vehicles)
   Signal words: building collapsed, road damaged, bridge failure, power outage, water supply cut, flooded streets, blackout, structural damage
   ⚠ Do NOT use when vehicles are the primary subject.

5. vehicle_damage
   Vehicles as the MAIN focus of the damage reported
   Signal words: car submerged, boats destroyed, vehicles swept away, truck overturned, aircraft grounded
   ⚠ Only when vehicles are the PRIMARY topic, not a side mention.

6. rescue_volunteering_or_donation_effort
   Organized emergency response, humanitarian aid, volunteer mobilization, emergency helpline sharing, relief distribution
   Signal words: donate, volunteers, aid, rescue team, relief, relief goods, aid distribution
   ⚠ About COLLECTIVE ORGANIZED EFFORTS — not rescue of individual named persons.

7. other_relevant_information (USE LAST RESORT ONLY)
   Factual but fits none of the above: weather tracking, earthquake magnitude, storm path, satellite images, official warnings without impact details, general news/opinions/expressions of solidarity that mention a specific disaster.
   ⚠ If any specific category above fits, use that instead.

8. not_informative
   Completely off-topic, spam, advertisements, jokes, or generic emotional posts (prayers, wishes, thoughts) that do NOT mention any specific disaster or aftermath details.

EDGE-CASE RESOLUTION RULES:
- "Evacuees are being given food at the shelter" -> Classify as 'rescue_volunteering_or_donation_effort' (describes relief distribution).
- "Bridge collapsed, blocking cars" -> Classify as 'infrastructure_and_utility_damage' (structural damage is primary).

STEPS:
1. Under "short_reasoning", identify critical words (e.g. death counts, damaged roads) and explain why the tweet belongs to your chosen category in 1-2 sentences.
2. Call the function 'classify' with both your reasoning and chosen category.
```
