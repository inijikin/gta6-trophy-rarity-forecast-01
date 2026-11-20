# gta6-trophy-rarity-forecast-01

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/inijikin/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/inijikin/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast each GTA VI PlayStation 5 trophy's Exophase earn rate in week one — make.fun bounty entry.

`main.ts` (Node 22, standard library only, no network) reads every `<id>.in` case in the input directory and writes `<id>.out` — one row per trophy with the 0.05 / 0.25 / 0.5 / 0.75 / 0.95 quantiles of the earn rate (%) at the case's target. The model beats the persistence baseline by using the history: a mechanistic counts model (earned vs. players-tracked growth, so new-player dilution pulls rare-trophy rates down) blended with a shrunk logit trend, with HAC standard errors, a split-half consistency gate against spurious trends, and horizon-aware spreads.

Run it:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```
