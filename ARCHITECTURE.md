# CFB Scouting — Video Analysis v1 Architecture

## Goal

Enrich CollegeFootballData play-by-play with film-derived football context while keeping the current statistical tendency app intact.

## Core principle

CFBD remains the structured source of truth for game situation and result data.

Video analysis adds only the information that structured play-by-play usually cannot provide:

- Personnel
- Formation and alignment
- Motion and shifts
- Run/pass concepts
- Defensive fronts
- Coverage shell
- Blitz / pressure structure
- Player usage details

The shared key between the two layers is the CFBD play/game identifier whenever available.

## Planned pipeline

1. Retrieve CFBD plays.
2. Select one game.
3. Map game plays to timestamps in the local game video.
4. Extract short clips with ffmpeg.
5. Analyze each clip with a future multimodal model.
6. Review/correct the model output.
7. Join film observations back to CFBD plays.
8. Calculate team, player, defense, and coaching tendencies.
9. Generate a scouting report.

## Package layout

- `data/cfbd.py` — CollegeFootballData retrieval.
- `data/playbooks.py` — CFB 27 reference-playbook structures.
- `analytics/tendencies.py` — Existing success/explosive/run-pass tendency engine.
- `analytics/players.py` — Future impact-player analysis.
- `analytics/defense.py` — Blitz and pressure metrics.
- `analytics/coaching.py` — Fourth-down and coaching behavior.
- `models/schemas.py` — Structured film observation contract.
- `video/clips.py` — Local clip extraction.
- `video/synchronization.py` — Mapping film timestamps to plays.
- `video/analyze.py` — Provider-neutral video analysis interface.

## Large game files

Full game broadcasts should not be uploaded to GitHub.

For local development, keep the game file on the computer and extract short clips. The app should send only short clips to a video model or remote storage provider.

Recommended gitignore entries are included for common video formats and local clip directories.
