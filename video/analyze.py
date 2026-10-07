from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, Optional, Protocol

from google import genai
from pydantic import BaseModel, Field

from models.schemas import FilmObservation


class VideoAnalyzer(Protocol):
    """Interface that any video model integration must implement."""

    def analyze_clip(self, clip_path: str, play_context: dict) -> FilmObservation:
        ...


class FootballSnapChart(BaseModel):
    personnel: Optional[Literal["10", "11", "12", "13", "20", "21", "22", "Other", "Unknown"]] = None
    formation_family: Optional[Literal["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]] = None
    initial_formation: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    initial_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    final_formation: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    final_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    formation_strength: Optional[Literal["Left", "Right", "Balanced", "Boundary", "Field", "Unknown"]] = None

    motion_present: Optional[bool] = None
    motion_player: Optional[str] = None
    motion_type: Optional[Literal["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]] = None
    motion_direction: Optional[str] = None
    motion_start_alignment: Optional[str] = None
    motion_end_alignment: Optional[str] = None
    shift_present: Optional[bool] = None
    shift_description: Optional[str] = None

    film_play_type: Optional[Literal["Run", "Pass", "RPO", "Scramble", "Sack", "Other", "Unknown"]] = None
    run_concept: Optional[str] = None
    run_direction: Optional[Literal["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]] = None
    pass_concept: Optional[str] = None
    rpo: Optional[bool] = None
    play_action: Optional[bool] = None

    defensive_personnel: Optional[str] = None
    front: Optional[str] = None
    initial_box_count: Optional[int] = Field(default=None, ge=0, le=11)
    snap_box_count: Optional[int] = Field(default=None, ge=0, le=11)
    shell: Optional[Literal["1-High", "2-High", "0-High", "Unknown"]] = None
    coverage: Optional[str] = None
    rushers: Optional[int] = Field(default=None, ge=0, le=11)
    blitz: Optional[bool] = None
    pressure_family: Optional[Literal["Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"]] = None
    pressure_source: Optional[str] = None

    adjustment_trigger: Optional[Literal["None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"]] = None
    adjustment_type: Optional[Literal["None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"]] = None
    adjustment_player: Optional[str] = None
    adjustment_detail: Optional[str] = None

    overall_confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    analysis_notes: str = ""


class StructurePass(BaseModel):
    """Mechanical pre-snap observations. Avoid football shorthand when possible."""

    rb_count: Optional[int] = Field(default=None, ge=0, le=4)
    te_count: Optional[int] = Field(default=None, ge=0, le=4)
    formation_family: Optional[Literal["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]] = None
    initial_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    initial_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    final_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    final_backfield: Optional[Literal["Split backs", "RB left", "RB right", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    formation_strength: Optional[Literal["Left", "Right", "Balanced", "Boundary", "Field", "Unknown"]] = None

    defensive_personnel: Optional[str] = None
    front_family: Optional[Literal["Even", "Odd", "Bear", "Mint/Tite", "Other", "Unknown"]] = None
    initial_box_count: Optional[int] = Field(default=None, ge=0, le=11)
    snap_box_count: Optional[int] = Field(default=None, ge=0, le=11)
    shell: Optional[Literal["1-High", "2-High", "0-High", "Unknown"]] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class MovementPass(BaseModel):
    """Track only movement from the earliest settled set through the snap."""

    motion_present: Optional[bool] = None
    moving_at_snap: Optional[bool] = None
    motion_player: Optional[str] = None
    motion_type: Optional[Literal["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]] = None
    motion_direction: Optional[str] = None
    motion_start_alignment: Optional[str] = None
    motion_end_alignment: Optional[str] = None

    shift_present: Optional[bool] = None
    shift_player: Optional[str] = None
    shift_description: Optional[str] = None

    adjustment_trigger: Optional[Literal["None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"]] = None
    adjustment_type: Optional[Literal["None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"]] = None
    adjustment_player: Optional[str] = None
    adjustment_detail: Optional[str] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class PostSnapPass(BaseModel):
    """Post-snap result and defensive pressure/coverage observations."""

    film_play_type: Optional[Literal["Run", "Pass", "RPO", "Scramble", "Sack", "Other", "Unknown"]] = None
    run_concept: Optional[str] = None
    run_direction: Optional[Literal["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]] = None
    pass_concept: Optional[str] = None
    rpo: Optional[bool] = None
    play_action: Optional[bool] = None

    rushers: Optional[int] = Field(default=None, ge=0, le=11)
    pressure_family_observed: Optional[Literal["Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"]] = None
    pressure_source: Optional[str] = None
    coverage: Optional[str] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class StructurePassV21(BaseModel):
    """Mechanical pre-snap observations with mutually exclusive eligible-player buckets."""

    back_count: Optional[int] = Field(default=None, ge=0, le=3)
    attached_te_count: Optional[int] = Field(default=None, ge=0, le=3)
    wing_hback_count: Optional[int] = Field(default=None, ge=0, le=3)
    flexed_te_count: Optional[int] = Field(default=None, ge=0, le=3)
    split_wr_count: Optional[int] = Field(default=None, ge=0, le=5)

    formation_family: Optional[Literal["Gun", "Pistol", "Under Center", "Goalline", "Other", "Unknown"]] = None
    initial_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    final_receiver_structure: Optional[Literal["2x2", "3x1", "2x1", "3x2", "Quads", "Unbalanced", "Other", "Unknown"]] = None
    initial_backfield_shape: Optional[Literal["Single back", "Split backs", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    final_backfield_shape: Optional[Literal["Single back", "Split backs", "Pistol dot", "Empty", "Other", "Unknown"]] = None
    formation_strength: Optional[Literal["Left", "Right", "Balanced", "Boundary", "Field", "Unknown"]] = None

    defensive_personnel: Optional[str] = None
    front_family: Optional[Literal["Even", "Odd", "Bear", "Mint/Tite", "Other", "Unknown"]] = None
    initial_line_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    initial_second_level_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    snap_line_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    snap_second_level_box_defenders: Optional[int] = Field(default=None, ge=0, le=11)
    shell: Optional[Literal["1-High", "2-High", "0-High", "Unknown"]] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class MovementPassV21(BaseModel):
    """Mechanical movement tracking from the first settled set to the instant before the snap."""

    any_player_moving_at_snap: Optional[bool] = None
    any_alignment_change_set_before_snap: Optional[bool] = None
    primary_mover: Optional[str] = None
    movement_type: Optional[Literal["Across", "Jet", "Orbit", "Return", "Short", "Out to slot/wide", "Into backfield", "Trade", "Other", "Unknown"]] = None
    movement_direction: Optional[str] = None
    movement_start_alignment: Optional[str] = None
    movement_end_alignment: Optional[str] = None

    rb_initial_side: Optional[Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]] = None
    rb_final_side: Optional[Literal["Left", "Right", "Behind", "Multiple", "None", "Unknown"]] = None
    rb_changed_sides: Optional[bool] = None
    rb_set_before_snap: Optional[bool] = None

    adjustment_trigger: Optional[Literal["None", "Motion", "Shift", "Defensive stem", "Cadence/check", "Other", "Unknown"]] = None
    adjustment_type: Optional[Literal["None", "Bump", "Travel", "Safety rotation", "Front shift", "Box insert", "Box remove", "Other", "Unknown"]] = None
    adjustment_player: Optional[str] = None
    adjustment_detail: Optional[str] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class PostSnapPassV21(BaseModel):
    """Post-snap recognition with mechanical rusher accounting."""

    film_play_type: Optional[Literal["Run", "Pass", "RPO", "Scramble", "Sack", "Other", "Unknown"]] = None
    run_concept: Optional[str] = None
    run_direction: Optional[Literal["Left", "Right", "Middle", "Boundary", "Field", "Unknown"]] = None
    pass_concept: Optional[str] = None
    rpo: Optional[bool] = None
    play_action: Optional[bool] = None

    immediate_rushers_count: Optional[int] = Field(default=None, ge=0, le=11)
    delayed_rushers_count: Optional[int] = Field(default=0, ge=0, le=11)
    bluff_or_drop_count: Optional[int] = Field(default=None, ge=0, le=11)
    nontraditional_rushers_count: Optional[int] = Field(default=None, ge=0, le=11)
    pressure_family_observed: Optional[Literal["Standard rush", "Blitz", "Sim pressure", "Creeper", "Zero pressure", "Unknown"]] = None
    pressure_source: Optional[str] = None
    coverage: Optional[str] = None

    confidence: float = Field(ge=0, le=1)
    uncertain_fields: list[str] = Field(default_factory=list)
    notes: str = ""


def _json_safe(value):
    """Convert pandas/numpy scalar values into normal JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _context(play_context: dict) -> dict:
    return {
        "chart_team": _json_safe(play_context.get("team")),
        "chart_team_role": _json_safe(play_context.get("chart_side")),
        "quarter": _json_safe(play_context.get("period")),
        "game_clock": _json_safe(play_context.get("clock")),
        "down": _json_safe(play_context.get("down")),
        "distance": _json_safe(play_context.get("distance")),
        "play_text": _json_safe(play_context.get("play_text")),
    }


def _prompt(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are an expert college football film analyst charting ONE snap for a scouting database.

Game context:
{json.dumps(context, indent=2)}

Watch the supplied video clip from beginning to end and chart BOTH the offense and defense.

Charting rules:
- Personnel means WHO is on the field, not where players align. If a RB motions
  from the backfield to receiver, 20 personnel remains 20 personnel.
- initial_formation / initial_backfield describe the earliest settled offensive
  alignment visible before motion or shift.
- final_formation / final_backfield describe the alignment at the snap.
- Motion means a player is still moving immediately before/through the snap.
  A shift is a change of alignment followed by the offense becoming set.
- Distinguish field and boundary when the broadcast angle supports it.
- Count defenders structurally committed to the run box.
- Blitz normally means 5+ rushers. Sim pressure or creeper can rush four while
  bringing a non-traditional rusher and dropping a traditional rush player.
- Coverage must be conservative. Use Unknown if the broadcast angle or clip does
  not show enough of the secondary after the snap.
- Use common football terminology for concepts only when supported by the film.
- The play-by-play is outcome context only. It must NOT override what the film
  shows for personnel, formation, motion, front, box, shell, pressure, or concept.
- Use Unknown/null instead of guessing.
- uncertain_fields must list fields that should receive human review.

Pay special attention to the several seconds immediately before the snap so that
motions, shifts, defensive bumps, safety rotations, and box changes are not missed.
"""


def _structure_prompt(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY the PRE-SNAP STRUCTURE pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Do not identify a play concept. Do not infer personnel shorthand such as 11 or 20.
Instead make mechanical visual observations.

OFFENSE:
1. Count RB/FB/H-back type backfield personnel on the field as rb_count.
   Count who is on the field, not where they are aligned at the snap.
2. Count true TE/Y/H players on the field as te_count, including a TE flexed out.
3. Identify QB alignment: Gun, Pistol, Under Center, etc.
4. Identify receiver distribution in the earliest settled set and again at the snap.
5. Identify RB/backfield alignment in the earliest settled set and at the snap.
6. Only call Field/Boundary strength when the broadcast view supports it.

DEFENSE:
1. Identify defensive personnel if visually supportable.
2. Classify the front only as Even, Odd, Bear, Mint/Tite, Other, or Unknown.
3. Count defenders structurally in the run box in the initial settled picture and at the snap.
4. Identify safety shell only as 0-High, 1-High, 2-High, or Unknown.

Important:
- A RB who shifts or motions out still counts as a RB for rb_count.
- Do not default to common personnel. If you cannot distinguish a TE from a WR, use null
  and list the field as uncertain.
- Do not use the play-by-play to infer formation or personnel.
- Prefer Unknown/null over guessing.
"""


def _movement_prompt(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY the PRE-SNAP MOVEMENT pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Watch from the first settled offensive alignment through the snap. Ignore the result
of the play except to identify the moment of the snap.

Definitions:
- MOTION: a player changes location and is still moving immediately before or through the snap.
- SHIFT: one or more offensive players change alignment, then the offense becomes set before the snap.
- A back moving from the QB's right side to the QB's left side and becoming set is a SHIFT,
  not motion.
- If both occur, mark both.

For any movement, describe the player generically when jersey/name is unclear:
RB, TE, slot, outside WR, etc. Record start and end alignment whenever visible.

Also watch how the defense reacts before the snap:
- bump
- travel
- safety rotation
- front shift
- box insert/remove
- other stem/check

Do not classify personnel, formation family, run/pass concept, or coverage in this pass.
Use Unknown/null instead of inventing movement that is not visible.
"""


def _postsnap_prompt(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY the POST-SNAP pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Determine:
- run/pass/RPO/scramble/sack
- run concept and direction only when the blocking/action clearly supports it
- pass concept only when the route structure is visible enough
- play action and RPO
- exact number of defenders who rush the passer
- observed pressure family and source
- coverage only when the broadcast angle gives enough post-snap evidence

Pressure rules:
- Five or more rushers is a blitz for our deterministic charting logic.
- Four rushers with a non-traditional rusher replacing a dropping DL can be sim pressure/creeper.
- Do not call a blitz merely because the QB is pressured.
- Count actual rushers after the snap, not defenders threatening at the line.

Coverage rules:
- Do not guess from the pre-snap shell alone.
- If deep defenders leave the broadcast frame before the coverage can be confirmed, use Unknown.
- Man/zone and exact coverage labels should be conservative.

The play-by-play may help with outcome context, but it must not decide concept, pressure,
rush count, or coverage for you. Prefer Unknown/null over a confident guess.
"""


def _structure_prompt_v21(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY a MECHANICAL PRE-SNAP STRUCTURE count for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Do NOT return personnel shorthand such as 10/11/20. Count the five non-QB, non-offensive-line
eligible skill players using MUTUALLY EXCLUSIVE buckets. Every eligible player should belong to
exactly one bucket:

- back_count: RB/FB players by body type/role, even if one is split wide.
- attached_te_count: true TE/Y aligned on or immediately next to the offensive line.
- wing_hback_count: TE/H-back aligned as a wing/off the tackle. Do NOT also count him as a back.
- flexed_te_count: true TE body/role detached into slot or wide alignment.
- split_wr_count: true WRs. Do not put a TE in this bucket merely because he is split out.

SANITY CHECK:
back_count + attached_te_count + wing_hback_count + flexed_te_count + split_wr_count should equal 5.
If you cannot visually distinguish TE vs WR, use null for the uncertain bucket rather than defaulting
to 11 personnel. The play-by-play must NOT be used to decide player positions.

Also chart:
- QB formation family.
- receiver distribution at the earliest settled alignment and again immediately before the snap.
- whether the backfield is single-back, split-backs, pistol-dot, or empty at those two moments.
- formation strength only if clearly supported.

DEFENSIVE BOX COUNT:
Do NOT give one intuitive 'box count'. Count two layers separately at the earliest settled picture
and again immediately before the snap:
1. line_box_defenders = defenders on/near the LOS who are structurally part of the run front,
   including edge defenders aligned tight enough to be a run-box player.
2. second_level_box_defenders = linebackers/safeties between roughly 2-6 yards of the LOS and
   inside the offensive core who are structurally committed to the box.

The Python code will add those layers to create the box total. Do not omit an edge defender merely
because he is just outside the offensive tackle.

Finally identify defensive personnel when visible, front family (Even/Odd/Bear/Mint-Tite/etc.),
and the PRE-SNAP safety shell only. Prefer Unknown/null to a default guess.
"""


def _movement_prompt_v21(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY a MECHANICAL PRE-SNAP MOVEMENT TRACK for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Your most important job is to compare TWO moments:
A) the earliest settled offensive alignment visible in the clip
B) the final instant immediately before the snap

For the RB/back specifically:
- record rb_initial_side relative to the QB: Left, Right, Behind, Multiple, None, or Unknown.
- record rb_final_side using the same choices.
- explicitly answer rb_changed_sides.
- explicitly answer rb_set_before_snap.

Example: if the RB starts to the QB's RIGHT, moves to the QB's LEFT, then becomes stationary
before the snap:
rb_initial_side=Right, rb_final_side=Left, rb_changed_sides=true,
rb_set_before_snap=true, any_alignment_change_set_before_snap=true,
any_player_moving_at_snap=false.

Definitions used by our code:
- any_player_moving_at_snap=true means MOTION.
- any_alignment_change_set_before_snap=true means a SHIFT occurred.
Both may be true on a play if a shift occurs and a different player later motions.

Also describe the primary mover's generic position, start/end alignment, direction and movement type.
Do not infer movement from the play result. Watch the pre-snap sequence itself frame by frame.

For the defense, record any pre-snap response: bump, travel, safety rotation, front shift,
box insert/remove, or other stem/check. Prefer Unknown/null over inventing movement.
"""


def _postsnap_prompt_v21(play_context: dict) -> str:
    context = _context(play_context)
    return f"""
You are doing ONLY the POST-SNAP pass for one college football snap.

Game context:
{json.dumps(context, indent=2)}

Chart run/pass/RPO/scramble/sack, concept only when clearly supported, play action, pressure,
and coverage. Keep the existing conservative coverage approach: pre-snap shell does NOT determine
post-snap coverage, and a defense may show 2-High then rotate to Cover 1.

RUSHER ACCOUNTING IS THE PRIORITY:
At the snap, track each defender who threatens the line.

- immediate_rushers_count: defenders whose first post-snap action commits them into the rush,
  attacking the LOS/backfield/pass protection.
- delayed_rushers_count: defenders who initially hesitate/fit/cover, then clearly add to the rush.
- bluff_or_drop_count: defenders who threaten pressure pre-snap but actually drop/cover.
- nontraditional_rushers_count: rushing DB/LB players replacing a traditional DL/edge who drops.

Do not count a defender as a rusher because he merely shows pressure before the snap.
Do not stop counting at the first four rushers. Follow the first several seconds after the snap.
The Python code will derive total rushers as immediate + delayed and will derive blitz=true at 5+.

Pressure family:
- 5+ actual rushers: normally Blitz (or Zero pressure if clearly zero-man pressure).
- 4 rushers with a nontraditional rusher replacing a dropping traditional rusher can be Sim/Creeper.
- QB pressure alone does not equal blitz.

Coverage:
- identify the actual post-snap coverage when the broadcast angle supports it.
- 2-High pre-snap rotating to Cover 1 post-snap is valid.
- use Unknown when deep coverage leaves the broadcast frame too early.

The play-by-play may provide outcome context only; it must not decide rush count, pressure, concept,
or coverage. Prefer Unknown/null over a confident guess.
"""


def _derive_personnel(rb_count: Optional[int], te_count: Optional[int]) -> Optional[str]:
    if rb_count is None or te_count is None:
        return "Unknown"
    mapping = {
        (1, 0): "10",
        (1, 1): "11",
        (1, 2): "12",
        (1, 3): "13",
        (2, 0): "20",
        (2, 1): "21",
        (2, 2): "22",
    }
    return mapping.get((rb_count, te_count), "Other")


def _derive_pressure(rushers: Optional[int], observed: Optional[str]) -> tuple[Optional[bool], Optional[str]]:
    if rushers is None:
        return None, observed or "Unknown"

    if rushers >= 5:
        return True, "Zero pressure" if observed == "Zero pressure" else "Blitz"

    if rushers == 4 and observed in {"Sim pressure", "Creeper"}:
        return False, observed

    return False, observed if observed not in {None, "Blitz"} else "Standard rush"


def _derive_personnel_v21(
    back_count: Optional[int],
    attached_te_count: Optional[int],
    wing_hback_count: Optional[int],
    flexed_te_count: Optional[int],
    split_wr_count: Optional[int],
) -> tuple[str, Optional[int], Optional[int]]:
    counts = [back_count, attached_te_count, wing_hback_count, flexed_te_count, split_wr_count]
    if any(value is None for value in counts):
        return "Unknown", None, None

    te_count = int(attached_te_count) + int(wing_hback_count) + int(flexed_te_count)
    eligible_total = int(back_count) + te_count + int(split_wr_count)
    if eligible_total != 5:
        return "Unknown", te_count, eligible_total

    personnel = _derive_personnel(int(back_count), te_count)
    return personnel or "Unknown", te_count, eligible_total


def _sum_optional(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None:
        return None
    return int(a) + int(b)


def _backfield_from_shape_and_side(shape: Optional[str], side: Optional[str]) -> Optional[str]:
    if shape in {"Split backs", "Pistol dot", "Empty"}:
        return shape
    if side == "Right":
        return "RB right"
    if side == "Left":
        return "RB left"
    if side == "Behind":
        return "Pistol dot"
    if side == "Multiple":
        return "Split backs"
    if side == "None":
        return "Empty"
    if shape in {"Other", "Unknown"}:
        return shape
    return "Unknown"


def _derive_movement_v21(movement: MovementPassV21) -> tuple[Optional[bool], Optional[bool]]:
    motion_present = movement.any_player_moving_at_snap
    shift_present = movement.any_alignment_change_set_before_snap

    if movement.rb_changed_sides is True and movement.rb_set_before_snap is True:
        shift_present = True
    if movement.rb_changed_sides is True and movement.rb_set_before_snap is False:
        motion_present = True

    return motion_present, shift_present


def _derive_rushers_v21(post: PostSnapPassV21) -> Optional[int]:
    if post.immediate_rushers_count is None:
        return None
    delayed = post.delayed_rushers_count or 0
    return int(post.immediate_rushers_count) + int(delayed)


def _merge_uncertain(*passes: BaseModel) -> list[str]:
    seen = []
    for item in passes:
        for field in getattr(item, "uncertain_fields", []) or []:
            if field not in seen:
                seen.append(field)
    return seen


def _pass_confidence_average(*passes: BaseModel) -> float:
    values = [float(getattr(item, "confidence")) for item in passes if getattr(item, "confidence", None) is not None]
    return sum(values) / len(values) if values else 0.0


def _call_structured(
    client,
    uploaded,
    model: str,
    fps: float,
    prompt: str,
    schema: type[BaseModel],
) -> BaseModel:
    interaction = None
    for attempt in range(4):
        try:
            interaction = client.interactions.create(
                model=model,
                input=[
                    {
                        "type": "video",
                        "uri": uploaded.uri,
                        "mime_type": uploaded.mime_type,
                        "processing": {"type": "static", "fps": float(fps)},
                    },
                    {"type": "text", "text": prompt},
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema.model_json_schema(),
                },
            )
            break
        except Exception as exc:
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in (
                    "503",
                    "service_unavailable",
                    "high demand",
                    "429",
                    "resource_exhausted",
                    "rate limit",
                )
            )
            if not retryable or attempt == 3:
                raise
            time.sleep(2 * (2 ** attempt))

    if interaction is None or not interaction.output_text:
        raise RuntimeError("Gemini returned no structured chart.")

    return schema.model_validate(json.loads(interaction.output_text))


def _upload_video(client, source: Path):
    uploaded = client.files.upload(file=str(source))
    while not uploaded.state or uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state and uploaded.state.name == "FAILED":
        raise RuntimeError("Gemini failed to process the uploaded video.")
    return uploaded


def analyze_clip_gemini(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 3.0,
) -> dict:
    """V1: one broad model pass over one snap."""
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)
        chart = _call_structured(client, uploaded, model, fps, _prompt(play_context), FootballSnapChart)
        result = chart.model_dump()
        result["_model"] = model
        result["_video_fps"] = float(fps)
        result["_provider"] = "google-gemini"
        result["_analyzer_version"] = "v1-single-pass"
        return result
    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v2(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V2: upload a snap once, then run three narrow visual tasks against the same video.

    Structure and movement are kept separate from post-snap concept recognition. Football
    shorthand that can be derived deterministically (personnel and blitz) is calculated
    in Python rather than guessed by the model.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        structure = _call_structured(
            client, uploaded, model, fps, _structure_prompt(play_context), StructurePass
        )
        time.sleep(1)
        movement = _call_structured(
            client, uploaded, model, fps, _movement_prompt(play_context), MovementPass
        )
        time.sleep(1)
        post = _call_structured(
            client, uploaded, model, fps, _postsnap_prompt(play_context), PostSnapPass
        )

        blitz, pressure_family = _derive_pressure(post.rushers, post.pressure_family_observed)
        personnel = _derive_personnel(structure.rb_count, structure.te_count)

        result = {
            "personnel": personnel,
            "formation_family": structure.formation_family,
            "initial_formation": structure.initial_receiver_structure,
            "initial_backfield": structure.initial_backfield,
            "final_formation": structure.final_receiver_structure,
            "final_backfield": structure.final_backfield,
            "formation_strength": structure.formation_strength,

            "motion_present": movement.motion_present,
            "motion_player": movement.motion_player,
            "motion_type": movement.motion_type,
            "motion_direction": movement.motion_direction,
            "motion_start_alignment": movement.motion_start_alignment,
            "motion_end_alignment": movement.motion_end_alignment,
            "shift_present": movement.shift_present,
            "shift_description": movement.shift_description,

            "film_play_type": post.film_play_type,
            "run_concept": post.run_concept,
            "run_direction": post.run_direction,
            "pass_concept": post.pass_concept,
            "rpo": post.rpo,
            "play_action": post.play_action,

            "defensive_personnel": structure.defensive_personnel,
            "front": structure.front_family,
            "initial_box_count": structure.initial_box_count,
            "snap_box_count": structure.snap_box_count,
            "shell": structure.shell,
            "coverage": post.coverage,
            "rushers": post.rushers,
            "blitz": blitz,
            "pressure_family": pressure_family,
            "pressure_source": post.pressure_source,

            "adjustment_trigger": movement.adjustment_trigger,
            "adjustment_type": movement.adjustment_type,
            "adjustment_player": movement.adjustment_player,
            "adjustment_detail": movement.adjustment_detail,

            "overall_confidence": _pass_confidence_average(structure, movement, post),
            "uncertain_fields": _merge_uncertain(structure, movement, post),
            "analysis_notes": " | ".join(
                note for note in (structure.notes, movement.notes, post.notes) if note
            ),

            "_model": model,
            "_video_fps": float(fps),
            "_provider": "google-gemini",
            "_analyzer_version": "v2-specialized",
            "_derived": {
                "rb_count": structure.rb_count,
                "te_count": structure.te_count,
                "personnel_rule": "RB/TE counts -> personnel grouping",
                "blitz_rule": "5+ actual rushers -> blitz",
            },
            "_pass_confidence": {
                "structure": structure.confidence,
                "movement": movement.confidence,
                "post_snap": post.confidence,
            },
            "_passes": {
                "structure": structure.model_dump(),
                "movement": movement.model_dump(),
                "post_snap": post.model_dump(),
            },
        }
        return result
    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_gemini_v21(
    clip_path: str | Path,
    play_context: dict,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    fps: float = 5.0,
) -> dict:
    """
    V2.1: three specialized calls, but with mechanical counting and deterministic derivation.

    The video is uploaded once. Personnel, backfield changes, box totals and blitz are calculated
    from lower-level visual observations instead of asking Gemini for the final shorthand directly.
    """
    source = Path(clip_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Clip not found: {source}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = _upload_video(client, source)

        structure = _call_structured(
            client, uploaded, model, fps, _structure_prompt_v21(play_context), StructurePassV21
        )
        time.sleep(1)
        movement = _call_structured(
            client, uploaded, model, fps, _movement_prompt_v21(play_context), MovementPassV21
        )
        time.sleep(1)
        post = _call_structured(
            client, uploaded, model, fps, _postsnap_prompt_v21(play_context), PostSnapPassV21
        )

        personnel, te_count, eligible_total = _derive_personnel_v21(
            structure.back_count,
            structure.attached_te_count,
            structure.wing_hback_count,
            structure.flexed_te_count,
            structure.split_wr_count,
        )

        initial_box_count = _sum_optional(
            structure.initial_line_box_defenders,
            structure.initial_second_level_box_defenders,
        )
        snap_box_count = _sum_optional(
            structure.snap_line_box_defenders,
            structure.snap_second_level_box_defenders,
        )

        initial_backfield = _backfield_from_shape_and_side(
            structure.initial_backfield_shape,
            movement.rb_initial_side,
        )
        final_backfield = _backfield_from_shape_and_side(
            structure.final_backfield_shape,
            movement.rb_final_side,
        )

        motion_present, shift_present = _derive_movement_v21(movement)
        rushers = _derive_rushers_v21(post)
        blitz, pressure_family = _derive_pressure(rushers, post.pressure_family_observed)

        shift_description = None
        if shift_present:
            if movement.rb_changed_sides:
                shift_description = (
                    f"RB changed from {movement.rb_initial_side} to {movement.rb_final_side} "
                    f"and {'set' if movement.rb_set_before_snap else 'did not set'} before snap"
                )
            elif movement.primary_mover:
                shift_description = (
                    f"{movement.primary_mover}: {movement.movement_start_alignment} -> "
                    f"{movement.movement_end_alignment}"
                )

        result = {
            "personnel": personnel,
            "formation_family": structure.formation_family,
            "initial_formation": structure.initial_receiver_structure,
            "initial_backfield": initial_backfield,
            "final_formation": structure.final_receiver_structure,
            "final_backfield": final_backfield,
            "formation_strength": structure.formation_strength,

            "motion_present": motion_present,
            "motion_player": movement.primary_mover if motion_present else None,
            "motion_type": movement.movement_type if motion_present else None,
            "motion_direction": movement.movement_direction if motion_present else None,
            "motion_start_alignment": movement.movement_start_alignment if motion_present else None,
            "motion_end_alignment": movement.movement_end_alignment if motion_present else None,
            "shift_present": shift_present,
            "shift_description": shift_description,

            "film_play_type": post.film_play_type,
            "run_concept": post.run_concept,
            "run_direction": post.run_direction,
            "pass_concept": post.pass_concept,
            "rpo": post.rpo,
            "play_action": post.play_action,

            "defensive_personnel": structure.defensive_personnel,
            "front": structure.front_family,
            "initial_box_count": initial_box_count,
            "snap_box_count": snap_box_count,
            "shell": structure.shell,
            "coverage": post.coverage,
            "rushers": rushers,
            "blitz": blitz,
            "pressure_family": pressure_family,
            "pressure_source": post.pressure_source,

            "adjustment_trigger": movement.adjustment_trigger,
            "adjustment_type": movement.adjustment_type,
            "adjustment_player": movement.adjustment_player,
            "adjustment_detail": movement.adjustment_detail,

            "overall_confidence": _pass_confidence_average(structure, movement, post),
            "uncertain_fields": _merge_uncertain(structure, movement, post),
            "analysis_notes": " | ".join(
                note for note in (structure.notes, movement.notes, post.notes) if note
            ),

            "_model": model,
            "_video_fps": float(fps),
            "_provider": "google-gemini",
            "_analyzer_version": "v2.1-mechanical",
            "_derived": {
                "back_count": structure.back_count,
                "attached_te_count": structure.attached_te_count,
                "wing_hback_count": structure.wing_hback_count,
                "flexed_te_count": structure.flexed_te_count,
                "te_count": te_count,
                "split_wr_count": structure.split_wr_count,
                "eligible_total": eligible_total,
                "personnel_rule": "5 eligible-player buckets -> RB/TE personnel grouping",
                "rb_initial_side": movement.rb_initial_side,
                "rb_final_side": movement.rb_final_side,
                "rb_changed_sides": movement.rb_changed_sides,
                "rb_set_before_snap": movement.rb_set_before_snap,
                "initial_line_box_defenders": structure.initial_line_box_defenders,
                "initial_second_level_box_defenders": structure.initial_second_level_box_defenders,
                "snap_line_box_defenders": structure.snap_line_box_defenders,
                "snap_second_level_box_defenders": structure.snap_second_level_box_defenders,
                "immediate_rushers_count": post.immediate_rushers_count,
                "delayed_rushers_count": post.delayed_rushers_count,
                "bluff_or_drop_count": post.bluff_or_drop_count,
                "nontraditional_rushers_count": post.nontraditional_rushers_count,
                "blitz_rule": "immediate + delayed actual rushers; 5+ -> blitz",
            },
            "_pass_confidence": {
                "structure": structure.confidence,
                "movement": movement.confidence,
                "post_snap": post.confidence,
            },
            "_passes": {
                "structure": structure.model_dump(),
                "movement": movement.model_dump(),
                "post_snap": post.model_dump(),
            },
        }
        return result
    finally:
        if uploaded is not None and uploaded.name:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def analyze_clip_stub(clip_path: str, play_context: dict) -> FilmObservation:
    """Legacy placeholder retained for provider-neutral architecture tests."""
    return FilmObservation(
        play_id=str(play_context.get("playId")) if play_context.get("playId") is not None else None,
        game_id=str(play_context.get("gameId")) if play_context.get("gameId") is not None else None,
        notes=f"Video analysis not configured yet for {clip_path}",
    )
