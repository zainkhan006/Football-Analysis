"""
Feature 10 — Formation Analysis
================================
Uses Feature 8's team_labels.json to detect formations based on X-coordinate sorting.

Complete Logic:
1. Calculate defense for BOTH teams first (using own side detection)
2. For midfield/attack, use OPPONENT'S defense average to decide last midfielder
3. Attack has NO min/max limits - determined naturally by positioning

Defense: min=3, max=4 (enforced)
Midfield: min=3, max=4 (enforced with proximity check using opponent's defense)
Attack: NO limits (natural positioning)

Usage:
    python feature_10.py --seq v_HdiyOtliFiw_c003 --frames 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
import numpy as np
import cv2

# Import helpers from feature_8
import sys
sys.path.append(str(Path(__file__).parent))

from feature_8 import (
    _load_all_gt,
    _collect_frame_paths,
    _frame_path_for_idx,
    _frame_index_of,
)

try:
    import config
    _DATASET_ROOT = config.datasetRoot / config.testSplit
except ImportError:
    config = None
    _DATASET_ROOT = Path(r"C:\Users\samee\Documents\Computer Vision\project\videos")


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FormationConfig:
    smoothing_window: int = 10  # frames for formation smoothing
    dot_radius: int = 7
    line_thickness: int = 3
    
    # Min/Max limits for Defense and Midfield (Attack has NO limits)
    defense_min: int = 3
    defense_max: int = 4
    midfield_min: int = 3
    midfield_max: int = 4
    
    # Colors (BGR) - ONE COLOR PER TEAM
    home_color: Tuple[int, int, int] = (60, 200, 60)      # GREEN for HOME
    away_color: Tuple[int, int, int] = (0, 100, 255)      # ORANGE for AWAY
    text_color: Tuple[int, int, int] = (255, 255, 255)    # White


# ══════════════════════════════════════════════════════════════════════
# Get bottom center of bounding box (player's feet position)
# ══════════════════════════════════════════════════════════════════════

def get_bottom_center(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    """Return (x, y) of player's feet position (bottom center of bbox)."""
    return (x + w // 2, y + h)


def get_player_positions_from_gt(
    entries: List[Tuple[int, Tuple[int, int, int, int]]],
    track_ids_to_include: set
) -> Dict[int, Tuple[int, int]]:
    """Get bottom center positions for specific track IDs."""
    positions = {}
    for tid, (x, y, w, h) in entries:
        if tid in track_ids_to_include:
            positions[tid] = get_bottom_center(x, y, w, h)
    return positions


# ══════════════════════════════════════════════════════════════════════
# Determine which team is on which side of the frame
# ══════════════════════════════════════════════════════════════════════

def determine_team_sides(
    home_positions: Dict[int, Tuple[int, int]],
    away_positions: Dict[int, Tuple[int, int]],
    frame_width: int
) -> Tuple[str, str]:
    """Determine which team is on left/right side of the frame."""
    home_x_avg = np.mean([x for x, y in home_positions.values()]) if home_positions else frame_width / 2
    away_x_avg = np.mean([x for x, y in away_positions.values()]) if away_positions else frame_width / 2
    
    if home_x_avg < away_x_avg:
        return "left", "right"
    else:
        return "right", "left"


# ══════════════════════════════════════════════════════════════════════
# Calculate Defense Line (using only own side, no opponent reference)
# ══════════════════════════════════════════════════════════════════════

def calculate_defense_line(
    players: List[Tuple[int, int, int]],  # (track_id, x, y)
    team_side: str,  # "left" or "right"
    cfg: FormationConfig
) -> Tuple[List[int], List[Tuple[int, int, int]]]:
    """
    Calculate defense line using own side only.
    
    For LEFT side: take smallest X values (closest to left touchline)
    For RIGHT side: take largest X values (closest to right touchline)
    
    Returns:
        defensive_ids, remaining_players
    """
    if len(players) < cfg.defense_min:
        # Not enough players, all are defense
        return [p[0] for p in players], []
    
    # Sort by X
    sorted_players = sorted(players, key=lambda p: p[1])
    
    if team_side == "right":
        # Right side: defense is largest X, so reverse
        sorted_players = list(reversed(sorted_players))
    
    # Defense size = min(max, total - minimum needed for midfield)
    min_needed_for_midfield_attack = cfg.midfield_min
    defense_size = min(cfg.defense_max, len(sorted_players) - min_needed_for_midfield_attack)
    defense_size = max(cfg.defense_min, defense_size)
    
    defensive_ids = [p[0] for p in sorted_players[:defense_size]]
    remaining = sorted_players[defense_size:]
    
    return defensive_ids, remaining


# ══════════════════════════════════════════════════════════════════════
# Calculate Midfield and Attack using opponent's defense average
# ══════════════════════════════════════════════════════════════════════

def calculate_midfield_and_attack(
    remaining_players: List[Tuple[int, int, int]],
    team_side: str,
    opponent_defense_avg_x: float,
    cfg: FormationConfig
) -> Tuple[List[int], List[int]]:
    """
    Calculate midfield and attack using opponent's defense average.
    
    Logic:
    1. Take tentative midfield = max - 1 (3 players)
    2. Compute own midfield average
    3. Check last midfielder against opponent's defense average
       - If closer to opponent defense → move to attack
       - If closer to own midfield → keep in midfield
    4. All remaining players go to attack
    
    Returns:
        midfield_ids, attacking_ids
    """
    if not remaining_players:
        return [], []
    
    # Sort by X
    sorted_players = sorted(remaining_players, key=lambda p: p[1])
    
    if team_side == "right":
        # Right side: reverse for consistent ordering
        sorted_players = list(reversed(sorted_players))
    
    total = len(sorted_players)
    
    # Tentative midfield = max - 1
    midfield_tentative_size = cfg.midfield_max - 1
    if midfield_tentative_size > total:
        midfield_tentative_size = total
    
    # Get tentative midfield players
    midfield_players = sorted_players[:midfield_tentative_size]
    midfield_avg_x = np.mean([p[1] for p in midfield_players]) if midfield_players else 0
    
    midfield_size = midfield_tentative_size
    move_to_attack = False
    
    # Check last midfielder if we have opponent defense average
    if len(midfield_players) > 0 and opponent_defense_avg_x > 0:
        last_midfielder = midfield_players[-1]
        last_midfielder_x = last_midfielder[1]
        
        dist_to_own_midfield = abs(last_midfielder_x - midfield_avg_x)
        dist_to_opponent_defense = abs(last_midfielder_x - opponent_defense_avg_x)
        
        if dist_to_opponent_defense < dist_to_own_midfield:
            # Last midfielder belongs to attack
            midfield_size = midfield_tentative_size - 1
            move_to_attack = True
    
    # Finalize midfield
    if move_to_attack:
        midfield_players = midfield_players[:-1]
    
    midfield_ids = [p[0] for p in midfield_players]
    
    # ALL remaining players go to attack (no min/max limits)
    remaining_after_midfield = sorted_players[len(midfield_ids):]
    attacking_ids = [p[0] for p in remaining_after_midfield]
    
    return midfield_ids, attacking_ids


# ══════════════════════════════════════════════════════════════════════
# Main grouping function for one team
# ══════════════════════════════════════════════════════════════════════

def group_team_players(
    players: List[Tuple[int, int, int]],
    team_side: str,
    opponent_defense_avg_x: float,
    cfg: FormationConfig
) -> Tuple[List[int], List[int], List[int]]:
    """
    Group players into defense, midfield, attack for one team.
    
    Steps:
    1. Calculate defense line (using own side only)
    2. Calculate midfield and attack using opponent's defense average
    3. Enforce midfield min limit (if below min, move from attack)
    4. Enforce defense min/max (already enforced)
    
    Returns:
        defensive_ids, midfield_ids, attacking_ids
    """
    if len(players) < 3:
        return [p[0] for p in players], [], []
    
    # Step 1: Defense line
    defensive_ids, remaining = calculate_defense_line(players, team_side, cfg)
    
    if not remaining:
        return defensive_ids, [], []
    
    # Step 2: Midfield and Attack using opponent's defense average
    midfield_ids, attacking_ids = calculate_midfield_and_attack(
        remaining, team_side, opponent_defense_avg_x, cfg
    )
    
    # Step 3: Enforce midfield minimum
    # If midfield < min and attack has players, move from attack to midfield
    if len(midfield_ids) < cfg.midfield_min and attacking_ids:
        # Need to move players from attack to midfield
        needed = cfg.midfield_min - len(midfield_ids)
        if needed <= len(attacking_ids):
            # Get the players to move (those with X closest to midfield)
            # For left side: smaller X, for right side: larger X
            attack_players = [(tid, x, y) for tid, x, y in remaining if tid in attacking_ids]
            attack_players_sorted = sorted(attack_players, key=lambda p: p[1])
            
            if team_side == "right":
                attack_players_sorted = list(reversed(attack_players_sorted))
            
            players_to_move = attack_players_sorted[:needed]
            midfield_ids.extend([p[0] for p in players_to_move])
            attacking_ids = [tid for tid in attacking_ids if tid not in midfield_ids]
    
    return defensive_ids, midfield_ids, attacking_ids


def formation_to_string(defensive: List[int], midfield: List[int], attacking: List[int]) -> str:
    """Convert line counts to formation string."""
    return f"{len(defensive)}-{len(midfield)}-{len(attacking)}"


# ══════════════════════════════════════════════════════════════════════
# Load team labels from Feature 8 JSON
# ══════════════════════════════════════════════════════════════════════

def load_team_labels(seq_out: Path, frame_indices: List[int]) -> Tuple[set, set, Dict[int, Dict[int, str]]]:
    """Load team labels from Feature 8's team_labels.json."""
    json_path = seq_out / "team_labels.json"
    
    if not json_path.exists():
        print(f"  Warning: {json_path} not found. Run feature_8.py first.")
        return set(), set(), {}
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    home_track_ids = set(data.get("home_track_ids", []))
    away_track_ids = set(data.get("away_track_ids", []))
    
    per_frame_labels = {}
    for fi_str, labels in data.get("per_frame", {}).items():
        fi = int(fi_str)
        if fi in frame_indices:
            per_frame_labels[fi] = {int(tid): lbl for tid, lbl in labels.items()}
    
    return home_track_ids, away_track_ids, per_frame_labels


# ══════════════════════════════════════════════════════════════════════
# Draw formation on frame (ONE COLOR PER TEAM)
# ══════════════════════════════════════════════════════════════════════

def draw_formation_on_frame(
    frame: np.ndarray,
    positions: Dict[int, Tuple[int, int]],
    defensive_ids: List[int],
    midfield_ids: List[int],
    attacking_ids: List[int],
    team_color: Tuple[int, int, int],
    team_name: str,
    formation_str: str,
    team_side: str
) -> np.ndarray:
    """Draw dots and connecting lines using ONE color per team."""
    img = frame.copy()
    
    def get_line_positions(line_ids):
        return [(tid, positions[tid]) for tid in line_ids if tid in positions]
    
    defense = get_line_positions(defensive_ids)
    midfield = get_line_positions(midfield_ids)
    attack = get_line_positions(attacking_ids)
    
    defense.sort(key=lambda p: p[1][0])
    midfield.sort(key=lambda p: p[1][0])
    attack.sort(key=lambda p: p[1][0])
    
    # Draw connecting lines
    for line in [defense, midfield, attack]:
        for i in range(len(line) - 1):
            x1, y1 = line[i][1]
            x2, y2 = line[i + 1][1]
            cv2.line(img, (x1, y1), (x2, y2), team_color, FormationConfig.line_thickness)
    
    # Draw dots at player feet
    for tid, (x, y) in positions.items():
        cv2.circle(img, (x, y), FormationConfig.dot_radius, (255, 255, 255), -1)
        cv2.circle(img, (x, y), FormationConfig.dot_radius, team_color, 2)
        cv2.putText(img, f"#{tid}", (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, team_color, 1)
    
    return img


def draw_header(
    frame: np.ndarray,
    home_formation: str,
    away_formation: str,
    home_color: Tuple[int, int, int],
    away_color: Tuple[int, int, int]
) -> np.ndarray:
    """Draw header at top of frame showing formations with team colors."""
    h, w = frame.shape[:2]
    header_h = 70
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    
    cv2.putText(header, "FORMATION ANALYSIS", (w // 2 - 100, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(header, f"HOME: {home_formation}", (20, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, home_color, 2)
    
    cv2.putText(header, f"AWAY: {away_formation}", (w - 180, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, away_color, 2)
    
    return np.vstack([header, frame])


# ══════════════════════════════════════════════════════════════════════
# Print coordinates
# ══════════════════════════════════════════════════════════════════════

def print_coordinates(
    frame_idx: int,
    home_positions: Dict[int, Tuple[int, int]],
    away_positions: Dict[int, Tuple[int, int]],
    home_formation: str,
    away_formation: str,
    home_lines: Tuple[List[int], List[int], List[int]],
    away_lines: Tuple[List[int], List[int], List[int]]
) -> None:
    """Print all player coordinates with their IDs and line assignments."""
    home_def, home_mid, home_att = home_lines
    away_def, away_mid, away_att = away_lines
    
    print(f"\n{'='*80}")
    print(f"FRAME {frame_idx:06d}")
    print(f"{'='*80}")
    
    print(f"\n🏠 HOME TEAM (Formation: {home_formation})")
    print(f"   Defense ({len(home_def)}): {home_def}")
    print(f"   Midfield ({len(home_mid)}): {home_mid}")
    print(f"   Attack ({len(home_att)}): {home_att}")
    print(f"{'-'*60}")
    print(f"{'Track ID':<10} {'X':<10} {'Y':<10} {'Line':<12}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    
    for tid, (x, y) in sorted(home_positions.items()):
        if tid in home_def:
            line = "DEFENSE"
        elif tid in home_mid:
            line = "MIDFIELD"
        elif tid in home_att:
            line = "ATTACK"
        else:
            line = "UNKNOWN"
        print(f"{tid:<10} {x:<10} {y:<10} {line:<12}")
    
    print(f"\n✈️ AWAY TEAM (Formation: {away_formation})")
    print(f"   Defense ({len(away_def)}): {away_def}")
    print(f"   Midfield ({len(away_mid)}): {away_mid}")
    print(f"   Attack ({len(away_att)}): {away_att}")
    print(f"{'-'*60}")
    print(f"{'Track ID':<10} {'X':<10} {'Y':<10} {'Line':<12}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    
    for tid, (x, y) in sorted(away_positions.items()):
        if tid in away_def:
            line = "DEFENSE"
        elif tid in away_mid:
            line = "MIDFIELD"
        elif tid in away_att:
            line = "ATTACK"
        else:
            line = "UNKNOWN"
        print(f"{tid:<10} {x:<10} {y:<10} {line:<12}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Home: D={len(home_def)} M={len(home_mid)} A={len(home_att)} (Total: {len(home_positions)})")
    print(f"  Away: D={len(away_def)} M={len(away_mid)} A={len(away_att)} (Total: {len(away_positions)})")
    print(f"{'='*80}\n")


# ══════════════════════════════════════════════════════════════════════
# Main processing
# ══════════════════════════════════════════════════════════════════════

def process_sequence(
    seq_name: str,
    n_frames: int = 300,
    verbose: bool = True,
    print_coords: bool = True
) -> None:
    """Process sequence and generate formation analysis."""
    here = Path(__file__).parent
    seq_dir = _DATASET_ROOT / seq_name
    f8_out = here / "output" / "feature_8" / seq_name
    output_dir = here / "output" / "feature_10" / seq_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    
    if not img_dir.is_dir():
        print(f"Error: img1/ not found at {img_dir}")
        return
    if not gt_path.is_file():
        print(f"Error: gt.txt not found at {gt_path}")
        return
    
    all_paths = _collect_frame_paths(str(img_dir))
    all_indices = [_frame_index_of(p) for p in all_paths if _frame_index_of(p) >= 0]
    frame_indices = sorted(all_indices)[:n_frames]
    
    if not frame_indices:
        print("No frames found")
        return
    
    cfg = FormationConfig()
    
    print(f"\n{'='*60}")
    print(f"FEATURE 10 - FORMATION ANALYSIS")
    print(f"{'='*60}")
    print(f"  Sequence: {seq_name}")
    print(f"  Frames: {len(frame_indices)}")
    print(f"  Limits: Defense={cfg.defense_min}-{cfg.defense_max}, Midfield={cfg.midfield_min}-{cfg.midfield_max}")
    print(f"  Attack: NO limits (natural positioning)")
    print(f"  Logic: Defense first for both teams, then midfield using opponent's defense avg")
    print(f"  Colors: HOME=GREEN, AWAY=ORANGE")
    print(f"{'='*60}\n")
    
    home_track_ids, away_track_ids, per_frame_labels = load_team_labels(f8_out, set(frame_indices))
    
    if not home_track_ids and not away_track_ids:
        print("  ERROR: No team labels found. Run feature_8.py first.")
        return
    
    print(f"  Home team tracks: {len(home_track_ids)} - {sorted(home_track_ids)}")
    print(f"  Away team tracks: {len(away_track_ids)} - {sorted(away_track_ids)}")
    
    frame_gt = _load_all_gt(str(gt_path))
    
    home_formations: Dict[int, str] = {}
    away_formations: Dict[int, str] = {}
    
    home_formation_history: List[str] = []
    away_formation_history: List[str] = []
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frames_saved = 0
    coordinate_log = output_dir / "coordinates_log.txt"
    
    log_file = open(coordinate_log, "w") if print_coords else None
    if log_file:
        log_file.write(f"FEATURE 10 - COORDINATES LOG\n")
        log_file.write(f"Sequence: {seq_name}\n")
        log_file.write(f"Limits: Defense={cfg.defense_min}-{cfg.defense_max}, Midfield={cfg.midfield_min}-{cfg.midfield_max}\n")
        log_file.write(f"Attack: NO limits\n")
        log_file.write(f"{'='*60}\n\n")
    
    for fi in frame_indices:
        fp = _frame_path_for_idx(all_paths, fi)
        if fp is None:
            continue
        
        frame = cv2.imread(fp)
        if frame is None:
            continue
        
        entries = frame_gt.get(fi, [])
        if not entries:
            continue
        
        frame_width = frame.shape[1]
        
        if fi in per_frame_labels:
            frame_labels = per_frame_labels[fi]
            home_tids_in_frame = [tid for tid, lbl in frame_labels.items() if lbl == "home"]
            away_tids_in_frame = [tid for tid, lbl in frame_labels.items() if lbl == "away"]
        else:
            home_tids_in_frame = [tid for tid in home_track_ids if any(tid == e[0] for e in entries)]
            away_tids_in_frame = [tid for tid in away_track_ids if any(tid == e[0] for e in entries)]
        
        home_positions = get_player_positions_from_gt(entries, set(home_tids_in_frame))
        away_positions = get_player_positions_from_gt(entries, set(away_tids_in_frame))
        
        # Determine sides
        if home_positions and away_positions:
            home_side, away_side = determine_team_sides(home_positions, away_positions, frame_width)
        elif home_positions:
            home_side, away_side = "left", "right"
        else:
            home_side, away_side = "left", "right"
        
        # Convert positions to list format for processing
        home_players_list = [(tid, x, y) for tid, (x, y) in home_positions.items()]
        away_players_list = [(tid, x, y) for tid, (x, y) in away_positions.items()]
        
        # Calculate defense for both teams first (to get opponent defense averages)
        home_defense_ids, home_remaining = calculate_defense_line(home_players_list, home_side, cfg)
        away_defense_ids, away_remaining = calculate_defense_line(away_players_list, away_side, cfg)
        
        # Calculate opponent defense averages
        home_defense_avg_x = np.mean([p[1] for p in home_players_list if p[0] in home_defense_ids]) if home_defense_ids else 0
        away_defense_avg_x = np.mean([p[1] for p in away_players_list if p[0] in away_defense_ids]) if away_defense_ids else 0
        
        # Now calculate midfield and attack using opponent's defense average
        if home_remaining:
            home_midfield_ids, home_attacking_ids = calculate_midfield_and_attack(
                home_remaining, home_side, away_defense_avg_x, cfg
            )
        else:
            home_midfield_ids, home_attacking_ids = [], []
        
        if away_remaining:
            away_midfield_ids, away_attacking_ids = calculate_midfield_and_attack(
                away_remaining, away_side, home_defense_avg_x, cfg
            )
        else:
            away_midfield_ids, away_attacking_ids = [], []
        
        # Enforce midfield minimum for home
        if len(home_midfield_ids) < cfg.midfield_min and home_attacking_ids:
            needed = cfg.midfield_min - len(home_midfield_ids)
            if needed <= len(home_attacking_ids):
                # Move closest players from attack to midfield
                attack_players = [(tid, x, y) for tid, x, y in home_players_list if tid in home_attacking_ids]
                attack_players_sorted = sorted(attack_players, key=lambda p: p[1])
                if home_side == "right":
                    attack_players_sorted = list(reversed(attack_players_sorted))
                players_to_move = attack_players_sorted[:needed]
                home_midfield_ids.extend([p[0] for p in players_to_move])
                home_attacking_ids = [tid for tid in home_attacking_ids if tid not in home_midfield_ids]
        
        # Enforce midfield minimum for away
        if len(away_midfield_ids) < cfg.midfield_min and away_attacking_ids:
            needed = cfg.midfield_min - len(away_midfield_ids)
            if needed <= len(away_attacking_ids):
                attack_players = [(tid, x, y) for tid, x, y in away_players_list if tid in away_attacking_ids]
                attack_players_sorted = sorted(attack_players, key=lambda p: p[1])
                if away_side == "right":
                    attack_players_sorted = list(reversed(attack_players_sorted))
                players_to_move = attack_players_sorted[:needed]
                away_midfield_ids.extend([p[0] for p in players_to_move])
                away_attacking_ids = [tid for tid in away_attacking_ids if tid not in away_midfield_ids]
        
        home_formation_str = formation_to_string(home_defense_ids, home_midfield_ids, home_attacking_ids)
        away_formation_str = formation_to_string(away_defense_ids, away_midfield_ids, away_attacking_ids)
        
        home_lines = (home_defense_ids, home_midfield_ids, home_attacking_ids)
        away_lines = (away_defense_ids, away_midfield_ids, away_attacking_ids)
        
        # Smoothing
        home_formation_history.append(home_formation_str)
        if len(home_formation_history) > cfg.smoothing_window:
            home_formation_history.pop(0)
        if home_formation_history:
            home_smoothed = Counter(home_formation_history).most_common(1)[0][0]
        else:
            home_smoothed = home_formation_str
        home_formations[fi] = home_smoothed
        
        away_formation_history.append(away_formation_str)
        if len(away_formation_history) > cfg.smoothing_window:
            away_formation_history.pop(0)
        if away_formation_history:
            away_smoothed = Counter(away_formation_history).most_common(1)[0][0]
        else:
            away_smoothed = away_formation_str
        away_formations[fi] = away_smoothed
        
        # Draw on frame
        frame = draw_formation_on_frame(
            frame, home_positions, home_defense_ids, home_midfield_ids, home_attacking_ids,
            cfg.home_color, "HOME", home_smoothed, home_side
        )
        frame = draw_formation_on_frame(
            frame, away_positions, away_defense_ids, away_midfield_ids, away_attacking_ids,
            cfg.away_color, "AWAY", away_smoothed, away_side
        )
        frame = draw_header(frame, home_smoothed, away_smoothed, cfg.home_color, cfg.away_color)
        
        # Print coordinates
        if print_coords and (n_frames <= 50 or frames_saved % 10 == 0 or frames_saved < 5):
            print_coordinates(fi, home_positions, away_positions, 
                            home_smoothed, away_smoothed,
                            home_lines, away_lines)
        
        # Log to file
        if log_file:
            log_file.write(f"FRAME {fi:06d}\n")
            log_file.write(f"{'='*50}\n")
            log_file.write(f"HOME: {home_smoothed} ({home_side} side)\n")
            for tid, (x, y) in sorted(home_positions.items()):
                if tid in home_defense_ids:
                    line = "DEFENSE"
                elif tid in home_midfield_ids:
                    line = "MIDFIELD"
                elif tid in home_attacking_ids:
                    line = "ATTACK"
                else:
                    line = "UNKNOWN"
                log_file.write(f"  #{tid}: ({x},{y}) - {line}\n")
            log_file.write(f"AWAY: {away_smoothed} ({away_side} side)\n")
            for tid, (x, y) in sorted(away_positions.items()):
                if tid in away_defense_ids:
                    line = "DEFENSE"
                elif tid in away_midfield_ids:
                    line = "MIDFIELD"
                elif tid in away_attacking_ids:
                    line = "ATTACK"
                else:
                    line = "UNKNOWN"
                log_file.write(f"  #{tid}: ({x},{y}) - {line}\n")
            log_file.write(f"\n")
        
        # Save frame
        frame_path = frames_dir / f"frame_{fi:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frames_saved += 1
        
        if frames_saved % 50 == 0 and verbose:
            print(f"  Processed {frames_saved} frames...")
    
    if log_file:
        log_file.close()
        print(f"\n  Coordinates log saved to: {coordinate_log}")
    
    # Save results
    results = {
        "sequence": seq_name,
        "frames_processed": len(frame_indices),
        "limits": {
            "defense_min": cfg.defense_min,
            "defense_max": cfg.defense_max,
            "midfield_min": cfg.midfield_min,
            "midfield_max": cfg.midfield_max,
            "attack": "NO limits"
        },
        "logic": "Defense first for both teams, then midfield using opponent's defense average",
        "home_track_ids": list(home_track_ids),
        "away_track_ids": list(away_track_ids),
        "home_formations": {str(fi): formation for fi, formation in home_formations.items()},
        "away_formations": {str(fi): formation for fi, formation in away_formations.items()}
    }
    
    if home_formations:
        home_common = Counter(home_formations.values()).most_common(1)[0]
        print(f"\n  HOME most common formation: {home_common[0]} (in {home_common[1]} frames)")
    if away_formations:
        away_common = Counter(away_formations.values()).most_common(1)[0]
        print(f"  AWAY most common formation: {away_common[0]} (in {away_common[1]} frames)")
    
    results_path = output_dir / "formation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {results_path}")
    print(f"  Frames saved to: {frames_dir}")
    print(f"\n{'='*60}")
    print("FEATURE 10 COMPLETE")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature 10 - Formation Analysis")
    parser.add_argument("--seq", type=str, default="v_HdiyOtliFiw_c003",
                       help="Sequence folder name")
    parser.add_argument("--frames", type=int, default=300,
                       help="Number of frames to process")
    parser.add_argument("--no-coords", action="store_true",
                       help="Don't print coordinates (faster)")
    
    args = parser.parse_args()
    
    process_sequence(
        seq_name=args.seq,
        n_frames=args.frames,
        verbose=True,
        print_coords=not args.no_coords
    )