# the-learning-of-isaac
An RL system learning to play the game 'The Binding of Isaac: Rebirth'.

---

## Feature Vector

### Player
(This concerns itself with the inherent player stats excluding items/trinkets)

#### HP: [0–15] (16 floats)

Storage:
- 12 floats for each potential heart
- 1 for >12 hearts (0/1)
- 1 for holy mantle (0/1)
- 1 for eternal heart (0/1)
- 1 for gold heart (0/1)

Encoding:
- 0: No heart
- 1: Full red heart
- 2: Half full red heart
- 3: Empty red heart (only container)
- 4: Soul heart
- 5: Half soul heart
- 6: Black heart
- 7: Half black heart
- 8: Empty bone heart
- 9: Rotten heart

---

#### Stats: [16–21] (6 floats)

(speed, firerate, dmg, range, shotspeed, luck)

---

#### Devil / Angel Chance: [22–23] (2 floats)

(devil chance, angel chance)  
Both chances must be equal if the player has Duality.

---

#### Collectibles (resources): [24–28] (5 floats)

Values:
- (coin, bomb, key)

Golden states:
- (has_golden_bomb (0/1), has_golden_key (0/1))

---

#### Bugs: [29–31] (3 floats)

(num_of_flies, num_of_spiders, num_of_locusts)

---

#### Pocket: [32–33] (2 floats)

Represents the cards/runes/pills the player may hold.

(first_slot, second_slot)

- First slot is 0 if it is empty, otherwise uses the ID of the object being held
- Second slot is always 0 if not unlocked

---

## Map

### Rooms Grid: [34–1047] (1014 floats)

Represents the entire floor layout as a fixed 13×13 grid (169 cells), flattened in row-major order.

For each cell, we store 6 floats.

Per-cell features (6 floats):

0: known (0/1)  
- 1 if the room is known/revealed on the minimap (or has been visited)  
- If known == 0, all remaining values for this cell are forced to 0

1: room_type (label-encoded float)  
- Encodes the room’s type using the game’s RoomType enum value  
  (e.g., default/shop/treasure/boss/curse/etc.)  
- -1 is reserved for “empty/no room”

2: visited (0/1)  
- 1 if the room has been entered at least once on this floor

3: current_room (0/1)  
- 1 if this cell corresponds to the player’s current room

4: has_unopened_door (0/1)  
- 1 if the room has at least one exit leading to a room that exists but has not been visited yet  
  (i.e., there is still an unexplored path from this room)

5: distance_to_current (>= 0)  
- Manhattan distance (in grid cells) from this room’s grid position to the current room’s grid position

Flattening / indexing:
- Each room cell contributes 6 floats
- Cells are ordered by grid index 0..168 (row-major over a 13×13 grid)

idx = y * 13 + x, with x in [0..12], y in [0..12]

---

## Items

### Items: [1048–1815] (768 floats)

---

#### Collectibles (multi-hot): [1048–1797] (750 floats)

Represents all collectible items (passive + active) by ID.

- Index (1047 + i) corresponds to Collectible ID = i, for i in [1, 750]
- Value:
  - 1.0 if the player currently owns at least one copy of the collectible
  - 0.0 otherwise

This encoding is maintained via a cache:
- Updated immediately on item pickup
- Fully resynchronized if items are removed, rerolled, or dropped

---

#### Active Item ID: [1798] (1 float)

- The ID of the currently held active item
- 0 if no active item is held

---

#### Active Item Charge (normalized): [1799] (1 float)

- The current charge of the active item, normalized to [0, 1]
- Defined as (current_charge / max_charge)
- If no active item is held or the item has no charge, the value is 0

---

#### Trinket ID: [1800] (1 float)

- The ID of the trinket held in slot 0
- 0 if no trinket is held

---

#### Transformations (multi-hot): [1801–1815] (15 floats)

Represents permanent player transformations (PlayerForm enum).

- One float per transformation
- Value:
  - 1.0 if the player currently has the transformation
  - 0.0 otherwise