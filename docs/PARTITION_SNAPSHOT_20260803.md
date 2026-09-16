# Training Partition Snapshot — 2026-08-03

Source annotation SHA-256:
`5706B042F6B6C50267ED06076B8C27594FF4DA76569EDBD99ED53ED8BFA1DB95`

The 8,930 official training images are assigned by indivisible acquisition group. A video is the
preferred group key. The 1,554 Trinity RGB/multispectral images have no video key and are kept in
one conservative `source.drone:trinity` group to prevent cross-sensor leakage.

- Base seed: `20260803`
- Selected deterministic candidate seed: `20260842`
- Search offset: `39`
- Source groups: `38`
- Constraint: every role has at least 25 boxes from every active class
- Overlap: zero images and zero source groups

| Role | Groups | Images | Boxes | Swimmer | Boat | Jetski | Life-saving appliances | Buoy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| detector-train | 20 | 6,251 | 41,485 | 29,003 | 7,582 | 1,551 | 422 | 2,927 |
| detector-dev | 7 | 893 | 5,437 | 3,115 | 1,533 | 306 | 64 | 419 |
| calibration-fit | 6 | 893 | 6,041 | 2,437 | 2,454 | 95 | 364 | 691 |
| policy-tune | 5 | 893 | 4,797 | 2,541 | 1,453 | 378 | 73 | 352 |

The split is generated, not hand-picked after model results. `partition_training_coco.py` searches
only the official training annotations for a class-coverage-valid candidate and records the chosen
offset. The official validation set is not consulted during partitioning.
