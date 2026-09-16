# Dataset Contract

SeaDronesSee is not stored in this repository.

Before any experiment, record the following in an untracked local manifest:

- official download page and access date;
- archive filenames, byte sizes, and SHA-256 values;
- annotation filenames and SHA-256 values;
- image and annotation counts per split;
- category IDs and names;
- metadata fields actually present in Object Detection v2;
- source sequence fields available for leakage-safe partitioning;
- license and required citation.

The project currently expects COCO-style `images`, `annotations`, and `categories`. Metadata may
be nested under each image's `meta` dictionary or exposed directly on the image record. The
validator reports what is actually present rather than silently inventing missing values.

Important schema warning: top-level `height` is image height in pixels. Flight altitude is stored
under `meta.altitude` or `meta.height_above_takeoff(meter)`. Gimbal pitch is stored under
`meta.gimbal_pitch` or `meta.gimbal_pitch(degrees)`.

Official test labels are withheld. The official validation set is therefore the frozen final
evaluation set for the first release.

The public Nextcloud ZIP endpoint creates an on-demand streaming archive and does not advertise
reliable byte-range resume support. `scripts/download_official_images.py` therefore uses the
official public WebDAV listing, checks the exact annotation filename set and byte length of every
JPEG, downloads independent files concurrently, and atomically finalizes each completed image.
The generated WebDAV manifests make interrupted downloads safely resumable without trusting a
partial multi-gigabyte ZIP.

The official category with ID `0` is named `ignored` and is not a detector class. Conversion
must map active COCO IDs `1..5` to contiguous YOLO class indices `0..4`; this repository's
converter records that mapping in a manifest. Derived YOLO labels live beside, but never replace,
the official image folders.
