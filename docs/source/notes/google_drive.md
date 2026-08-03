# Google Drive file access

RESource can expose a Google Drive remote as a read-only local filesystem and
resolve input data by filename. The integration uses
[rclone](https://rclone.org/drive/) so OAuth credentials remain in rclone's local
configuration rather than in the repository.

First install rclone and create a Drive remote interactively:

```bash
rclone config
rclone lsd gdrive:
```

Do not commit the rclone configuration or OAuth tokens. Then mount the remote for
the lifetime of a Python block:

```python
from pathlib import Path

import pandas as pd

from RESource.google_drive import GoogleDriveMount

mount = GoogleDriveMount("gdrive:RESource/input", Path("data/google-drive"))
with mount as drive:
    weather = drive.path("weather_2024.csv")
    frame = pd.read_csv(weather)
```

Filename lookup is recursive. Because Google Drive permits duplicate names,
`drive.path("results.csv")` raises `AmbiguousDriveFileError` when more than one
file matches. Pass a path relative to the mounted root to select one explicitly:

```python
results = drive.path("baseline/BC/results.csv")
all_results = drive.find_all("results.csv")
```

The default mount is read-only. Enable writes only for a controlled workflow with
`GoogleDriveMount(..., read_only=False)`; generated regional results should still
use scenario-specific paths and must not overwrite comparison evidence silently.
