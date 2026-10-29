# Data

The HASTIKA files are not part of this repository (the shared task terms do not allow redistribution).

Download them from the organizers' repository, <https://github.com/shankarb14/Hastika-ICON2026>, and place them here.
The scripts expect, for example:

| File | Columns |
|---|---|
| `binary_train.csv` | `id, Comment, Label` |
| `binary_test.csv` | `id, Comment` (and the gold `Label` once released) |
| `multiclass_train.csv` | `id, Comment, Hate Category` |
| `multiclass_test.csv` | `id, Comment` (and the gold `Hate Category` once released) |

You can use any file names, since every script takes the paths as arguments. The files are read as UTF-8
with a possible byte order mark.
