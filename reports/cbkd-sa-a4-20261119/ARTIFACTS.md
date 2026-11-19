# Drive backup of stage artifacts

Destination: Drive folder "Self-Audit" (`1KLYuulOT6riQjeF1oarmeBDmwcsD3aEX`), `pg-artifacts-20261119/{S1,S2,S3-SA,A4-SA}/`.
Each stage folder has a `SHA256SUMS`; uploads are verified with `rclone check` and sizes. Drive file IDs are recorded in
`<run>/markers/<stage>.UPLOADED.json`, merged into `<run>/markers/<stage>.DONE.json` (`drive` key) and appended to
`<run>/ARTIFACTS.md` on the volume.

Status: the upload path was tested end to end with a small file (3 files verified on Drive, IDs recorded, test folder removed).
The currently running A4-SA job started before this change and therefore does **not** upload; relaunching the same
command picks it up (finished stages are skipped by their DONE markers).
