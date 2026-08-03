# Local credentials

This directory is the canonical location for personal credentials used by a
RESource checkout. Real credential files are local-only and must never be committed,
attached to an issue, pasted into a notebook, or included in logs.

## CODERS API

1. Obtain authorized access to the private modeling inventory.
2. Download
   [`coders_api.yaml`](https://github.com/eliasinul/modeling_inventory/blob/main/PyPSA/coders_api.yaml).
3. Place it at exactly:

   ```text
   credentials/coders_api.yaml
   ```

4. Keep the following structure:

   ```yaml
   api_keys:
     - your_api_key  # optional local note
   ```

5. Run the Canadian workflow normally. RESource reads this path from the `CODERS`
   configuration section.

Use `coders_api.example.yaml` as a schema reference only. Do not put a real key in
the example. The real filename is explicitly ignored by Git.
