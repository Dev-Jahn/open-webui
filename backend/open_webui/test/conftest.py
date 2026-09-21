"""Point Open WebUI at a throwaway data directory before any `open_webui` module is imported.

Importing `open_webui.config` creates DATA_DIR and runs the DB migrations there.
"""

import os
import tempfile

os.environ.setdefault('DATA_DIR', tempfile.mkdtemp(prefix='open-webui-test-'))
os.environ.setdefault('WEBUI_SECRET_KEY', 'test-secret')
