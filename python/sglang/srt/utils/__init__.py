# Temporarily do this to avoid changing all imports in the repo
from sglang.srt.utils.common import *

# Network helpers were split out of common into network.py. Re-export here so
# downstream consumers pinned to the older `sglang.srt.utils` import path
# (e.g. smg-grpc-servicer's request_manager.py) keep resolving symbols like
# `get_zmq_socket`, `get_local_ip_auto`, `NetworkAddress`. Drop this once the
# downstreams have migrated to `sglang.srt.utils.network`.
from sglang.srt.utils.network import *  # noqa: F401, F403
