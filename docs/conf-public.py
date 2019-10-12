# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Inherit everything from the local config
from conf import *  # isort:skip

OUTPUT = "../../habitat-sim/build/docs-public/habitat-api/"

HTML_HEADER = """<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-66408458-4"></script>
<script>
 window.dataLayer = window.dataLayer || [];
 function gtag(){dataLayer.push(arguments);}
 gtag('js', new Date());
 gtag('config', 'UA-66408458-4');
</script>
"""

SEARCH_DOWNLOAD_BINARY = "searchdata-v1.bin"
SEARCH_BASE_Url = "https://aihabitat.org/docs/habitat-api/"
SEARCH_EXTERNAL_Url = "https://google.com/search?q=site:aihabitat.org+{query}"
