# Third-Party Notices

hyperOSD itself is released under the MIT License (see `LICENSE`). It ships with,
or embeds, the third-party components listed below. Each keeps its own license.
Paths are relative to the repository root. Line numbers refer to the files as
shipped.

The files `hyperblend-webgl.js`, `osd-annotator.js`, `hyperpicasso.js`,
`hyperpicasso.worker.js`, the HTML viewers and the Python tools are original to
this project. `openseadragon/style.css` is original to this project except that
it carries a NIST public-domain notice at lines 1-10 (it was started from NIST
demo styles; see section 2f). The file
`openseadragon/openseadragon-filtering-nistfuncs-12ch.js` (and its `.min.js`) is
a mixed work: it bundles the third-party code listed in section 2 together with
modifications and additions made for hyperOSD (for example the `HyperBlend`
filter definition and its CPU processor).

## Summary

| # | Component | Version | License (SPDX) | Files |
|---|-----------|---------|----------------|-------|
| 1 | OpenSeadragon | 6.0.2 | BSD-3-Clause | `openseadragon/openseadragon-bin-6.0.2/` |
| 1b | Portions of OpenSeadragon 6.0.2: John Resig (jQuery), Matt Snider, Gregg Tavares (WebGL Fundamentals) | not stated | MIT; MIT; BSD-3-Clause | `openseadragon.js`, `openseadragon.min.js` |
| 2a | OpenSeadragon (embedded) | 2.1.0 | BSD-3-Clause | bundle module 8 |
| 2b | jQuery (embedded) | 2.2.0 | MIT | bundle module 4 |
| 2c | Sizzle (embedded in jQuery) | 2.2.1 | MIT | bundle module 4 |
| 2d | jQuery UI (embedded) | 1.10.3 | MIT | bundle module 5 |
| 2e | jQuery Color (embedded in jQuery UI) | 2.1.2 | MIT | bundle module 5 |
| 2f | NIST OpenSeadragonFiltering plugin and widgets | not stated | Public domain (17 U.S.C. 105) | bundle modules 1, 6, 7, 9; `openseadragon/style.css` (notice only) |
| 2g | jQuery-derived code (John Resig) and mattsnider.com code, inside OpenSeadragon 2.1.0 | not stated | MIT | bundle module 8 |
| 3 | jQuery | 2.2.0 | MIT | `openseadragon/lib/jquery-2.2.0.min.js` |
| 4 | jQuery UI (script) | 1.12.1 | MIT | `openseadragon/lib/jquery-ui-1.12.1.min.js` |
| 5 | jQuery UI (theme CSS and images) | 1.11.4 | MIT (CSS banner); images: not stated | `openseadragon/lib/jquery-ui.css`, `openseadragon/lib/images/` |
| 6 | openseadragon-scalebar | not stated | Public domain (17 U.S.C. 105) | `openseadragon/openseadragon-scalebar.js` |
| 7 | normalize.css | 3.0.0 | MIT | `openseadragon/static/css/NISTStyle.css` (line 1) |
| 8 | OpenSeadragon button images | not stated | BSD-3-Clause (as part of the OpenSeadragon distribution) | `openseadragon/openseadragon-bin-6.0.2/*.png`, `openseadragon-bin-6.0.2/images/`, `openseadragon/images/` (OpenSeadragon-style buttons) |

"Bundle" means `openseadragon/openseadragon-filtering-nistfuncs-12ch.js` and its
minified build `openseadragon/openseadragon-filtering-nistfuncs-12ch.min.js`,
which is generated from it and carries the same code. Both are shipped, but the `.min.js` strips the license comments of every
component below (verified: no `CodePlex` or `Pursuant to title 17` text remains
in it), so this file is the only place the notices travel with the minified
build. Module numbers are the `/* N */` markers in the unminified bundle.

---

## 1. OpenSeadragon 6.0.2

- Files: `openseadragon/openseadragon-bin-6.0.2/` (`openseadragon.js`,
  `openseadragon.min.js`, source maps, `changelog.txt`, `images/`, `LICENSE.txt`)
- License: BSD-3-Clause, as stated in `openseadragon-bin-6.0.2/LICENSE.txt` and in
  the header of `openseadragon.js` (header line 5: `//! License: http://openseadragon.github.io/license/`)
- Copyright: Copyright (C) 2009 CodePlex Foundation; Copyright (C) 2010-2025 OpenSeadragon contributors
- Upstream: http://openseadragon.github.io
- Also shipped here: `openseadragon.js.map`, `openseadragon.min.js.map` (source maps), `changelog.txt`.

```
Copyright (C) 2009 CodePlex Foundation
Copyright (C) 2010-2025 OpenSeadragon contributors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

- Neither the name of CodePlex Foundation nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### 1b. Portions of OpenSeadragon 6.0.2 under other licenses

`openseadragon.js` contains three further notices:

- "Portions of this source file taken from jQuery", Copyright 2011 John Resig
  (lines 42-65). MIT text is given in the file.
- "Portions of this source file taken from mattsnider.com", Copyright (c) 2006-2013
  Matt Snider (lines 67-87). MIT license text is given in the file.
- "Portions of this source file are taken from WegGL Fundamentals", Copyright 2012,
  Gregg Tavares (lines 3128-3157). BSD-3-Clause text is given in the file
  (names Gregg Tavares as the non-endorsement party).

`openseadragon.min.js` retains none of these notices, nor the BSD-3-Clause
license body (the minifier stripped them; it keeps only a short `//! License:`
URL header). The texts below
are therefore reproduced here, verbatim from `openseadragon.js`.

```
Copyright 2011 John Resig

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

```
Copyright (c) 2006-2013 Matt Snider

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
```

```
Copyright 2012, Gregg Tavares.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Gregg Tavares. nor the names of his
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
```

---

## 2. Components embedded in the NIST filter bundle

`openseadragon/openseadragon-filtering-nistfuncs-12ch.js`, a webpack bundle.
Each component below keeps the notice found in the bundle.

### 2a. OpenSeadragon 2.1.0 (module 8, bundle lines 25255-44838)

- BSD-3-Clause. Header at line 25258 (`//! openseadragon 2.1.0`), line 25262
  (`//! License: http://openseadragon.github.io/license/`).
- Copyright (C) 2009 CodePlex Foundation; Copyright (C) 2010-2013 OpenSeadragon
  contributors (lines 25267-25268). The licence text is the same BSD-3-Clause
  text as in section 1.

### 2g. Portions taken from jQuery and mattsnider.com (module 8)

- Copyright 2011 John Resig (line 25301), MIT.
- Copyright (c) 2006-2013 Matt Snider (line 25326), MIT.
  Full MIT text for both is in the bundle at lines 25299-25346.

### 2b. jQuery 2.2.0 and 2c. Sizzle 2.2.1 (module 4, lines 239-10077)

- "jQuery JavaScript Library v2.2.0 ... Copyright jQuery Foundation and other
  contributors. Released under the MIT license" (lines 243-251).
- "Sizzle CSS Selector Engine v2.2.1 ... Copyright jQuery Foundation and other
  contributors. Released under the MIT license" (lines 783-788).

### 2d. jQuery UI 1.10.3 and 2e. jQuery Color 2.1.2 (module 5, lines 10078-25095)

- "jQuery UI - v1.10.3 - 2013-05-03 ... Copyright 2013 jQuery Foundation and
  other contributors; Licensed MIT" (lines 10083-10086).
- "jQuery Color Animations v2.1.2 ... Copyright 2013 jQuery Foundation and other
  contributors. Released under the MIT license." (lines 14865-14870).

### 2f. NIST OpenSeadragonFiltering plugin and widgets (modules 1, 6, 7, 9)

Notice at the comment block whose "Pursuant to title 17" line is bundle line 62
(module 1), 25102 (module 6), 25173 (module 7) and 44845 (module 9); author line "Antoine Vandecreme
<antoine.vandecreme@nist.gov>" (line 73):

```
This software was developed at the National Institute of Standards and
Technology by employees of the Federal Government in the course of
their official duties. Pursuant to title 17 Section 105 of the United
States Code this software is not subject to copyright protection and is
in the public domain. This software is an experimental system. NIST assumes
no responsibility whatsoever for its use by other parties, and makes no
guarantees, expressed or implied, about its quality, reliability, or
any other characteristic. We would appreciate acknowledgement if the
software is used.
```

NIST asks for acknowledgement of use. The same notice appears at the top of
`openseadragon/style.css` and `openseadragon/openseadragon-scalebar.js`.
Module 1 additionally contains hyperOSD modifications (the `HyperBlend` filter
definition); those are covered by this project's MIT License.

---

## 3-4. jQuery 2.2.0 and jQuery UI 1.12.1 scripts (`openseadragon/lib/`)

- `jquery-2.2.0.min.js`, line 1: `jQuery v2.2.0 | (c) jQuery Foundation | jquery.org/license`.
  The licence is MIT: the identical release embedded in the bundle states
  "Released under the MIT license" (bundle line 250); the `.min.js` banner
  itself only points to jquery.org/license.
- `jquery-ui-1.12.1.min.js`, header: `jQuery UI - v1.12.1 - 2016-09-14 ...
  Copyright jQuery Foundation and other contributors; Licensed MIT`.

## 5. jQuery UI 1.11.4 theme (`openseadragon/lib/jquery-ui.css`, `openseadragon/lib/images/`)

- `jquery-ui.css`, lines 1-5: `jQuery UI - v1.11.4 - 2015-03-11 ... Copyright 2015
  jQuery Foundation and other contributors; Licensed MIT`.
- The PNG sprites in `lib/images/` carry no embedded notice and are referenced
  by that stylesheet. Their licence is not stated in the repository. They are
  generated by the jQuery UI ThemeRoller for this theme; treat as MIT with the
  theme (UNVERIFIED against upstream).
- Note: the CSS and scripts above are different jQuery UI releases (1.11.4 CSS,
  1.12.1 script, 1.10.3 inside the bundle).

## 6. openseadragon-scalebar (`openseadragon/openseadragon-scalebar.js`)

NIST public-domain notice at lines 1-11 (same text as section 2f); author line
"Antoine Vandecreme <antoine.vandecreme@nist.gov>" (line 15). No version is
stated in the file.

## 7. normalize.css 3.0.0 (`openseadragon/static/css/NISTStyle.css`)

Line 1: `normalize.css v3.0.0 | MIT License | git.io/normalize`. The copyright
holders are not named in the file. (Upstream normalize.css is by Nicolas Gallagher
and Jonathan Neal; UNVERIFIED in-repo.) The remainder of `NISTStyle.css` (6,897
lines) includes Bootstrap-style rules such as `.glyphicon` and an `@font-face`
for `Glyphicons Halflings` (lines 2442-2452). The font files are NOT shipped.
The provenance and licence of the rest of this stylesheet are UNVERIFIED.

## 8. OpenSeadragon button images

`openseadragon/openseadragon-bin-6.0.2/` contains OpenSeadragon's own button
images (`images/` and the PNGs at the top level), part of the BSD-3-Clause
distribution in section 1. `openseadragon/images/` holds, for 32 file names
(`fullpage_*`, `home_*`, `next_*`, `previous_*`, `rotateleft_*`, `rotateright_*`,
`zoomin_*`, `zoomout_*`), byte-identical copies (sha256) of the top-level PNGs in
`openseadragon-bin-6.0.2/`, so they are treated as the same OpenSeadragon assets.

---

## Standard MIT License text

Applies to the MIT-licensed components above. The in-repo notices for jQuery,
jQuery UI, Sizzle, jQuery Color and normalize.css name the licence but do not
reproduce its text; the standard text is reproduced here with each component's
own copyright line.

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
