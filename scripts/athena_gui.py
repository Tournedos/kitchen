#!/usr/bin/env python3
"""Launch the ATHENA GUI with enlarged Tk widgets/fonts (HiDPI screens).

Usage:  python scripts/athena_gui.py [scale]      (default scale 2.0, try 1.5-2.5)
"""
import sys
import tkinter as tk
import tkinter.font as tkfont

SCALE = float(sys.argv.pop(1)) if len(sys.argv) > 1 else 2.0

_orig_init = tk.Tk.__init__


def _scaled_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.tk.call('tk', 'scaling', SCALE * 96 / 72)  # Tk scaling = pixels per point
    for name in ('TkDefaultFont', 'TkTextFont', 'TkFixedFont', 'TkMenuFont',
                 'TkHeadingFont', 'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont'):
        try:
            f = tkfont.nametofont(name, root=self)
            size = f.cget('size')
            if size < 0:  # pixel-sized fonts ignore 'tk scaling': scale them explicitly
                f.configure(size=-int(round(-size * SCALE)))
        except tk.TclError:
            pass


tk.Tk.__init__ = _scaled_init

from athena.athena import main  # noqa: E402

sys.exit(main())
